"""Causal online SARIMAX + soft-tree ensemble trained with an EKF.

The EKF state contains the time-varying SARIMAX coefficients followed by the
parameters of every soft decision tree. The process model is an identity
(random-walk) model. Consequently, process noise is added to the predicted
state covariance; it must not replace the estimated state.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np


MEAS_NOISE_STD = 1.0
UPD_NOISE_STD = 1.0


# Explicit, deliberately wide search spaces.  Unlike LightGBM, this model has
# full binary soft trees, so the leaf count is fixed at 2**depth.  Parameters
# such as subsample, colsample_bytree, max_bin, and L1/L2 penalties are not part
# of the manuscript's EKF/SGBDT formulation and are therefore not invented here.
SGBDT_SEARCH_SPACE = {
    "n_estimators": (1, 2, 3, 5, 10, 25, 50, 75, 100, 150, 250, 500),
    "learning_rate": (
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.2,
        0.5,
    ),
    "depth": (1, 2, 3, 4, 5, 6, 7, 8),
}

SARIMAX_SEARCH_SPACE = {
    "p": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15),
    "d": (0, 1, 2),
    "q": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15),
    "seasonal_p": (0, 1, 2, 3, 4, 5, 6, 7, 8),
    "seasonal_d": (0, 1, 2),
    "seasonal_q": (0, 1, 2, 3, 4, 5, 6, 7, 8),
}

EKF_SEARCH_SPACE = {
    "process_noise_std": (
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.3,
        0.5,
        1.0,
    ),
    "measurement_noise_std": (
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.3,
        0.5,
        1.0,
    ),
}


def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass(frozen=True)
class _TreeSlices:
    weights: slice
    biases: slice
    leaves: slice


class SX_sGBDT:
    """Joint SARIMAX/soft-tree state-space model.

    ``learning_rate`` is the manuscript's lambda_e and may be a scalar or one
    value per tree. Differencing is causal; forecasts are converted back to the
    original observation scale before the EKF innovation is computed.
    """

    def __init__(
        self,
        n_estimator: int = 1,
        depth: int = 1,
        sx_order: tuple[int, int, int] = (1, 0, 0),
        sx_seas_order: tuple[int, int, int, int] | None = None,
        exog=None,
        n_data=None,
        learning_rate: float | Sequence[float] = 1.0,
        process_noise_std: float = UPD_NOISE_STD,
        measurement_noise_std: float = MEAS_NOISE_STD,
        random_state: int | None = None,
    ):
        self.n_data = n_data
        self.sx_p, self.sx_d, self.sx_q = map(int, sx_order)
        if min(self.sx_p, self.sx_d, self.sx_q) < 0:
            raise ValueError("sx_order entries must be non-negative")

        if sx_seas_order is None:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = 0, 0, 0, None
        else:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = map(int, sx_seas_order)
            if min(self.sx_P, self.sx_D, self.sx_Q) < 0 or self.sx_m <= 0:
                raise ValueError("seasonal orders must be non-negative and s must be positive")
        if (self.sx_P or self.sx_D or self.sx_Q) and self.sx_m is None:
            raise ValueError("a seasonal period is required for seasonal terms")

        self.exog = None if exog is None else np.asarray(exog, dtype=float)
        if self.exog is not None and self.exog.ndim != 2:
            raise ValueError("exog must be a two-dimensional array")
        self.n_features = 0 if self.exog is None else self.exog.shape[1]
        self._has_sgbdt = self.n_features > 0

        self.n_estimators = int(n_estimator)
        self.depth = int(depth)
        if self._has_sgbdt and self.n_estimators < 1:
            raise ValueError("n_estimator must be positive when exog is supplied")
        if self._has_sgbdt and self.depth < 1:
            raise ValueError("depth must be at least one when exog is supplied")

        if not self._has_sgbdt:
            if exog is None:
                warnings.warn("No exogenous variables passed; SGBDT is inactive", stacklevel=2)
            self.n_estimators = 0
            self.n_internal = 0
            self.n_leaves = 0
            self.n_node = 0
            self.leaf_node = 0
            self.innder_node = 0
            self.learning_rates = np.empty(0, dtype=float)
        else:
            self.n_internal = 2**self.depth - 1
            self.n_leaves = 2**self.depth
            self.n_node = self.n_internal + self.n_leaves
            self.leaf_node = self.n_leaves
            self.innder_node = self.n_internal  # Backward-compatible typo.
            rates = np.asarray(learning_rate, dtype=float)
            if rates.ndim == 0:
                rates = np.full(self.n_estimators, float(rates))
            if rates.shape != (self.n_estimators,):
                raise ValueError("learning_rate must be scalar or have one value per estimator")
            self.learning_rates = rates

        # [AR, MA, seasonal AR, seasonal MA, exogenous coefficients].
        # The manuscript's constant is fixed to c = 0.
        self.sx_size = (
            self.sx_p
            + self.sx_q
            + self.sx_P
            + self.sx_Q
            + self.n_features
        )

        # Per manuscript: each tree has internal-node weights and biases, then
        # one scalar delta for each leaf.
        self._tree_slices: list[_TreeSlices] = []
        offset = self.sx_size
        for _ in range(self.n_estimators):
            weight_count = self.n_internal * self.n_features
            weight_slice = slice(offset, offset + weight_count)
            offset = weight_slice.stop
            bias_slice = slice(offset, offset + self.n_internal)
            offset = bias_slice.stop
            leaf_slice = slice(offset, offset + self.n_leaves)
            offset = leaf_slice.stop
            self._tree_slices.append(_TreeSlices(weight_slice, bias_slice, leaf_slice))

        self.state_dim = offset
        self._rng = np.random.default_rng(random_state)
        self.state_vector = self._rng.normal(size=self.state_dim)

        self.G_t = np.eye(self.state_dim, dtype=float)
        self.C_t = (float(process_noise_std) ** 2) * np.eye(self.state_dim, dtype=float)
        self.I = np.eye(self.state_dim, dtype=float)
        self.P_t = np.zeros((1, self.state_dim), dtype=float)
        self.Z_t_1 = 0.1 * np.eye(self.state_dim, dtype=float)
        self.Q_t = np.array([[float(measurement_noise_std) ** 2]], dtype=float)

        self._measurements: list[float] = []
        self._transformed_measurements: list[float] = []
        self._errors: list[float] = []
        self._predictions: list[float] = []
        self._difference_coefficients = self._make_difference_coefficients()

    def _make_difference_coefficients(self) -> np.ndarray:
        coefficients = np.array([1.0])
        for _ in range(self.sx_d):
            coefficients = np.convolve(coefficients, np.array([1.0, -1.0]))
        for _ in range(self.sx_D):
            seasonal = np.zeros(self.sx_m + 1, dtype=float)
            seasonal[0], seasonal[-1] = 1.0, -1.0
            coefficients = np.convolve(coefficients, seasonal)
        return coefficients

    def _sx_state_transition(self):
        """Identity transition for the SARIMAX parameter block."""
        return self.state_vector[: self.sx_size].copy()

    def _sgbdt_state_transition(self):
        """Identity transition for the soft-tree parameter block."""
        return self.state_vector[self.sx_size :].copy()

    def _state_transition(self, t=None):
        """Return the random-walk predicted mean without sampling process noise."""
        del t
        return self.state_vector.copy()

    def get_state(self):
        return self.state_vector

    def put_state(self, state_vector):
        state_vector = np.asarray(state_vector, dtype=float)
        if state_vector.shape != (self.state_dim,):
            raise ValueError(f"state_vector must have shape ({self.state_dim},)")
        self.state_vector = state_vector.copy()

    def get_state_cov(self):
        return self.Z_t_1

    def put_state_cov(self, covariance):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (self.state_dim, self.state_dim):
            raise ValueError("state covariance has the wrong shape")
        self.Z_t_1 = covariance.copy()

    @staticmethod
    def _lagged(values: Sequence[float], lag: int) -> float:
        return float(values[-lag]) if len(values) >= lag else 0.0

    def _resolve_x(self, t: int, x_t=None) -> np.ndarray:
        if not self._has_sgbdt:
            return np.empty(0, dtype=float)
        if x_t is None:
            if self.exog is None or not 0 <= t < len(self.exog):
                raise ValueError("x_t is required for this time step")
            x_t = self.exog[t]
        x_t = np.asarray(x_t, dtype=float)
        if x_t.shape != (self.n_features,):
            raise ValueError(f"x_t must have shape ({self.n_features},)")
        return x_t

    def _sx_features(self, x_t: np.ndarray) -> np.ndarray:
        features: list[float] = []
        features.extend(self._lagged(self._transformed_measurements, lag) for lag in range(1, self.sx_p + 1))
        features.extend(self._lagged(self._errors, lag) for lag in range(1, self.sx_q + 1))
        features.extend(
            self._lagged(self._transformed_measurements, self.sx_m * lag)
            for lag in range(1, self.sx_P + 1)
        )
        features.extend(self._lagged(self._errors, self.sx_m * lag) for lag in range(1, self.sx_Q + 1))
        features.extend(x_t.tolist())
        result = np.asarray(features, dtype=float)
        if result.shape != (self.sx_size,):
            raise RuntimeError("internal SARIMAX feature/state dimension mismatch")
        return result

    def _inverse_difference_offset(self) -> float:
        # w_t = y_t + sum_{k>0} a_k y_{t-k}; hence y_t = w_t - ...
        return -sum(
            coefficient * self._lagged(self._measurements, lag)
            for lag, coefficient in enumerate(self._difference_coefficients[1:], start=1)
        )

    def _append_observation(self, y_t: float, innovation: float) -> None:
        y_t = float(y_t)
        sequence = [*self._measurements, y_t]
        index = len(sequence) - 1
        transformed = sum(
            coefficient * sequence[index - lag]
            for lag, coefficient in enumerate(self._difference_coefficients)
            if index >= lag
        )
        self._measurements.append(y_t)
        self._transformed_measurements.append(float(transformed))
        self._errors.append(float(innovation))

    def _tree_output_and_gradient(self, state, tree_slices: _TreeSlices, x_t):
        weights = state[tree_slices.weights].reshape(self.n_internal, self.n_features)
        biases = state[tree_slices.biases]
        leaves = state[tree_slices.leaves]

        routing = sigmoid(weights @ x_t + biases)
        reach = np.zeros(self.n_node, dtype=float)
        reach[0] = 1.0
        for node in range(self.n_internal):
            left, right = 2 * node + 1, 2 * node + 2
            reach[left] = reach[node] * routing[node]
            reach[right] = reach[node] * (1.0 - routing[node])

        subtree_value = np.empty(self.n_node, dtype=float)
        subtree_value[self.n_internal :] = leaves
        for node in range(self.n_internal - 1, -1, -1):
            left, right = 2 * node + 1, 2 * node + 2
            subtree_value[node] = (
                routing[node] * subtree_value[left]
                + (1.0 - routing[node]) * subtree_value[right]
            )

        activation_gradient = np.empty(self.n_internal, dtype=float)
        for node in range(self.n_internal):
            left, right = 2 * node + 1, 2 * node + 2
            activation_gradient[node] = (
                reach[node]
                * routing[node]
                * (1.0 - routing[node])
                * (subtree_value[left] - subtree_value[right])
            )

        gradient = np.concatenate(
            (
                (activation_gradient[:, None] * x_t[None, :]).ravel(),
                activation_gradient,
                reach[self.n_internal :],
            )
        )
        return float(subtree_value[0]), gradient

    def _measurement_model(self, x_t, t: int, state_vector=None, return_jacobian=False):
        state = self.state_vector if state_vector is None else np.asarray(state_vector, dtype=float)
        if state.shape != (self.state_dim,):
            raise ValueError(f"state_vector must have shape ({self.state_dim},)")
        x_t = self._resolve_x(t, x_t)
        sx_features = self._sx_features(x_t)

        forecast = self._inverse_difference_offset() + float(state[: self.sx_size] @ sx_features)
        jacobian = np.zeros(self.state_dim, dtype=float)
        jacobian[: self.sx_size] = sx_features

        for rate, tree_slices in zip(self.learning_rates, self._tree_slices):
            tree_output, tree_gradient = self._tree_output_and_gradient(state, tree_slices, x_t)
            forecast += float(rate) * tree_output
            jacobian[tree_slices.weights.start : tree_slices.leaves.stop] = float(rate) * tree_gradient

        if return_jacobian:
            return float(forecast), jacobian.reshape(1, -1)
        return float(forecast)

    def _sx_prediction(self, t, x_t=None, state_vector=None):
        state = self.state_vector if state_vector is None else np.asarray(state_vector, dtype=float)
        x_t = self._resolve_x(t, x_t)
        return self._inverse_difference_offset() + float(state[: self.sx_size] @ self._sx_features(x_t))

    def _sgbdt_prediction(self, x_t, state_vector=None):
        if not self._has_sgbdt:
            return 0.0
        state = self.state_vector if state_vector is None else np.asarray(state_vector, dtype=float)
        x_t = np.asarray(x_t, dtype=float)
        prediction = 0.0
        for rate, tree_slices in zip(self.learning_rates, self._tree_slices):
            tree_output, _ = self._tree_output_and_gradient(state, tree_slices, x_t)
            prediction += float(rate) * tree_output
        return float(prediction)

    def prediction(self, h_t=None, x_t=None, z_t=None, t=0):
        """Compute and store a causal pre-update prediction.

        ``h_t`` is retained for backward compatibility; the causal regressor is
        constructed internally from observations available through ``t-1``.
        """
        del h_t
        forecast = self._measurement_model(x_t, t, z_t, return_jacobian=False)
        self._predictions.append(forecast)
        return forecast


def ekf(
    model: SX_sGBDT,
    y_t: float,
    x_t=None,
    t: int = 0,
    dt=1,
    return_prediction_time: bool = False,
):
    """Run one causal EKF prediction/observation/update cycle.

    When requested, the returned prediction time covers only work performed
    before ``y_t`` is observed. The target-dependent EKF update is excluded.
    """
    del dt

    prediction_started_at = perf_counter() if return_prediction_time else None

    # Identity random-walk prediction. Zero-mean process noise affects the
    # covariance, not the state mean.
    z_prior = model._state_transition(t)
    # G_t is the identity for the random-walk process.  Exploiting that fact is
    # important during hyperparameter search because dense identity matrix
    # products would otherwise turn this O(V^2) update into O(V^3).
    covariance_prior = model.get_state_cov() + model.C_t

    forecast_y, measurement_jacobian = model._measurement_model(
        x_t, t, z_prior, return_jacobian=True
    )
    model.P_t = measurement_jacobian
    model._predictions.append(forecast_y)

    prediction_time = (
        perf_counter() - prediction_started_at
        if prediction_started_at is not None
        else None
    )

    innovation = float(y_t) - forecast_y
    innovation_covariance = measurement_jacobian @ covariance_prior @ measurement_jacobian.T + model.Q_t
    innovation_variance = float(innovation_covariance[0, 0])
    if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
        raise np.linalg.LinAlgError("innovation covariance must be finite and positive")

    kalman_gain = covariance_prior @ measurement_jacobian.T / innovation_variance
    z_posterior = z_prior + kalman_gain[:, 0] * innovation

    # Algebraically expanded Joseph update.  Since the observation is scalar,
    # this rank-one form is equivalent but avoids dense V x V matrix products.
    gain_times_cross_covariance = kalman_gain @ (
        measurement_jacobian @ covariance_prior
    )
    covariance_posterior = (
        covariance_prior
        - gain_times_cross_covariance
        - gain_times_cross_covariance.T
        + innovation_variance * (kalman_gain @ kalman_gain.T)
    )
    covariance_posterior = 0.5 * (covariance_posterior + covariance_posterior.T)

    model.put_state(z_posterior)
    model.put_state_cov(covariance_posterior)

    # The current target becomes history only after its pre-update prediction
    # and innovation have been computed.
    model._append_observation(float(y_t), innovation)
    if return_prediction_time:
        return forecast_y, prediction_time
    return forecast_y


def EKF_SGBDT(y_obs: Iterable[float], model: SX_sGBDT):
    """Process observations sequentially and return causal one-step forecasts."""
    observations = np.asarray(list(y_obs), dtype=float)
    if model.exog is not None and len(model.exog) < len(observations):
        raise ValueError("exog has fewer rows than y_obs")

    forecasts = np.empty(len(observations), dtype=float)
    for t, y_t in enumerate(observations):
        x_t = None if model.exog is None else model.exog[t]
        forecasts[t] = ekf(model=model, y_t=y_t, x_t=x_t, t=t)
    return forecasts


def _synthetic_example(n_samples: int, n_features: int, seed: int):
    """Create a reproducible causal nonlinear time-series example."""
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3")
    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    generator = np.random.default_rng(seed)
    features = generator.normal(size=(n_samples, n_features))
    target = np.zeros(n_samples, dtype=float)
    for t in range(1, n_samples):
        nonlinear_effect = 0.35 * np.sin(features[t, 0])
        linear_effect = -0.2 * features[t, 1] if n_features > 1 else 0.0
        target[t] = (
            0.65 * target[t - 1]
            + nonlinear_effect
            + linear_effect
            + generator.normal(scale=0.05)
        )
    return target, features


def _load_csv_data(csv_path: str, target_name: str, feature_names: Sequence[str]):
    """Load numeric target/features while preserving the file's row order."""
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - environment dependent.
        raise RuntimeError("pandas is required when --csv is used") from error

    frame = pd.read_csv(csv_path)
    if target_name not in frame.columns:
        raise ValueError(f"target column {target_name!r} was not found in {csv_path}")
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"feature columns were not found: {missing}")
    if target_name in feature_names:
        raise ValueError("the current target column cannot also be an input feature")

    target = frame[target_name].to_numpy(dtype=float)
    features = (
        frame[list(feature_names)].to_numpy(dtype=float)
        if feature_names
        else None
    )
    if not np.isfinite(target).all():
        raise ValueError("target contains missing or non-finite values")
    if features is not None and not np.isfinite(features).all():
        raise ValueError("features contain missing or non-finite values")
    return target, features


@dataclass(frozen=True)
class HyperParameters:
    """One jointly tuned SGBDT + SARIMAX configuration."""

    n_estimators: int
    depth: int
    sx_order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    learning_rate: float
    process_noise_std: float
    measurement_noise_std: float


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    mse: float
    mape: float
    mase: float
    nmae: float


def _forecast_metrics(
    actual: np.ndarray,
    forecasts: np.ndarray,
    scaling_target: np.ndarray,
    mase_lag: int,
    *,
    emit_warnings: bool = False,
) -> ForecastMetrics:
    """Calculate errors using only the supplied earlier training data for scaling."""
    actual = np.asarray(actual, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    scaling_target = np.asarray(scaling_target, dtype=float)
    if actual.shape != forecasts.shape or actual.ndim != 1:
        raise ValueError("actual and forecasts must be equal-length one-dimensional arrays")
    if len(actual) == 0:
        raise ValueError("at least one forecast is required")
    if len(scaling_target) <= mase_lag:
        raise ValueError("the scaling training set is too short for the MASE lag")

    errors = actual - forecasts
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))

    nonzero = np.abs(actual) > np.finfo(float).eps
    if emit_warnings and not nonzero.all():
        warnings.warn(
            "zero evaluation targets were excluded from MAPE because percentage error is undefined",
            stacklevel=2,
        )
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual[nonzero])))
        if nonzero.any()
        else np.nan
    )

    naive_scale = float(
        np.mean(
            np.abs(
                scaling_target[mase_lag:]
                - scaling_target[:-mase_lag]
            )
        )
    )
    if naive_scale <= np.finfo(float).eps:
        if emit_warnings:
            warnings.warn("training naive scale is zero; MASE is undefined", stacklevel=2)
        mase = np.nan
    else:
        mase = mae / naive_scale

    # Equation/definition in the manuscript: test MAE divided by the range
    # that would map the training targets to [0, 1].
    training_range = float(np.max(scaling_target) - np.min(scaling_target))
    if training_range <= np.finfo(float).eps:
        if emit_warnings:
            warnings.warn("training target range is zero; nMAE is undefined", stacklevel=2)
        nmae = np.nan
    else:
        nmae = mae / training_range

    return ForecastMetrics(mae=mae, mse=mse, mape=mape, mase=mase, nmae=nmae)


def _make_model(
    parameters: HyperParameters,
    features: np.ndarray | None,
    seed: int,
) -> SX_sGBDT:
    return SX_sGBDT(
        n_estimator=parameters.n_estimators,
        depth=parameters.depth,
        sx_order=parameters.sx_order,
        sx_seas_order=parameters.seasonal_order,
        exog=features,
        learning_rate=parameters.learning_rate,
        process_noise_std=parameters.process_noise_std,
        measurement_noise_std=parameters.measurement_noise_std,
        random_state=seed,
    )


def _state_dimension(parameters: HyperParameters, n_features: int) -> int:
    p, _, q = parameters.sx_order
    seasonal_p = seasonal_q = 0
    if parameters.seasonal_order is not None:
        seasonal_p, _, seasonal_q, _ = parameters.seasonal_order
    sarimax_size = (
        p
        + q
        + seasonal_p
        + seasonal_q
        + n_features
    )
    if n_features == 0:
        return sarimax_size
    internal = 2**parameters.depth - 1
    leaves = 2**parameters.depth
    per_tree = internal * n_features + internal + leaves
    return sarimax_size + parameters.n_estimators * per_tree


def _anchor_parameters(args, has_features: bool) -> HyperParameters:
    return HyperParameters(
        n_estimators=args.n_estimators if has_features else 0,
        depth=args.depth if has_features else 1,
        sx_order=tuple(args.sx_order),
        seasonal_order=(
            None if args.seasonal_order is None else tuple(args.seasonal_order)
        ),
        learning_rate=args.learning_rate if has_features else 1.0,
        process_noise_std=args.process_noise_std,
        measurement_noise_std=args.measurement_noise_std,
    )


def _draw_parameters(rng: np.random.Generator, args, has_features: bool) -> HyperParameters:
    """Draw from a deliberately broad, manuscript-aligned random-search space."""
    if has_features:
        n_estimators = int(rng.choice(SGBDT_SEARCH_SPACE["n_estimators"]))
        depth = int(rng.choice(SGBDT_SEARCH_SPACE["depth"]))
        learning_rate = float(rng.choice(SGBDT_SEARCH_SPACE["learning_rate"]))
    else:
        n_estimators, depth, learning_rate = 0, 1, 1.0

    sx_order = (
        int(rng.choice(SARIMAX_SEARCH_SPACE["p"])),
        int(rng.choice(SARIMAX_SEARCH_SPACE["d"])),
        int(rng.choice(SARIMAX_SEARCH_SPACE["q"])),
    )

    seasonal_periods = list(args.seasonal_periods)
    if args.seasonal_order is not None and args.seasonal_order[3] not in seasonal_periods:
        seasonal_periods.append(args.seasonal_order[3])

    seasonal_order = None
    if seasonal_periods and rng.random() < 0.75:
        period = int(rng.choice(seasonal_periods))
        seasonal_order = (
            int(rng.choice(SARIMAX_SEARCH_SPACE["seasonal_p"])),
            int(rng.choice(SARIMAX_SEARCH_SPACE["seasonal_d"])),
            int(rng.choice(SARIMAX_SEARCH_SPACE["seasonal_q"])),
            period,
        )
        if seasonal_order[:3] == (0, 0, 0):
            seasonal_order = None

    return HyperParameters(
        n_estimators=n_estimators,
        depth=depth,
        sx_order=sx_order,
        seasonal_order=seasonal_order,
        learning_rate=learning_rate,
        process_noise_std=float(rng.choice(EKF_SEARCH_SPACE["process_noise_std"])),
        measurement_noise_std=float(
            rng.choice(EKF_SEARCH_SPACE["measurement_noise_std"])
        ),
    )


def _validation_trial(
    parameters: HyperParameters,
    target: np.ndarray,
    features: np.ndarray | None,
    train_end: int,
    validation_end: int,
    mase_lag: int,
    seed: int,
) -> ForecastMetrics:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No exogenous variables passed; SGBDT is inactive",
        )
        model = _make_model(parameters, features, seed)
    validation_forecasts = np.empty(validation_end - train_end, dtype=float)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for t in range(train_end):
            x_t = None if features is None else features[t]
            ekf(model=model, y_t=target[t], x_t=x_t, t=t)
        for output_index, t in enumerate(range(train_end, validation_end)):
            x_t = None if features is None else features[t]
            validation_forecasts[output_index] = ekf(
                model=model,
                y_t=target[t],
                x_t=x_t,
                t=t,
            )
    if not np.isfinite(validation_forecasts).all():
        raise FloatingPointError("non-finite validation forecast")
    return _forecast_metrics(
        target[train_end:validation_end],
        validation_forecasts,
        target[:train_end],
        mase_lag,
    )


def _write_search_results(path: str, results: list[dict]) -> None:
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def _random_search(
    args,
    target: np.ndarray,
    features: np.ndarray | None,
    train_end: int,
    validation_end: int,
) -> tuple[HyperParameters, ForecastMetrics, list[dict], float]:
    """Select exclusively by chronological validation MASE; never inspect test targets."""
    rng = np.random.default_rng(args.seed)
    has_features = features is not None and features.shape[1] > 0
    n_features = 0 if features is None else features.shape[1]
    requested_trials = 1 if args.no_search else args.search_iterations
    anchor = _anchor_parameters(args, has_features)

    seen: set[HyperParameters] = set()
    results: list[dict] = []
    best_parameters = None
    best_metrics = None
    attempts = 0
    maximum_attempts = max(100, requested_trials * 200)
    search_started_at = perf_counter()

    while len(results) < requested_trials and attempts < maximum_attempts:
        parameters = anchor if attempts == 0 else _draw_parameters(rng, args, has_features)
        attempts += 1
        if parameters in seen:
            continue
        seen.add(parameters)
        state_dimension = _state_dimension(parameters, n_features)
        if state_dimension == 0 or state_dimension > args.max_state_dim:
            if args.no_search:
                raise ValueError(
                    "the anchor model has an invalid or over-limit state dimension; "
                    "change its orders or increase --max-state-dim"
                )
            continue

        trial_started_at = perf_counter()
        status = "ok"
        try:
            metrics = _validation_trial(
                parameters,
                target,
                features,
                train_end,
                validation_end,
                args.mase_lag,
                args.seed,
            )
            if not np.isfinite(metrics.mase):
                raise FloatingPointError("validation MASE is not finite")
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError, ValueError) as error:
            metrics = ForecastMetrics(*(np.nan for _ in range(5)))
            status = f"failed: {type(error).__name__}: {error}"

        row = asdict(parameters)
        row.update(
            {
                "state_dimension": state_dimension,
                "validation_mae": metrics.mae,
                "validation_mse": metrics.mse,
                "validation_mape": metrics.mape,
                "validation_mase": metrics.mase,
                "validation_nmae": metrics.nmae,
                "seconds": perf_counter() - trial_started_at,
                "status": status,
            }
        )
        results.append(row)

        if status == "ok" and (
            best_metrics is None
            or (metrics.mase, metrics.nmae) < (best_metrics.mase, best_metrics.nmae)
        ):
            best_parameters, best_metrics = parameters, metrics

        if (
            args.search_progress_every > 0
            and (
                len(results) % args.search_progress_every == 0
                or len(results) == requested_trials
            )
        ):
            best_text = "none" if best_metrics is None else f"{best_metrics.mase:.8f}"
            print(
                f"Search progress: {len(results)}/{requested_trials} trials; "
                f"best validation MASE={best_text}"
            )

    search_time = perf_counter() - search_started_at
    if len(results) < requested_trials:
        warnings.warn(
            f"only {len(results)} unique configurations fit --max-state-dim; "
            f"{requested_trials} were requested",
            stacklevel=2,
        )
    if best_parameters is None or best_metrics is None:
        raise RuntimeError(
            "hyperparameter search produced no finite validation MASE; "
            "check the data or increase --max-state-dim"
        )
    return best_parameters, best_metrics, results, search_time


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune and run causal one-step SX_sGBDT forecasting with the "
            "chronological train/validation/test protocol from the manuscript. "
            "With no --csv argument, a reproducible synthetic example is used."
        )
    )
    data = parser.add_argument_group("data")
    data.add_argument("--csv", help="optional input CSV; rows must be time ordered")
    data.add_argument("--target", default="y", help="target column used with --csv")
    data.add_argument(
        "--features",
        nargs="*",
        default=[],
        help="feature columns known when each target is forecast",
    )
    data.add_argument("--n-samples", type=int, default=200, help="synthetic sample count")
    data.add_argument("--n-features", type=int, default=2, help="synthetic feature count")
    data.add_argument(
        "--test-fraction",
        type=float,
        default=0.30,
        help=(
            "chronological fraction reserved at the end for testing (manuscript "
            "real-world default: 0.30); "
            "ignored when --test-size is supplied"
        ),
    )
    data.add_argument(
        "--test-size",
        type=int,
        help="exact number of chronological rows reserved at the end for testing",
    )
    data.add_argument(
        "--validation-fraction",
        type=float,
        default=0.30,
        help=(
            "fraction of the pre-test data reserved at its end for validation "
            "(manuscript real-world default: 0.30)"
        ),
    )
    data.add_argument(
        "--validation-size",
        type=int,
        help="exact validation rows at the end of the pre-test data",
    )
    data.add_argument(
        "--validation-equals-test",
        action="store_true",
        help=(
            "reserve a validation window equal to the test window, matching the "
            "manuscript's M3/M4/ETT tuning protocol"
        ),
    )
    data.add_argument(
        "--mase-lag",
        "--mase-seasonality",
        dest="mase_lag",
        type=int,
        default=1,
        help="training-only naive scaling lag (default: 1 for one-step-ahead MASE)",
    )
    data.add_argument("--predictions-out", help="optional output CSV for targets and predictions")

    model = parser.add_argument_group("anchor model")
    model.description = (
        "This explicit configuration is evaluated as the first search trial; "
        "with --no-search it is the only configuration."
    )
    model.add_argument("--n-estimators", type=int, default=5)
    model.add_argument("--depth", type=int, default=2)
    model.add_argument(
        "--sx-order",
        type=int,
        nargs=3,
        metavar=("P", "D", "Q"),
        default=(2, 0, 1),
    )
    model.add_argument(
        "--seasonal-order",
        type=int,
        nargs=4,
        metavar=("P", "D", "Q", "S"),
        help="optional seasonal order",
    )
    model.add_argument("--learning-rate", type=float, default=0.1)
    model.add_argument(
        "--process-noise-std",
        type=float,
        default=1.0,
        help="anchor EKF process-noise standard deviation (paper default: 1.0)",
    )
    model.add_argument(
        "--measurement-noise-std",
        type=float,
        default=1.0,
        help="anchor EKF measurement-noise standard deviation (paper default: 1.0)",
    )
    model.add_argument("--seed", type=int, default=0)

    search = parser.add_argument_group("large random hyperparameter search")
    search.add_argument(
        "--search-iterations",
        type=int,
        default=10000,
        help=(
            "number of unique random configurations (default: 10000); the search "
            "space spans billions of combinations"
        ),
    )
    search.add_argument(
        "--no-search",
        action="store_true",
        help="evaluate only the explicitly supplied anchor model",
    )
    search.add_argument(
        "--seasonal-periods",
        type=int,
        nargs="*",
        default=[],
        metavar="S",
        help=(
            "known seasonal periods to include in random SARIMAX order search; "
            "omit when the series has no known period"
        ),
    )
    search.add_argument(
        "--max-state-dim",
        type=int,
        default=1000,
        help=(
            "skip configurations above this EKF state dimension to bound the "
            "quadratic covariance memory (default: 1000)"
        ),
    )
    search.add_argument(
        "--search-progress-every",
        type=int,
        default=10,
        help="print search progress every N evaluated configurations; 0 disables it",
    )
    search.add_argument(
        "--search-results-out",
        help="optional CSV containing every configuration and its validation metrics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Tune on validation, refit on train+validation, then evaluate test once."""
    parser = _argument_parser()
    args = parser.parse_args(argv)

    if args.csv:
        target, features = _load_csv_data(args.csv, args.target, args.features)
        source = args.csv
    else:
        if args.features:
            parser.error("--features can only be used together with --csv")
        target, features = _synthetic_example(args.n_samples, args.n_features, args.seed)
        source = "synthetic causal example"

    if args.search_iterations < 1:
        parser.error("--search-iterations must be positive")
    if args.max_state_dim < 1:
        parser.error("--max-state-dim must be positive")
    if args.search_progress_every < 0:
        parser.error("--search-progress-every cannot be negative")
    if any(period <= 0 for period in args.seasonal_periods):
        parser.error("all --seasonal-periods must be positive")
    if args.validation_size is not None and args.validation_equals_test:
        parser.error("--validation-size and --validation-equals-test are mutually exclusive")

    if args.test_size is None:
        if not 0.0 < args.test_fraction < 1.0:
            parser.error("--test-fraction must be between 0 and 1")
        test_size = int(np.ceil(len(target) * args.test_fraction))
    else:
        if not 1 <= args.test_size < len(target):
            parser.error("--test-size must be between 1 and the number of rows minus 1")
        test_size = args.test_size
    pretest_size = len(target) - test_size

    if args.validation_size is not None:
        validation_size = args.validation_size
    elif args.validation_equals_test:
        validation_size = test_size
    else:
        if not 0.0 < args.validation_fraction < 1.0:
            parser.error("--validation-fraction must be between 0 and 1")
        validation_size = int(np.ceil(pretest_size * args.validation_fraction))
    if not 1 <= validation_size < pretest_size:
        parser.error(
            "validation size must leave at least one pre-validation training row"
        )
    train_end = pretest_size - validation_size

    mase_lag = args.mase_lag
    if mase_lag < 1:
        parser.error("--mase-lag must be positive")
    if train_end <= mase_lag:
        parser.error("the hyperparameter-training split is too short for the MASE scaling lag")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be positive")
    if args.depth < 1:
        parser.error("--depth must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.process_noise_std < 0.0 or args.measurement_noise_std <= 0.0:
        parser.error("noise standard deviations must be non-negative, with measurement > 0")

    print(f"Data: {source}")
    print(f"Observations: {len(target)}")
    # The manuscript calls all observations before the held-out test the
    # training data; a tail of it is used temporarily for model selection.
    print(f"Training observations (train + validation): {pretest_size}")
    print(f"Hyperparameter-training observations: {train_end}")
    print(f"Validation observations: {validation_size}")
    print(f"Test observations: {test_size}")
    print("Selection rule: lowest chronological validation MASE")

    best_parameters, validation_metrics, search_results, search_time = _random_search(
        args,
        target,
        features,
        train_end,
        pretest_size,
    )
    if args.search_results_out:
        _write_search_results(args.search_results_out, search_results)
        print(f"Search results written to: {args.search_results_out}")

    print(f"Hyperparameter search time: {search_time:.6f} s")
    print(f"Best validation MASE: {validation_metrics.mase:.8f}")
    print(f"Best validation nMAE: {validation_metrics.nmae:.8f}")
    print("Selected hyperparameters:")
    for name, value in asdict(best_parameters).items():
        print(f"  {name}: {value}")

    # Manuscript protocol: discard the tuned instance and refit a fresh model
    # on the complete train+validation partition before touching the test set.
    model = _make_model(best_parameters, features, args.seed)
    forecasts = np.empty(len(target), dtype=float)

    fitting_started_at = perf_counter()
    for t in range(pretest_size):
        x_t = None if features is None else features[t]
        forecasts[t] = ekf(model=model, y_t=target[t], x_t=x_t, t=t)
    total_fit_time = perf_counter() - fitting_started_at
    mean_fit_time = total_fit_time / pretest_size

    prediction_times = []
    for t in range(pretest_size, len(target)):
        x_t = None if features is None else features[t]
        forecasts[t], prediction_time = ekf(
            model=model,
            y_t=target[t],
            x_t=x_t,
            t=t,
            return_prediction_time=True,
        )
        prediction_times.append(prediction_time)
    mean_prediction_time = float(np.mean(prediction_times))
    combined_time = total_fit_time + test_size * mean_prediction_time

    test_metrics = _forecast_metrics(
        target[pretest_size:],
        forecasts[pretest_size:],
        target[:pretest_size],
        mase_lag,
        emit_warnings=True,
    )

    print("Test protocol: rolling causal one-step prediction with post-observation updates")
    print(f"MAE: {test_metrics.mae:.8f}")
    print(f"MSE: {test_metrics.mse:.8f}")
    print(f"MAPE: {test_metrics.mape:.8f} ({100.0 * test_metrics.mape:.4f}%)")
    print(
        f"MASE (final-training naive lag={mase_lag}): "
        f"{test_metrics.mase:.8f}"
    )
    print(
        "nMAE (MAE / final-training target range): "
        f"{test_metrics.nmae:.8f}"
    )
    print(
        f"Final refit time ({pretest_size} online update(s)): "
        f"{total_fit_time:.6f} s"
    )
    print(f"Mean final-refit time per online update: {mean_fit_time:.6f} s")
    print(
        f"1 inference time (mean over {test_size} predictions): "
        f"{mean_prediction_time:.6f} s "
        f"({1000.0 * mean_prediction_time:.3f} ms)"
    )
    print(
        f"Training time + {test_size} * 1 inference time: "
        f"{combined_time:.6f} s"
    )

    if args.predictions_out:
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - environment dependent.
            raise RuntimeError("pandas is required for --predictions-out") from error
        pd.DataFrame(
            {
                "target": target,
                "prediction": forecasts,
                "innovation": target - forecasts,
                "split": np.select(
                    [
                        np.arange(len(target)) < train_end,
                        np.arange(len(target)) < pretest_size,
                    ],
                    ["train", "validation"],
                    default="test",
                ),
            }
        ).to_csv(args.predictions_out, index=False)
        print(f"Predictions written to: {args.predictions_out}")

    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error {e}")
