from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, MAE, MQLoss
from neuralforecast.models import DeepAR, Informer, NBEATSx, NHITS, PatchTST, iTransformer
from sklearn.model_selection import ParameterGrid, ParameterSampler


MODEL_CHOICES = ("nbeats", "nhits", "patchtst", "deepar", "itransformer", "informer")
MODEL_ALIASES = {
    "nbeats": "NBEATSx",
    "nhits": "NHITS",
    "patchtst": "PatchTST",
    "deepar": "DeepAR",
    "itransformer": "iTransformer",
    "informer": "Informer",
}
SERIES_ID = "series"


COMMON_SEARCH_SPACE: dict[str, Sequence[Any]] = {
    "input_size": [6, 12, 18, 24, 36, 48, 72, 96],
    "max_steps": [100, 250, 500, 750, 1000],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    # NeuralForecast's batch_size counts series per batch, so it has no effect
    # on a single series; windows_batch_size is the effective batch size.
    "windows_batch_size": [32, 64, 128, 256, 512, 1024],
    "scaler_type": ["identity", "standard", "robust"],
}

MODEL_SEARCH_SPACES: dict[str, dict[str, Sequence[Any]]] = {
    "nbeats": {
        **COMMON_SEARCH_SPACE,
        "n_blocks": [1, 2, 3, 4],
        "mlp_width": [64, 128, 256, 512, 1024],
        "mlp_depth": [1, 2, 3, 4],
        "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
        "activation": ["ReLU", "Softplus", "Tanh", "SELU"],
    },
    "nhits": {
        **COMMON_SEARCH_SPACE,
        "n_blocks": [1, 2, 3],
        "mlp_width": [64, 128, 256, 512, 1024],
        "mlp_depth": [1, 2, 3, 4],
        "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
        "pooling_mode": ["MaxPool1d", "AvgPool1d"],
        "interpolation_mode": ["linear", "nearest", "cubic"],
    },
    "patchtst": {
        **COMMON_SEARCH_SPACE,
        "encoder_layers": [1, 2, 3, 4, 6],
        "hidden_n_heads": [(32, 2), (64, 2), (64, 4), (128, 4), (128, 8), (256, 8)],
        "linear_hidden_size": [64, 128, 256, 512],
        "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
        "patch_len": [2, 4, 6, 8, 12, 16, 24],
        "stride": [1, 2, 4, 6, 8, 12],
        "revin": [True, False],
        "activation": ["gelu", "relu"],
    },
    "deepar": {
        **COMMON_SEARCH_SPACE,
        "lstm_n_layers": [1, 2, 3, 4, 5],
        "lstm_hidden_size": [32, 64, 128, 256, 512],
        "lstm_dropout": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
        "decoder_hidden_layers": [0, 1, 2, 3],
        "decoder_hidden_size": [16, 32, 64, 128, 256],
        "trajectory_samples": [50, 100, 200, 500],
    },
    "itransformer": {
        **COMMON_SEARCH_SPACE,
        "hidden_n_heads": [(32, 2), (64, 2), (64, 4), (128, 4), (128, 8), (256, 8)],
        # iTransformer has no decoder and uses full attention, so NeuralForecast
        # ignores d_layers and factor; they are not searched.
        "e_layers": [1, 2, 3, 4, 6],
        "d_ff": [64, 128, 256, 512, 1024],
        "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
        "use_norm": [True, False],
    },
    "informer": {
        **COMMON_SEARCH_SPACE,
        "hidden_n_heads": [(32, 2), (64, 2), (64, 4), (128, 4), (128, 8), (256, 8)],
        "conv_hidden_size": [16, 32, 64, 128, 256],
        "encoder_layers": [1, 2, 3, 4, 6],
        "decoder_layers": [1, 2, 3, 4],
        "decoder_input_size_multiplier": [0.25, 0.5, 0.75, 0.9],
        "dropout": [0.0, 0.05, 0.1, 0.2, 0.3],
        "factor": [1, 2, 3, 5],
        "activation": ["gelu", "relu"],
        "distil": [True, False],
    },
}


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    train_validation: pd.DataFrame
    test: pd.DataFrame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-search deep one-step-ahead forecasting models on any "
            "time-ordered CSV file."
        )
    )
    parser.add_argument("--csv", required=True, help="time-ordered input CSV")
    parser.add_argument("--target", default="y", help="numeric target column")
    parser.add_argument(
        "--date-column",
        default="date",
        help="timestamp column; a synthetic index is created when it is absent",
    )
    parser.add_argument(
        "--frequency",
        default="auto",
        help="pandas frequency such as h, D, W, MS, QS, or YS; default: infer",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help=(
            "future-known numeric feature columns; default: every column except "
            "target and date; pass --features with no names to disable exogenous features"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
        help="deep models to search and evaluate",
    )
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument(
        "--validation-equals-test",
        action="store_true",
        help="reserve a validation window equal to the test window",
    )
    parser.add_argument("--mase-lag", type=int, default=1)
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=10000,
        help="random configurations evaluated per model (default: 10000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="optional override for max_steps in every sampled configuration",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Lightning training accelerator",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--results-out",
        default="deep_benchmark_results.json",
        help="JSON containing best parameters, metrics, and timings",
    )
    parser.add_argument("--predictions-out", help="optional test prediction CSV")
    parser.add_argument(
        "--search-results-dir",
        help="optional directory for per-trial validation results",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.search_iterations < 1:
        raise ValueError("--search-iterations must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.mase_lag < 1:
        raise ValueError("--mase-lag must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.test_size is not None and args.test_size < 1:
        raise ValueError("--test-size must be positive")
    if args.validation_size is not None and args.validation_size < 1:
        raise ValueError("--validation-size must be positive")
    if args.validation_size is not None and args.validation_equals_test:
        raise ValueError(
            "--validation-size and --validation-equals-test are mutually exclusive"
        )
    if args.test_size is None and not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between zero and one")
    if (
        args.validation_size is None
        and not args.validation_equals_test
        and not 0.0 < args.validation_fraction < 1.0
    ):
        raise ValueError("--validation-fraction must be between zero and one")


def _resolve_time_index(
    df: pd.DataFrame,
    date_column: str,
    requested_frequency: str,
) -> tuple[pd.Series, str]:
    if date_column in df.columns:
        ds = pd.to_datetime(df[date_column], errors="raise")
        if ds.duplicated().any() or not ds.is_monotonic_increasing:
            raise ValueError("the date column must be unique and increasing")
        inferred = pd.infer_freq(ds)
        frequency = inferred if requested_frequency == "auto" else requested_frequency
        if frequency is None:
            raise ValueError(
                "the timestamp frequency could not be inferred; pass --frequency"
            )
        expected = pd.date_range(ds.iloc[0], periods=len(ds), freq=frequency)
        if not np.array_equal(ds.to_numpy(), expected.to_numpy()):
            raise ValueError(
                "timestamps are not regular at the requested frequency; fill missing rows first"
            )
        return ds.reset_index(drop=True), frequency

    frequency = "D" if requested_frequency == "auto" else requested_frequency
    ds = pd.Series(pd.date_range("2000-01-01", periods=len(df), freq=frequency))
    return ds, frequency


def load_csv(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], str]:
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    raw = pd.read_csv(csv_path)
    unnamed_columns = [column for column in raw.columns if column.startswith("Unnamed:")]
    if unnamed_columns:
        raw = raw.drop(columns=unnamed_columns)
    if args.target not in raw.columns:
        raise ValueError(f"target column {args.target!r} was not found")

    ds, frequency = _resolve_time_index(raw, args.date_column, args.frequency)
    excluded = {args.target, args.date_column, "ds", "unique_id"}
    if args.features is None:
        feature_columns = [column for column in raw.columns if column not in excluded]
    else:
        feature_columns = list(args.features)
    missing_features = sorted(set(feature_columns) - set(raw.columns))
    if missing_features:
        raise ValueError(f"feature columns were not found: {missing_features}")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature column names must be unique")

    prepared = pd.DataFrame(
        {
            "unique_id": SERIES_ID,
            "ds": ds,
            "y": pd.to_numeric(raw[args.target], errors="raise").astype(float),
        }
    )
    for column in feature_columns:
        prepared[column] = pd.to_numeric(raw[column], errors="raise").astype(float)
    numeric = prepared[["y", *feature_columns]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("target and feature columns must be finite and contain no missing values")
    return prepared, feature_columns, frequency


def chronological_split(data: pd.DataFrame, args: argparse.Namespace) -> ChronologicalSplit:
    test_size = (
        args.test_size
        if args.test_size is not None
        else int(np.ceil(len(data) * args.test_fraction))
    )
    train_validation_size = len(data) - test_size
    if args.validation_size is not None:
        validation_size = args.validation_size
    elif args.validation_equals_test:
        validation_size = test_size
    else:
        validation_size = int(
            np.ceil(train_validation_size * args.validation_fraction)
        )
    train_size = train_validation_size - validation_size
    if train_size <= args.mase_lag or validation_size < 1 or test_size < 1:
        raise ValueError("the CSV is too short for the requested train/validation/test split")

    return ChronologicalSplit(
        train=data.iloc[:train_size].reset_index(drop=True),
        validation=data.iloc[train_size:train_validation_size].reset_index(drop=True),
        train_validation=data.iloc[:train_validation_size].reset_index(drop=True),
        test=data.iloc[train_validation_size:].reset_index(drop=True),
    )


def mase_scale(training_target: Sequence[float], lag: int = 1) -> float:
    target = np.asarray(training_target, dtype=float)
    if lag < 1 or len(target) <= lag:
        raise ValueError("training data are too short for the requested MASE lag")
    scale = float(np.mean(np.abs(target[lag:] - target[:-lag])))
    return np.nan if scale <= np.finfo(float).eps else scale


def calculate_metrics(
    actual: Sequence[float],
    prediction: Sequence[float],
    scaling_target: Sequence[float],
    mase_lag: int = 1,
) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    scaling_array = np.asarray(scaling_target, dtype=float)
    if actual_array.shape != prediction_array.shape or actual_array.ndim != 1:
        raise ValueError("actual and prediction must be equal-length one-dimensional arrays")
    error = actual_array - prediction_array
    absolute_error = np.abs(error)
    mae = float(np.mean(absolute_error))
    nonzero = np.abs(actual_array) > np.finfo(float).eps
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual_array[nonzero])))
        if nonzero.any()
        else np.nan
    )
    scale = mase_scale(scaling_array, mase_lag)
    target_range = float(np.max(scaling_array) - np.min(scaling_array))
    cumulative_mse = np.cumsum(error**2) / np.arange(1, len(error) + 1)
    return {
        "mae": mae,
        "mse": float(np.mean(error**2)),
        "mape": mape,
        "mase": mae / scale if np.isfinite(scale) else np.nan,
        "nmae": (
            mae / target_range
            if target_range > np.finfo(float).eps
            else np.nan
        ),
        "cumulative_mse": cumulative_mse.tolist(),
    }


def _safe_input_size(requested: int, training_size: int) -> int:
    maximum = training_size - 2
    if maximum < 2:
        raise ValueError("at least four hyperparameter-training observations are required")
    return max(2, min(int(requested), maximum))


def normalize_parameters(
    parameters: dict[str, Any],
    training_size: int,
    max_steps_override: int | None,
) -> dict[str, Any]:
    normalized = dict(parameters)
    normalized["input_size"] = _safe_input_size(
        int(normalized["input_size"]), training_size
    )
    if max_steps_override is not None:
        normalized["max_steps"] = max_steps_override
    normalized["windows_batch_size"] = int(normalized["windows_batch_size"])
    return normalized


def build_model(
    model_name: str,
    parameters: dict[str, Any],
    feature_columns: list[str],
    seed: int,
    accelerator: str,
):
    alias = MODEL_ALIASES[model_name]
    common: dict[str, Any] = {
        "h": 1,
        "input_size": parameters["input_size"],
        "futr_exog_list": feature_columns or None,
        "max_steps": parameters["max_steps"],
        "learning_rate": parameters["learning_rate"],
        "windows_batch_size": parameters["windows_batch_size"],
        "scaler_type": parameters["scaler_type"],
        "random_seed": seed,
        "alias": alias,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "logger": False,
        "accelerator": accelerator,
        "devices": 1,
    }

    if model_name == "nbeats":
        # NeuralForecast reads mlp_units as one [in, out] pair per hidden layer,
        # shared by every block.
        width = int(parameters["mlp_width"])
        depth = int(parameters["mlp_depth"])
        return NBEATSx(
            **common,
            stack_types=["identity"],
            n_blocks=[int(parameters["n_blocks"])],
            mlp_units=[[width, width] for _ in range(depth)],
            dropout_prob_theta=parameters["dropout"],
            activation=parameters["activation"],
            loss=MAE(),
        )

    if model_name == "nhits":
        width = int(parameters["mlp_width"])
        depth = int(parameters["mlp_depth"])
        return NHITS(
            **common,
            n_blocks=[int(parameters["n_blocks"])] * 3,
            mlp_units=[[width, width] for _ in range(depth)],
            n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[4, 2, 1],
            dropout_prob_theta=parameters["dropout"],
            pooling_mode=parameters["pooling_mode"],
            interpolation_mode=parameters["interpolation_mode"],
            loss=MAE(),
        )

    if model_name == "patchtst":
        hidden_size, n_heads = parameters["hidden_n_heads"]
        patch_len = min(int(parameters["patch_len"]), parameters["input_size"])
        stride = min(int(parameters["stride"]), patch_len)
        return PatchTST(
            **common,
            encoder_layers=int(parameters["encoder_layers"]),
            n_heads=int(n_heads),
            hidden_size=int(hidden_size),
            linear_hidden_size=int(parameters["linear_hidden_size"]),
            dropout=parameters["dropout"],
            fc_dropout=parameters["dropout"],
            head_dropout=parameters["dropout"],
            attn_dropout=parameters["dropout"],
            patch_len=patch_len,
            stride=max(1, stride),
            revin=bool(parameters["revin"]),
            activation=parameters["activation"],
            loss=MAE(),
        )

    if model_name == "deepar":
        return DeepAR(
            **common,
            lstm_n_layers=int(parameters["lstm_n_layers"]),
            lstm_hidden_size=int(parameters["lstm_hidden_size"]),
            lstm_dropout=parameters["lstm_dropout"],
            decoder_hidden_layers=int(parameters["decoder_hidden_layers"]),
            decoder_hidden_size=int(parameters["decoder_hidden_size"]),
            trajectory_samples=int(parameters["trajectory_samples"]),
            loss=DistributionLoss(
                distribution="StudentT", level=[80, 90], return_params=False
            ),
            valid_loss=MQLoss(level=[80, 90]),
        )

    if model_name == "itransformer":
        hidden_size, n_heads = parameters["hidden_n_heads"]
        return iTransformer(
            **common,
            n_series=1,
            hidden_size=int(hidden_size),
            n_heads=int(n_heads),
            e_layers=int(parameters["e_layers"]),
            d_ff=int(parameters["d_ff"]),
            dropout=parameters["dropout"],
            use_norm=bool(parameters["use_norm"]),
            loss=MAE(),
            valid_loss=MAE(),
        )

    hidden_size, n_head = parameters["hidden_n_heads"]
    return Informer(
        **common,
        hidden_size=int(hidden_size),
        n_head=int(n_head),
        conv_hidden_size=int(parameters["conv_hidden_size"]),
        encoder_layers=int(parameters["encoder_layers"]),
        decoder_layers=int(parameters["decoder_layers"]),
        decoder_input_size_multiplier=parameters["decoder_input_size_multiplier"],
        dropout=parameters["dropout"],
        factor=int(parameters["factor"]),
        activation=parameters["activation"],
        distil=bool(parameters["distil"]),
        loss=MAE(),
    )


def supported_feature_columns(
    model_name: str,
    feature_columns: list[str],
) -> list[str]:
    # PatchTST is univariate-only in NeuralForecast 3.x and rejects future
    # exogenous variables; every other model here accepts them.
    if model_name == "patchtst":
        return []
    return feature_columns


def fit_model(
    model_name: str,
    parameters: dict[str, Any],
    training_data: pd.DataFrame,
    feature_columns: list[str],
    frequency: str,
    seed: int,
    accelerator: str,
) -> tuple[NeuralForecast, float]:
    model = build_model(model_name, parameters, feature_columns, seed, accelerator)
    neural_forecast = NeuralForecast(models=[model], freq=frequency)
    model_data = training_data[["unique_id", "ds", "y", *feature_columns]]
    started_at = time.perf_counter()
    neural_forecast.fit(df=model_data, val_size=0)
    return neural_forecast, time.perf_counter() - started_at


def _point_forecast_column(forecast: pd.DataFrame, model_name: str) -> str:
    alias = MODEL_ALIASES[model_name]
    preferred = [f"{alias}-median", alias, f"{alias}-mean"]
    for column in preferred:
        if column in forecast.columns:
            return column
    candidates = [
        column
        for column in forecast.columns
        if column not in {"unique_id", "ds", "cutoff"}
        and pd.api.types.is_numeric_dtype(forecast[column])
        and not column.endswith(("-lo-80", "-lo-90", "-hi-80", "-hi-90"))
    ]
    if not candidates:
        raise ValueError(
            f"no point forecast column was found; columns={forecast.columns.tolist()}"
        )
    return candidates[0]


def walk_forward_one_step_predict(
    neural_forecast: NeuralForecast,
    model_name: str,
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, float, float]:
    predictions: list[float] = []
    inference_times: list[float] = []
    for step in range(len(evaluation_data)):
        history = pd.concat(
            [training_data, evaluation_data.iloc[:step]], ignore_index=True
        )[["unique_id", "ds", "y", *feature_columns]]
        current_row = evaluation_data.iloc[step]
        future_data = None
        if feature_columns:
            future_data = neural_forecast.make_future_dataframe(df=history)
            for column in feature_columns:
                future_data[column] = float(current_row[column])

        started_at = time.perf_counter()
        forecast = neural_forecast.predict(df=history, futr_df=future_data)
        inference_times.append(time.perf_counter() - started_at)
        prediction_column = _point_forecast_column(forecast, model_name)
        predictions.append(float(forecast[prediction_column].iloc[0]))

    if not predictions:
        raise ValueError("evaluation data must contain at least one row")
    total_time = float(np.sum(inference_times))
    return np.asarray(predictions), float(np.mean(inference_times)), total_time


def release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except (AssertionError, RuntimeError):
            pass


def resolve_accelerator(requested: str) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        probe = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del probe
    except (AssertionError, RuntimeError) as exc:
        # NeuralForecast enables NVML-only detection, which can report a GPU even
        # when the installed driver cannot execute this PyTorch CUDA build.
        os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
        print(
            f"CUDA was detected but is not usable ({exc}); falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    return "gpu"


def random_candidates(
    model_name: str,
    search_iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    search_space = MODEL_SEARCH_SPACES[model_name]
    total_combinations = len(ParameterGrid(search_space))
    if total_combinations < search_iterations:
        raise ValueError(
            f"{model_name} has only {total_combinations} configurations, fewer than "
            f"the requested {search_iterations}"
        )
    return list(
        ParameterSampler(
            search_space,
            n_iter=search_iterations,
            random_state=seed,
        )
    )


def search_model(
    model_name: str,
    split: ChronologicalSplit,
    feature_columns: list[str],
    frequency: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    candidates = random_candidates(model_name, args.search_iterations, args.seed)
    best_parameters: dict[str, Any] | None = None
    best_validation_mase = np.inf
    trial_results: list[dict[str, Any]] = []
    search_started_at = time.perf_counter()

    for trial_number, sampled_parameters in enumerate(candidates, start=1):
        parameters = normalize_parameters(
            sampled_parameters,
            len(split.train),
            args.max_steps,
        )
        neural_forecast: NeuralForecast | None = None
        trial_started_at = time.perf_counter()
        try:
            neural_forecast, fit_time = fit_model(
                model_name,
                parameters,
                split.train,
                feature_columns,
                frequency,
                args.seed + trial_number,
                args.accelerator,
            )
            prediction, mean_inference_time, inference_time = (
                walk_forward_one_step_predict(
                    neural_forecast,
                    model_name,
                    split.train,
                    split.validation,
                    feature_columns,
                )
            )
            metrics = calculate_metrics(
                split.validation["y"],
                prediction,
                split.train["y"],
                args.mase_lag,
            )
            validation_mase = float(metrics["mase"])
            status = "ok" if np.isfinite(validation_mase) else "non_finite"
            error = ""
            if np.isfinite(validation_mase) and validation_mase < best_validation_mase:
                best_validation_mase = validation_mase
                best_parameters = dict(parameters)
        except Exception as exc:
            fit_time = np.nan
            mean_inference_time = np.nan
            inference_time = np.nan
            validation_mase = np.nan
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            neural_forecast = None
            release_accelerator_memory()

        trial_results.append(
            {
                "trial": trial_number,
                "status": status,
                "validation_mase": validation_mase,
                "fit_time_seconds": fit_time,
                "mean_inference_time_seconds": mean_inference_time,
                "validation_inference_time_seconds": inference_time,
                "trial_time_seconds": time.perf_counter() - trial_started_at,
                "parameters": json.dumps(_json_ready(parameters), sort_keys=True),
                "error": error,
            }
        )
        if trial_number % args.progress_every == 0 or trial_number == len(candidates):
            best_text = (
                f"{best_validation_mase:.8f}"
                if np.isfinite(best_validation_mase)
                else "none"
            )
            print(
                f"{MODEL_ALIASES[model_name]} search "
                f"{trial_number}/{len(candidates)}; best validation MASE={best_text}"
            )

    search_time = time.perf_counter() - search_started_at
    if best_parameters is None:
        failure_details: list[str] = []
        for row in trial_results:
            detail = row["error"] or f"status={row['status']}"
            if detail not in failure_details:
                failure_details.append(detail)
            if len(failure_details) == 3:
                break
        detail_text = " | ".join(failure_details)
        raise RuntimeError(
            f"all {len(candidates)} {MODEL_ALIASES[model_name]} trials failed or "
            f"produced non-finite validation MASE; details: {detail_text}"
        )
    return best_parameters, search_time, trial_results


def evaluate_model(
    model_name: str,
    split: ChronologicalSplit,
    feature_columns: list[str],
    frequency: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    model_feature_columns = supported_feature_columns(model_name, feature_columns)
    best_parameters, search_time, trial_results = search_model(
        model_name,
        split,
        model_feature_columns,
        frequency,
        args,
    )
    neural_forecast: NeuralForecast | None = None
    try:
        neural_forecast, training_time = fit_model(
            model_name,
            best_parameters,
            split.train_validation,
            model_feature_columns,
            frequency,
            args.seed,
            args.accelerator,
        )
        prediction, one_prediction_time, test_inference_time = (
            walk_forward_one_step_predict(
                neural_forecast,
                model_name,
                split.train_validation,
                split.test,
                model_feature_columns,
            )
        )
    finally:
        neural_forecast = None
        release_accelerator_memory()

    metrics = calculate_metrics(
        split.test["y"],
        prediction,
        split.train_validation["y"],
        args.mase_lag,
    )
    result = {
        "model": MODEL_ALIASES[model_name],
        "features_used": model_feature_columns,
        "best_parameters": best_parameters,
        "search_iterations": args.search_iterations,
        "validation_mase": min(
            float(row["validation_mase"])
            for row in trial_results
            if np.isfinite(row["validation_mase"])
        ),
        **metrics,
        "search_time_seconds": search_time,
        "training_time_seconds": training_time,
        "one_prediction_time_seconds": one_prediction_time,
        "test_inference_time_seconds": test_inference_time,
        "total_time_seconds": training_time + test_inference_time,
    }
    return result, prediction, trial_results


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def print_result(result: dict[str, Any]) -> None:
    print(f"Model: {result['model']}")
    print(f"Best parameters: {result['best_parameters']}")
    print(f"Validation MASE: {result['validation_mase']:.8f}")
    print(f"MASE: {result['mase']:.8f}")
    print(f"nMAE: {result['nmae']:.8f}")
    print(f"MSE: {result['mse']:.8f}")
    print(f"MAPE: {result['mape']:.8f} ({100.0 * result['mape']:.4f}%)")
    print(f"MAE: {result['mae']:.8f}")
    print(f"Search time: {result['search_time_seconds']:.6f} s")
    print(f"Training time: {result['training_time_seconds']:.6f} s")
    print(f"1 inference time: {result['one_prediction_time_seconds']:.6f} s")
    print(f"Full test inference time: {result['test_inference_time_seconds']:.6f} s")
    print(f"Training + test time: {result['total_time_seconds']:.6f} s")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.accelerator = resolve_accelerator(args.accelerator)

    data, feature_columns, frequency = load_csv(args)
    split = chronological_split(data, args)
    print(f"CSV: {args.csv}")
    print(f"Rows: {len(data)}")
    print(f"Features: {feature_columns}")
    print(f"Frequency: {frequency}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Train: {len(split.train)}")
    print(f"Validation: {len(split.validation)}")
    print(f"Train + validation: {len(split.train_validation)}")
    print(f"Test: {len(split.test)}")
    print(f"Random search iterations per model: {args.search_iterations}")

    payload: dict[str, Any] = {
        "csv": str(Path(args.csv).resolve()),
        "target": args.target,
        "date_column": args.date_column,
        "frequency": frequency,
        "features": feature_columns,
        "split": {
            "train": len(split.train),
            "validation": len(split.validation),
            "train_validation": len(split.train_validation),
            "test": len(split.test),
        },
        "models": {},
        "failures": {},
    }
    predictions = pd.DataFrame(
        {
            "ds": split.test["ds"].to_numpy(),
            "target": split.test["y"].to_numpy(),
        }
    )

    for model_name in args.models:
        print("=" * 80)
        print(f"Searching {MODEL_ALIASES[model_name]}")
        try:
            result, model_prediction, trial_results = evaluate_model(
                model_name,
                split,
                feature_columns,
                frequency,
                args,
            )
            payload["models"][model_name] = result
            predictions[MODEL_ALIASES[model_name]] = model_prediction
            print_result(result)
            if args.search_results_dir:
                search_path = Path(args.search_results_dir) / f"{model_name}_search.csv"
                search_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(trial_results).to_csv(search_path, index=False)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            payload["failures"][model_name] = message
            print(f"{MODEL_ALIASES[model_name]} failed: {message}", file=sys.stderr)
        write_json(args.results_out, payload)

    if args.predictions_out:
        prediction_path = Path(args.predictions_out)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(prediction_path, index=False)
    if payload["failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
