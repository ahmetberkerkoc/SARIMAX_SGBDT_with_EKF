import warnings

import numpy as np

from SX_sGBDT import (
    EKF_SEARCH_SPACE,
    EKF_SGBDT,
    SARIMAX_SEARCH_SPACE,
    SGBDT_SEARCH_SPACE,
    SX_sGBDT,
    _argument_parser,
    _draw_parameters,
    _forecast_metrics,
    ekf,
    main,
)


def sarima_without_warning(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SX_sGBDT(exog=None, **kwargs)


def test_current_target_is_not_used_as_its_own_lag():
    model = sarima_without_warning(
        sx_order=(1, 0, 0), process_noise_std=0.0, random_state=0
    )
    model.state_vector[:] = 1.0

    first = ekf(model, y_t=7.0, t=0)
    second = ekf(model, y_t=999.0, t=1)

    assert first == 0.0
    assert np.isclose(second, 7.0)


def test_identity_transition_preserves_the_estimated_state():
    model = sarima_without_warning(sx_order=(1, 0, 0), random_state=0)
    model.state_vector[:] = 5.0

    predicted_state = model._state_transition(1)

    np.testing.assert_array_equal(predicted_state, np.array([5.0]))
    np.testing.assert_array_equal(model.state_vector, np.array([5.0]))


def test_measurement_jacobian_matches_finite_differences():
    x = np.array([[0.3, -0.2]])
    model = SX_sGBDT(
        n_estimator=2,
        depth=2,
        sx_order=(1, 0, 1),
        exog=x,
        learning_rate=[0.7, 1.2],
        random_state=4,
    )
    model._append_observation(0.8, -0.1)
    state = model.state_vector.copy()
    _, analytic = model._measurement_model(x[0], 0, state, return_jacobian=True)

    epsilon = 1e-6
    numerical = np.empty(model.state_dim)
    for index in range(model.state_dim):
        upper, lower = state.copy(), state.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        numerical[index] = (
            model._measurement_model(x[0], 0, upper)
            - model._measurement_model(x[0], 0, lower)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic[0], numerical, rtol=2e-5, atol=2e-6)


def test_every_estimator_contributes_with_its_learning_rate():
    x = np.array([[0.5]])
    model = SX_sGBDT(
        n_estimator=2,
        depth=1,
        sx_order=(0, 0, 0),
        exog=x,
        learning_rate=[1.0, 2.0],
        random_state=0,
    )
    model.state_vector[:] = 0.0
    model.state_vector[model._tree_slices[1].leaves] = 1.0

    assert np.isclose(model._measurement_model(x[0], 0), 2.0)


def test_depth_three_has_the_correct_internal_and_leaf_counts():
    model = SX_sGBDT(depth=3, exog=np.ones((1, 2)), random_state=0)

    assert model.n_internal == 7
    assert model.n_leaves == 8
    assert model.n_node == 15


def test_first_differencing_is_causal_and_inverted_for_forecasting():
    model = sarima_without_warning(
        sx_order=(0, 1, 0), process_noise_std=0.0, random_state=0
    )

    forecasts = EKF_SGBDT([5.0, 8.0], model)

    np.testing.assert_allclose(forecasts, [0.0, 5.0])


def test_no_exogenous_variables_remains_supported():
    model = sarima_without_warning(sx_order=(1, 0, 0), random_state=0)

    forecasts = EKF_SGBDT([1.0, 2.0, 3.0], model)

    assert forecasts.shape == (3,)
    assert np.isfinite(forecasts).all()
    assert len(model._measurements) == 3
    assert len(model._errors) == 3


def test_ekf_can_report_pre_observation_prediction_time():
    model = sarima_without_warning(sx_order=(1, 0, 0), random_state=0)

    forecast, prediction_time = ekf(
        model,
        y_t=1.0,
        t=0,
        return_prediction_time=True,
    )

    assert forecast == 0.0
    assert prediction_time >= 0.0


def test_main_runs_the_synthetic_example(capsys):
    result = main(
        ["--n-samples", "20", "--n-estimators", "2", "--no-search", "--seed", "3"]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Observations: 20" in output
    assert "Training observations (train + validation): 14" in output
    assert "Hyperparameter-training observations: 9" in output
    assert "Validation observations: 5" in output
    assert "Test observations: 6" in output
    assert "Best validation MASE:" in output
    assert "MAE:" in output
    assert "MSE:" in output
    assert "MAPE:" in output
    assert "MASE (final-training naive lag=1):" in output
    assert "nMAE (MAE / final-training target range):" in output


def test_main_supports_exact_test_size_and_combined_timing(capsys):
    result = main(
        [
            "--n-samples",
            "20",
            "--test-size",
            "6",
            "--no-search",
            "--seed",
            "3",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Training observations (train + validation): 14" in output
    assert "Test observations: 6" in output
    assert "Training time + 6 * 1 inference time:" in output


def test_nmae_uses_training_range_from_the_manuscript_definition():
    metrics = _forecast_metrics(
        actual=np.array([2.0, 4.0]),
        forecasts=np.array([1.0, 3.0]),
        scaling_target=np.array([0.0, 2.0, 4.0]),
        mase_lag=1,
    )

    assert metrics.mae == 1.0
    assert metrics.nmae == 0.25


def test_identity_ekf_noise_is_default_and_noise_values_are_searched():
    args = _argument_parser().parse_args(["--seasonal-periods", "12"])
    parameters = _draw_parameters(np.random.default_rng(9), args, has_features=True)

    assert args.process_noise_std == 1.0
    assert args.measurement_noise_std == 1.0
    assert parameters.n_estimators in SGBDT_SEARCH_SPACE["n_estimators"]
    assert parameters.learning_rate in SGBDT_SEARCH_SPACE["learning_rate"]
    assert parameters.depth in SGBDT_SEARCH_SPACE["depth"]
    assert parameters.sx_order[0] in SARIMAX_SEARCH_SPACE["p"]
    assert parameters.sx_order[1] in SARIMAX_SEARCH_SPACE["d"]
    assert parameters.sx_order[2] in SARIMAX_SEARCH_SPACE["q"]
    assert parameters.process_noise_std in EKF_SEARCH_SPACE["process_noise_std"]
    assert parameters.measurement_noise_std in EKF_SEARCH_SPACE["measurement_noise_std"]


def test_sarimax_constant_is_always_zero_and_not_a_state_parameter():
    args = _argument_parser().parse_args([])
    model = sarima_without_warning(sx_order=(2, 0, 1), random_state=0)

    assert not hasattr(args, "include_intercept")
    assert model.sx_size == 3
    assert model._measurement_model(None, 0) == 0.0
