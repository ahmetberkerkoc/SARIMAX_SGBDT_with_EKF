# SX-sGBDT

Research code for causal, one-step-ahead time-series forecasting. The main
model combines SARIMAX with soft gradient-boosted decision trees (SX-sGBDT) and
uses an extended Kalman filter (EKF) to update its parameters online.

The repository also contains feature engineering and machine-learning and
deep-learning comparison models.

## Files

| File                        | Description                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `SX_sGBDT.py`             | Main SX-sGBDT model, EKF equations, hyperparameter search, evaluation metrics, synthetic example, and CSV experiment interface.          |
| `deepmodels.py`           | Deep-learning comparisons using NBEATSx, NHITS, PatchTST, DeepAR, iTransformer, and Informer.                                            |
| `feature_extractor.ipynb` | Creates lag and rolling-window features for different time resolutions. Input paths and the time resolution are set inside the notebook. |
| `othermodels.ipynb`       | Compares LightGBM, XGBoost, decision trees, linear regression, a naive forecast, and an artificial neural network.                       |
| `test_sx_sgbdt.py`        | Tests model causality, EKF calculations, differencing, metrics, search parameters, and the experiment interface.                         |
| `pyproject.toml`          | Defines the project, Python requirement, dependencies, development tools, and test configuration.                                        |
| `uv.lock`                 | Records the exact dependency versions used by the project.                                                                               |
| `uv copy.lock`            | A duplicate of `uv.lock`; it is not used by `uv`.                                                                                    |
| `.python-version`         | Selects Python 3.10 for the local environment.                                                                                           |
| `.gitignore`              | Excludes the virtual environment, caches, build files, and generated outputs from Git.                                                   |

## Environments

The project uses [uv](https://docs.astral.sh/uv/) for its Python environment and
dependencies. Create the `.venv` environment and install the locked packages:

```bash
uv sync --frozen
```
