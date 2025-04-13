# Time Series Forecasting with ML & DL Models

This repository provides a modular framework for time series forecasting using machine learning (ML) and deep learning (DL) models. It includes custom models, multi-resolution feature extraction, and a full pipeline for model comparison and evaluation.

---

## Repository Structure

- `SX_sGBDT.py`  
  Custom implementation of the **SX_sGBDT** model, optimized for time series forecasting.

- `feature_extractor.ipynb`  
  Jupyter notebook for extracting features from time series data at various temporal resolutions (e.g., hourly, daily).

- `model_comparision.ipynb`  
  Complete pipeline to:
  - Load time series data,
  - Apply feature engineering,
  - Train ML and DL models,
  - Evaluate and visualize forecasting performance.

- `requirements.txt`  
  List of required Python packages for the project environment.

---

## Models Included

This framework supports a variety of models:
- **Statistical Models**
  - Naive
  - SARIMAX

- **Machine Learning Models**
  - Gradient Boosted Decision Trees (GBDT)
  - Decision Trees, Linear Regressor etc.

- **Deep Learning Models**
  - ANN
  - Transformer-based architectures

- **Custom Model**
  - `SX_sGBDT` – A specialized model developed for this project.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create Environment and Install Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open and run the following notebooks in order:
- `feature_extractor.ipynb`
- `model_comparision.ipynb`

---

## Inputs & Outputs

### Input:
- Custom or provided time series dataset (must be time-indexed)

### Output:
- Engineered feature datasets
- Forecasts from various models
- Performance metrics (e.g., MAE, RMSE)
- Plots and visualizations

---

## Notes

- You can swap in your own dataset in the comparison notebook.
- Code is modular and easy to adapt to new models or feature schemes.
- Supports multi-resolution forecasting via customizable feature extraction.

---

