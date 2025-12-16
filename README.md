# 🚲 BoomBikes Bike Sharing — Multiple Linear Regression (MLR)

This project builds and evaluates a **Multiple Linear Regression** model to explain and predict **daily bike demand** for BoomBikes using the `day.csv` dataset. The analysis covers data preparation, feature engineering/selection, diagnostics (multicollinearity, residual analysis), and model evaluation with both **scikit-learn** and **statsmodels**. citeturn13search1

---

## 🎯 Objectives
- Understand drivers of daily rentals (weather, seasonality, holiday/working day, trend). citeturn13search1
- Build a parsimonious, interpretable MLR model to estimate demand and support pricing/operations. citeturn13search1
- Validate assumptions: linearity, multicollinearity (VIF), homoscedasticity, normality of residuals. citeturn13search1

---

## 📂 Project Structure
```
BoomBikes_Bike_Sharing_MLR.py (notebook-style script)
├─ Imports & setup          # numpy, pandas, matplotlib, seaborn; sklearn; statsmodels
├─ Data loading             # day.csv
├─ EDA & preprocessing      # scaling, encoding, train/test split
├─ Feature selection        # RFE with LinearRegression; domain curation
├─ Modeling (sklearn)       # LinearRegression fit; r2_score
├─ Modeling (statsmodels)   # OLS summary; p-values; VIF
├─ Diagnostics              # residual plots, Q-Q, influence
└─ Insights & next steps
```
citeturn13search1

---

## ⚙️ Environment & Dependencies
- **Python** ≥ 3.9
- **Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels` citeturn13search1

Install:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

---

## 🚀 Getting Started
1. Place `day.csv` in the project folder.
2. Run the notebook/script:
```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, warnings as wn
wn.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
day = pd.read_csv('day.csv')
```
citeturn13search1

---

## 🔧 Data Preparation (Typical)
- **Target**: `cnt` (total rentals). (If present in `day.csv` schema.)
- **Potential predictors**: `temp`, `atemp`, `hum`, `windspeed`, `season`, `yr`, `mnth`, `holiday`, `workingday`, `weathersit` (check actual columns). 
- **Train/Test split**: e.g., `train_test_split(..., test_size=0.3, random_state=42)`.
- **Scaling**: `MinMaxScaler` for continuous variables. 
- **Encoding**: Convert categorical indicators (season, weather) to dummies if needed.

---

## 🔎 Feature Selection
- **RFE** with `LinearRegression` to shortlist predictors. citeturn13search1
- **Domain curation**: Drop redundant/collinear variables (guided by **VIF** and **p-values**).

---

## 📐 Modeling & Evaluation
- **scikit-learn**: Fit `LinearRegression` on train set; evaluate **R²** on train/test; inspect coefficients.
- **statsmodels (OLS)**: Build an interpretable model; review **summary**, **p-values**, **adj R²**; compute **VIF** for multicollinearity. citeturn13search1
- **Diagnostics**: Residual vs. fitted plots, Q‑Q plots, influence (Cook’s distance).

---

## 📊 Typical Insights (Illustrative)
> Exact results depend on your `day.csv`. The following are common patterns in bike sharing data:
- **Seasonality/Trend**: Higher demand in pleasant seasons; positive trend across years.
- **Weather**: Adverse weather (heavy rain/snow) reduces rentals; temperature up to comfort range increases demand.
- **Working day & holiday**: Different usage patterns—commute vs. leisure.

---

## ✅ Recommendations
- **Capacity planning**: Increase availability during high‑demand seasons/weather windows.
- **Pricing & promotions**: Dynamic pricing and targeted offers for off‑peak/poor‑weather days.
- **Forecasting**: Integrate the MLR as a baseline; consider regularized models (Ridge/Lasso) for robustness.

---

## 📦 Outputs
- Cleaned feature set
- Selected predictor list (RFE/VIF)
- Model coefficients and metrics (R², Adj R²)
- Residual diagnostics plots
- Business insights & actions

---

## 🧪 Reproducibility & Extension
- Fully scripted; swap in updated `day.csv` to retrain.
- Extend with cross‑validation, regularization, interaction terms, and non‑linear effects.

---

## 👤 Maintainer
**Gouri Karthik Gembali** — Quality Engineer, Bengaluru, Karnataka

