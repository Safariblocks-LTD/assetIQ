# app/model.py
# https://colab.research.google.com/drive/1P6j-ndmf08nCdf0uu4rdCzHZ3wYf08Be#scrollTo=RJMKdCvrymAd&line=199&uniqifier=1

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from arch import arch_model
from app.utils import safe_prediction

# Historical Data Setup (for demonstration, production would use live data)
def load_historical_data():
    # For simplicity, data is hard-coded here.
    data_smartphones = {
        'YEAR': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'Apple': [874, 899, 949, 949, 1199, 1199, 974],
        'Samsung': [1110, 1190, 1200, 1250, 1300, 1350, 1400],
        'Huawei': [450, 500, 550, 600, 650, 700, 750],
        'Google': [899, 399, 374, 699, 699, 849, 874],
        'Xiaomi': [150, 180, 200, 220, 250, 300, 350]
    }
    df = pd.DataFrame(data_smartphones)
    df.set_index('YEAR', inplace=True)
    # Simulated exogenous indicators
    np.random.seed(42)
    df['sentiment'] = np.random.uniform(0.95, 1.05, size=len(df))
    df['macro'] = np.random.uniform(0.98, 1.02, size=len(df))
    return df

# Base Ensemble Forecasting Functions
def train_arima(series):
    return auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)

def train_prophet(df, brand):
    df_prophet = df.reset_index()[['YEAR', brand]]
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
    model = Prophet(weekly_seasonality=True, daily_seasonality=True, n_changepoints=3)
    model.fit(df_prophet)
    return model

def train_rf_with_exog(df, brand, exog_cols):
    X = np.array(df.index).reshape(-1, 1)
    X_exog = np.hstack([X, df[exog_cols].values])
    y = df[brand].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_exog, y)
    return model

def predict_base(df, brand, future_years, exog_future):
    arima_model = train_arima(df[brand])
    prophet_model = train_prophet(df, brand)
    rf_model = train_rf_with_exog(df, brand, ['sentiment', 'macro'])
    
    n_future = len(future_years)
    arima_pred = arima_model.predict(n_periods=n_future)
    future_prophet = pd.DataFrame({'ds': pd.to_datetime(future_years, format='%Y')})
    prophet_pred = prophet_model.predict(future_prophet)['yhat'].values
    X_future = np.array(future_years).reshape(-1, 1)
    X_future_exog = np.hstack([X_future, exog_future.values])
    rf_pred = rf_model.predict(X_future_exog)
    
    last_val = df[brand].iloc[-1]
    arima_pred = safe_prediction(arima_pred, last_val)
    prophet_pred = safe_prediction(prophet_pred, last_val)
    rf_pred = safe_prediction(rf_pred, last_val)
    
    error = df[brand].iloc[-1] - df[brand].iloc[-2]
    corrected_arima = arima_pred + error
    corrected_prophet = prophet_pred + error
    corrected_rf = rf_pred + error
    
    return (corrected_arima + corrected_prophet + corrected_rf) / 3.0

# Volatility Adjustment using GARCH
def volatility_adjustment(df, brand):
    log_returns = np.log(df[brand]).diff().dropna()
    if len(log_returns) < 30:
        return 1.0
    garch_fit = arch_model(log_returns, vol='Garch', p=1, q=1, dist='t', rescale=False).fit(disp='off')
    sigma_forecast = np.sqrt(garch_fit.forecast(horizon=1).variance.values[-1, 0])
    historical_vol = log_returns.std()
    adjustment_factor = 1 - 0.1 * max(0, (sigma_forecast - historical_vol) / historical_vol)
    return adjustment_factor

# Kou's Jump Diffusion Adjustment
def kou_jump_adjustment(prediction, Delta_t, lambd, p_up, mu_jump_up, mu_jump_down):
    expected_jump = np.exp(lambd * Delta_t * (p_up * mu_jump_up + (1 - p_up) * mu_jump_down))
    return prediction * expected_jump

# Final Integrated Prediction Function
def predict_with_adjustments(df, brand, future_years, exog_future):
    base_pred = predict_base(df, brand, future_years, exog_future)
    # Jump Diffusion Adjustment parameters
    BEST_LAMBDA = 0.25
    BEST_P_UP = 0.50
    BEST_MU_JUMP_UP = 0.08
    BEST_MU_JUMP_DOWN = -0.05
    DELTA_T = 1.0
    jump_adjusted = kou_jump_adjustment(base_pred, DELTA_T, BEST_LAMBDA, BEST_P_UP, BEST_MU_JUMP_UP, BEST_MU_JUMP_DOWN)
    vol_factor = volatility_adjustment(df, brand)
    return jump_adjusted * vol_factor

# Function to run the model (for testing or production)
def run_model(future_years):
    results = {}
    # For demonstration, we use smartphones data. In production, use live data.
    df = load_historical_data()
    # Future exogenous indicators based on historical averages
    exog_future = pd.DataFrame({
        'sentiment': [df['sentiment'].mean()] * len(future_years),
        'macro': [df['macro'].mean()] * len(future_years)
    }, index=future_years)
    brands = df.columns.drop(['sentiment', 'macro'])
    for brand in brands:
        results[brand] = predict_with_adjustments(df, brand, future_years, exog_future)
    return results

if __name__ == "__main__":
    future_years = [2025, 2026, 2027]
    predictions = run_model(future_years)
    print("Predictions for Smartphones (2025-2027):")
    for brand, pred in predictions.items():
        print(f"{brand}: {pred}")
