# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.model import load_historical_data, predict_with_adjustments
import yaml

# Load configuration
with open("app/config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="AssetIQ Valuator API")

# Pydantic model for API request
class PredictionRequest(BaseModel):
    brand: str
    future_years: list

@app.get("/")
def read_root():
    return {"message": "Welcome to the AssetIQ Valuator API"}

@app.post("/predict")
def predict_asset(request: PredictionRequest):
    # Load historical data (for demonstration, using smartphones data)
    df = load_historical_data()
    # Create future exogenous indicators using historical averages
    future_years = request.future_years
    exog_future = pd.DataFrame({
        'sentiment': [df['sentiment'].mean()] * len(future_years),
        'macro': [df['macro'].mean()] * len(future_years)
    }, index=future_years)
    prediction = predict_with_adjustments(df, request.brand, future_years, exog_future)
    # Return prediction as a dictionary
    return {"brand": request.brand, "future_years": future_years, "predictions": prediction.tolist()}
