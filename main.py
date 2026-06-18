import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Linear Regression API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None

try:
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")


class PredictionInput(BaseModel):
    features: List[List[float]]


class PredictionOutput(BaseModel):
    predictions: List[float]


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Linear Regression API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        X = np.array(data.features)

        if len(X.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail="Input must be a 2D array"
            )

        if X.shape[1] != 4:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 4 features but got {X.shape[1]}"
            )

        predictions = model.predict(X)

        return PredictionOutput(
            predictions=predictions.tolist()
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )