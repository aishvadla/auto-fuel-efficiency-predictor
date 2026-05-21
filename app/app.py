"""FastAPI application for MPG prediction.

This module defines the API contract, validation schema, and endpoints for
serving predictions from the trained auto fuel efficiency model.
"""

from pydantic import BaseModel
from fastapi import FastAPI
from src.pipelines.prediction_pipeline import PredictionPipeline


class CarFeatures(BaseModel):
    Cylinders: int
    Displacement: float
    Horsepower: float
    Weight: float
    Acceleration: float
    ModelYear: int
    Origin: int


app = FastAPI()
pipeline = PredictionPipeline()


@app.get("/")
def root():
    """Health check endpoint for the API.

    Returns a short message and pointer to API documentation.
    """
    return {"message": "Auto MPG Prediction API", "docs": "/docs"}


@app.post("/predict")
def predict(features: CarFeatures):
    """Return a predicted MPG value for provided car features.

    Args:
        features (CarFeatures): Validated car feature values.

    Returns:
        dict: Prediction result with rounded MPG value.
    """
    mpg = pipeline.predict(features.model_dump())
    return {"predicted_mpg": round(mpg, 2)}


@app.get("/health")
def health():
    """Return application health status."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
