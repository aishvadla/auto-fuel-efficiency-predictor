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
    return {"message": "Auto MPG Prediction API", "docs": "/docs"}


@app.post("/predict")
def predict(features: CarFeatures):
    mpg = pipeline.predict(features.model_dump())
    return {"predicted_mpg": round(mpg, 2)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
