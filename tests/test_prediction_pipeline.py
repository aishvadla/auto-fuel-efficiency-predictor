# tests/test_prediction_pipeline.py
from src.pipelines.prediction_pipeline import PredictionPipeline


def test_prediction_pipeline():
    pipeline = PredictionPipeline()

    input_data = {
        "Cylinders": 4,
        "Displacement": 120.0,
        "Horsepower": 79.0,
        "Weight": 2625.0,
        "Acceleration": 18.6,
        "Model Year": 82,
        "Origin": 1,
    }

    mpg = pipeline.predict(input_data)
    print(f"Predicted MPG: {mpg:.2f}")

    assert isinstance(mpg, float), "Prediction should be a float"
    assert 5 < mpg < 60, "Prediction should be in realistic MPG range"
