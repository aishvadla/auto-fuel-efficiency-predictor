"""Tests for model evaluation and metrics output."""

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessing
from src.models.evaluate import ModelEvaluator


def test_model_evaluator(pipeline_data):
    evaluator = ModelEvaluator()

    X_train, X_val, X_test, y_train, y_val, y_test = pipeline_data
    metrics = evaluator.initiate_model_evaluation(X_test, y_test)
    print(metrics)

    assert metrics["r2"] > 0
    assert metrics["mae"] > 0
