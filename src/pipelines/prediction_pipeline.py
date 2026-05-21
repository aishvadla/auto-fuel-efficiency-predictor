"""Prediction pipeline for serving MPG estimates.

This module loads preprocessing and model artifacts to produce a single
point prediction from input feature values.
"""

from src.models.model_registry import NeuralNet
from src.data.preprocessing import DataPreprocessingConfig
from src.models.train import ModelTrainerConfig
from src.utils.helper import load_object
import torch
import pandas as pd


class PredictionPipeline:
    """Encapsulate inference logic for model predictions."""

    def __init__(self):
        self.preprocessing_config = DataPreprocessingConfig()
        self.trainer_config = ModelTrainerConfig()

    def predict(self, features: dict) -> float:
        """Predict MPG value for a single car feature dictionary.

        Parameters
        ----------
        features : dict
            Feature values keyed by input field names.

        Returns
        -------
        float
            Predicted MPG value.
        """
        preprocessor = load_object(self.preprocessing_config.preprocessor_obj_file_path)
        scaler_y = load_object(self.preprocessing_config.scaler_y_obj_file_path)
        state_dict = torch.load(self.trainer_config.model_obj_file_path)

        features["Model Year"] = features.pop("ModelYear")
        features_df = pd.DataFrame([features])
        features_transformed = preprocessor.transform(features_df)
        model = NeuralNet(
            features_transformed.shape[-1],
            self.trainer_config.hidden_units,
            self.trainer_config.output_features,
        )

        model.load_state_dict(state_dict)

        model.eval()
        with torch.no_grad():
            pred_scaled = (
                model(torch.from_numpy(features_transformed.astype("float32")))
                .numpy()
                .flatten()
            )
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        return float(pred)
