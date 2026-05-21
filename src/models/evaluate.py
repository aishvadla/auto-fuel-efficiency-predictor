from dataclasses import dataclass
from src.data.preprocessing import DataPreprocessingConfig
from src.models.train import ModelTrainerConfig
from src.utils.helper import load_object
from src.models.model_registry import NeuralNet
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from pathlib import Path
from src.utils.logger import logger

@dataclass
class ModelEvaluatorConfig:
    preprocessing_config = DataPreprocessingConfig()
    trainer_config = ModelTrainerConfig()
    metrics_path = Path("artifacts") / "metrics.json"

class ModelEvaluator:
    def __init__(self):
        self.config = ModelEvaluatorConfig()

    def initiate_model_evaluation(self, X_test, y_test):
        logger.info("Initiated model evaluation on test set")
        scaler_y = load_object(self.config.preprocessing_config.scaler_y_obj_file_path)
        state_dict = torch.load(self.config.trainer_config.model_obj_file_path)

        model = NeuralNet(
            X_test.shape[1],
            self.config.trainer_config.hidden_units,
            self.config.trainer_config.output_features,
        )

        model.load_state_dict(state_dict)

        X_test = torch.from_numpy(X_test)
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test).detach().numpy().flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Final Test Set Eval: MSE={mse:.4f}   MAE={mae:.4f}   r2_score={r2:.4f}")

        logger.info("Saving the metrics data")
        metrics = {'mse': round(mse, 4), 'mae':round(mae, 4), 'r2':round(float(r2), 4)}
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Saved metrics.json file to artifacts")

        logger.info("Finished model evaluation on test set")
        return metrics