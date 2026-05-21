# 1. take inputs from the datapreprocessing module
# 2. Convert to torch tensors, dataset and dataloaders
# 3. Initialize model
# 4. Training loop with validation loss per epoch
# 5. save model to artifacts

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from src.models.model_registry import NeuralNet
from src.utils.helper import load_config
from typing import List
from torchmetrics.regression import R2Score


@dataclass
class ModelTrainerConfig:
    model_obj_file_path = Path("artifacts") / "model.pth"
    epochs: int = None
    log_epochs: int = None
    eta: float = None
    batch_size: int = None
    hidden_units: List = None
    output_features: int = None

    def __post_init__(self):
        config = load_config()
        # use yaml value if not explicitly passed
        if self.epochs is None:
            self.epochs = config["training"]["epochs"]
        if self.log_epochs is None:
            self.log_epochs = config["training"]["log_epochs"]
        if self.eta is None:
            self.eta = config["training"]["eta"]
        if self.batch_size is None:
            self.batch_size = config["training"]["batch_size"]
        if self.hidden_units is None:
            self.hidden_units = config["model"]["hidden_units"]
        if self.output_features is None:
            self.output_features = config["model"]["output_features"]


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = None):
        if config is None:
            self.trainer_config = ModelTrainerConfig()
        else:
            self.trainer_config = config

    def initiate_model_training(self, X_train, X_val, y_train, y_val):
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        X_val = torch.from_numpy(X_val)
        y_val = torch.from_numpy(y_val)
        train_dl = DataLoader(
            train_ds, batch_size=self.trainer_config.batch_size, shuffle=True
        )
        
        model = NeuralNet(
            X_train.shape[1],
            self.trainer_config.hidden_units,
            self.trainer_config.output_features,
        )

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.trainer_config.eta)

        loss_hist_train = [0] * self.trainer_config.epochs
        r2_score_hist_train = [0] * self.trainer_config.epochs
        loss_hist_val = [0] * self.trainer_config.epochs
        r2_score_hist_val = [0] * self.trainer_config.epochs

        for epoch in range(self.trainer_config.epochs):
            # --- TRAINING PHASE ---
            model.train()  # 1. Set the model to training mode (enables Dropout/BatchNorm tracking)
            m_r2 = R2Score()
            for X_batch, y_batch in train_dl:
                y_pred = model(X_batch)[:, 0]
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_hist_train[epoch] += loss.item()
                m_r2.update(y_pred, y_batch)
            loss_hist_train[epoch] /= len(train_dl)
            r2_score_hist_train[epoch] = m_r2.compute()

            # --- EVALUATION / VALIDATION PHASE ---
            model.eval()  # 2. Switch to evaluation mode (disables Dropout, freezes BatchNorm)

            # validation dataset
            with torch.no_grad():
                y_pred = model(X_val)[:, 0]
                loss_hist_val[epoch] = loss_fn(y_pred, y_val).item()
                m_r2 = R2Score()
                r2_score_hist_val[epoch] = m_r2(y_pred, y_val)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: train_loss: {loss_hist_train[epoch]:.4f} train_r2: {r2_score_hist_train[epoch]:.4f} val_loss: {loss_hist_val[epoch]:.4f} val_r2: {r2_score_hist_val[epoch]:.4f}"
                )
        print(f"Final epoch {epoch}: train_loss: {loss_hist_train[epoch]:.4f} train_r2: {r2_score_hist_train[epoch]:.4f} val_loss: {loss_hist_val[epoch]:.4f} val_r2: {r2_score_hist_val[epoch]:.4f}")
        torch.save(model.state_dict(), self.trainer_config.model_obj_file_path)
        return (loss_hist_train, r2_score_hist_train, loss_hist_val, r2_score_hist_val)
