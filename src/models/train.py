# 1. take inputs from the datapreprocessing module
# 2. Convert to torch tensors, dataset and dataloaders
# 3. Initialize model
# 4. Training loop with validation loss per epoch
# 5. save model to artifacts

import torch
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from src.models.model_registry import NeuralNet


@dataclass
class ModelTrainerConfig:
    model_obj_file_path = Path("artifacts") / "model.pth"


class ModelTrainer:
    def __init__(self, num_epochs=10, eta=0.01, batch_size=2):
        self.num_epochs = num_epochs
        self.eta = eta
        self.batch_size = batch_size

    def train(self, X_train, y_train, X_val, y_val):
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        model = NeuralNet()
