"""Neural network registry for the auto fuel efficiency model.

Defines the model architecture used for both training and inference.
"""

import torch
import torch.nn as nn
from typing import List


class NeuralNet(nn.Module):
    """Feed-forward neural network with configurable hidden layers."""

    def __init__(self, input_features: int, hidden_units: List, output_features: int):
        super().__init__()
        all_layers = []
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_features, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_features = hidden_unit
        output_layer = nn.Linear(hidden_units[-1], output_features)
        all_layers.append(output_layer)
        self.model = nn.Sequential(*all_layers)

    def forward(self, X):
        """Forward pass for the neural network.

        Parameters
        ----------
        X : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """
        output = self.model(X)
        return output
