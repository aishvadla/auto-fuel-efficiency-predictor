from src.models.model_registry import NeuralNet
import torch


def test_model_registry():
    input_features = 2
    hidden_units = [6, 3]
    output_features = 1

    model = NeuralNet(input_features, hidden_units, output_features)
    print(model)
    X = torch.randn(32, 2)

    output = model(X)
    print(f"Output shape: {output.shape}")

    assert X.shape[0] == output.shape[0], f"Input and output row count don't match"
