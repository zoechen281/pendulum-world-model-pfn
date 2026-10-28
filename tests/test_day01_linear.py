"""验证线性层创建、批量前向计算和参数求导的连接。"""

import torch

from src.day01_regression import build_linear_model, make_inputs, make_targets, mse_loss


def test_linear_model_forward_and_parameter_gradients():
    torch.manual_seed(0)
    model = build_linear_model()
    assert isinstance(model, torch.nn.Linear)
    assert model.in_features == 1
    assert model.out_features == 1
    assert model.bias is not None
    assert model.weight.requires_grad and model.bias.requires_grad
    inputs = make_inputs([-2.0, -1.0, 0.0, 1.0, 2.0])
    predictions = model(inputs)
    assert predictions.shape == (5, 1)
    assert predictions.dtype == torch.float32
    loss = mse_loss(predictions, make_targets(inputs))
    loss.backward()
    assert model.weight.grad is not None
    assert model.bias.grad is not None
    assert torch.isfinite(model.weight.grad).all()
    assert torch.isfinite(model.bias.grad).all()
