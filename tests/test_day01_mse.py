"""验证手写 MSE 的数值与后续训练所需的梯度。"""

import torch

from src.day01_regression import mse_loss


def test_mse_value_and_scalar_tensor():
    predictions = torch.tensor([[2.0], [4.0]])
    targets = torch.tensor([[1.0], [6.0]])
    result = mse_loss(predictions, targets)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([])
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor(2.5))


def test_mse_preserves_gradient_for_training():
    predictions = torch.tensor([[2.0], [4.0]], requires_grad=True)
    targets = torch.tensor([[1.0], [6.0]])
    result = mse_loss(predictions, targets)
    result.backward()
    assert torch.allclose(predictions.grad, torch.tensor([[1.0], [-2.0]]))
