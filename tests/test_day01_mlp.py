"""检查多层网络的结构、批量形状及参数梯度连接。"""

import torch

from src.day01_regression import build_mlp_model, make_inputs, make_targets, mse_loss


def test_mlp_structure_forward_and_gradients():
    torch.manual_seed(0)
    model = build_mlp_model()
    assert isinstance(model, torch.nn.Sequential)
    assert len(model) == 3
    assert isinstance(model[0], torch.nn.Linear)
    assert (model[0].in_features, model[0].out_features) == (1, 16)
    assert isinstance(model[1], torch.nn.ReLU)
    assert isinstance(model[2], torch.nn.Linear)
    assert (model[2].in_features, model[2].out_features) == (16, 1)

    inputs = make_inputs([-2.0, -1.0, 0.0, 1.0, 2.0])
    hidden = model[0](inputs)
    activated = model[1](hidden)
    assert hidden.shape == activated.shape == (5, 16)
    assert torch.equal(activated, hidden.clamp_min(0))
    predictions = model(inputs)
    assert predictions.shape == (5, 1)
    assert predictions.dtype == torch.float32
    loss = mse_loss(predictions, make_targets(inputs))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
