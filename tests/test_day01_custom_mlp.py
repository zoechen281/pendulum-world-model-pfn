"""自定义 Module 的参数注册、前向计算和梯度检查。"""

import torch

from src.day01_regression import RegressionMLP, make_inputs, mse_loss


def test_registered_parameters_and_output_shape():
    model = RegressionMLP()
    parameters = dict(model.named_parameters())
    assert set(parameters) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}
    assert sum(parameter.numel() for parameter in parameters.values()) == 49
    output = model(make_inputs([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert output.shape == (5, 1)
    assert output.dtype == torch.float32


def test_forward_values_and_gradients_reach_both_layers():
    model = RegressionMLP()
    with torch.no_grad():
        model.fc1.weight.fill_(1.0)
        model.fc1.bias.zero_()
        model.fc2.weight.fill_(1.0)
        model.fc2.bias.zero_()
    # 每个隐藏单元为 max(x, 0)，输出应为 16 * max(x, 0)。
    inputs = make_inputs([-1.0, 1.0, 2.0])
    predictions = model(inputs)
    torch.testing.assert_close(predictions, torch.tensor([[0.0], [16.0], [32.0]]))
    loss = mse_loss(predictions, torch.zeros_like(predictions))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name
