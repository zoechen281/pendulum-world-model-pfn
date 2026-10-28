"""验证固定参数评估的数值、模式与无梯度前向计算。"""

import pytest
import torch

from src.day01_regression import evaluate_model, make_inputs, make_targets


def test_evaluation_mse_without_tracking_or_parameter_update():
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(1.0)
    model.train()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    inputs = make_inputs([-1.0, 0.0, 1.0])
    targets = make_targets(inputs)
    output_grad_flags = []
    hook = model.register_forward_hook(
        lambda module, args, output: output_grad_flags.append(output.requires_grad)
    )
    try:
        result = evaluate_model(model, inputs, targets)
    finally:
        hook.remove()
    assert isinstance(result, float)
    assert result == pytest.approx(10.0 / 3.0, rel=1e-6)
    assert model.training is False
    assert output_grad_flags == [False]
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter.detach(), original)
        assert parameter.requires_grad
        assert parameter.grad is None
