"""动力学 MLP 的输出、参数注册和可训练性。"""

import torch

from src.day04_world_model import DynamicsMLP


def test_shape_registered_parameters_and_backward():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = DynamicsMLP()
        inputs = torch.randn(8, 4)
    assert sum(p.numel() for p in model.parameters()) == 4675
    predictions = model(inputs)
    assert predictions.shape == (8, 3)
    assert predictions.dtype == torch.float32
    loss = (predictions - 1.0).square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_output_can_represent_negative_values():
    model = DynamicsMLP()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(torch.tensor([-1., -2., -3.]))
    output = model(torch.zeros(1, 4))
    torch.testing.assert_close(output, torch.tensor([[-1., -2., -3.]]))
