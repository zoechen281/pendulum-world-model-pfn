"""导师提供的结构模型检查：覆盖物理关系、尺度、批量形状和梯度。"""

import torch

from src.day08_physics_model import PhysicsStructuredMLP


def test_zero_delta_integrates_velocity_and_preserves_inputs():
    model = PhysicsStructuredMLP().double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    angles = torch.tensor([0.2, -0.7], dtype=torch.float64)
    velocity = torch.tensor([2.0, -3.0], dtype=torch.float64)
    observations = torch.stack([angles.cos(), angles.sin(), velocity], dim=1)
    actions = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    mean = torch.zeros(1, 4, dtype=torch.float64)
    std = torch.ones(1, 4, dtype=torch.float64)
    originals = [x.clone() for x in (observations, actions, mean, std)]
    output = model(observations, actions, mean, std)
    next_angles = angles + 0.05 * velocity
    expected = torch.stack([next_angles.cos(), next_angles.sin(), velocity], dim=1)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output[:, :2].square().sum(dim=1), torch.ones(2, dtype=output.dtype))
    assert output.shape == (2, 3) and output.dtype == observations.dtype
    assert model(observations[:1], actions[:1], mean, std).shape == (1, 3)
    for value, original in zip((observations, actions, mean, std), originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_normalized_network_input_raw_velocity_and_updated_velocity_integration():
    model = PhysicsStructuredMLP().double()
    # 以确定性的线性头替换网络：Δω = 2 * 标准化动作，检出漏标准化等错误。
    model.network = torch.nn.Linear(4, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.network.weight.zero_()
        model.network.weight[0, 3] = 2.0
    observations = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float64, requires_grad=True)
    actions = torch.tensor([[1.0]], dtype=torch.float64, requires_grad=True)
    mean = torch.tensor([[0.5, 0.1, 1.0, -1.0]], dtype=torch.float64)
    std = torch.tensor([[2.0, 3.0, 2.0, 2.0]], dtype=torch.float64)
    # 标准化动作为1，Δω=2，原速度2加回后为4；下一角度为0.2。
    output = model(observations, actions, mean, std)
    angle = torch.tensor(0.2, dtype=torch.float64)
    expected = torch.stack([angle.cos(), angle.sin(), angle.new_tensor(4.0)]).reshape(1, 3)
    torch.testing.assert_close(output, expected)
    output[:, 2].sum().backward()
    torch.testing.assert_close(actions.grad, torch.ones_like(actions))
    assert observations.grad is not None and torch.isfinite(observations.grad).all()
    assert model.network.weight.grad is not None and torch.isfinite(model.network.weight.grad).all()
    with torch.no_grad():
        evaluated = model(observations, actions, mean, std)
    assert not evaluated.requires_grad


def test_clips_velocity_before_integrating_angle():
    model = PhysicsStructuredMLP().double()
    model.network = torch.nn.Linear(4, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.network.weight.zero_()
        model.network.weight[0, 3] = 4.0
    observations = torch.tensor([[1.0, 0.0, 7.0], [1.0, 0.0, -7.0]], dtype=torch.float64)
    actions = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    output = model(observations, actions, torch.zeros(1, 4, dtype=torch.float64),
                   torch.ones(1, 4, dtype=torch.float64))
    velocity = torch.tensor([8.0, -8.0], dtype=torch.float64)
    angles = velocity * 0.05
    expected = torch.stack([angles.cos(), angles.sin(), velocity], dim=1)
    torch.testing.assert_close(output, expected)
