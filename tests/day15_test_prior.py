"""仅检查Day15未裁剪增量公式；不执行训练或读取真实数据。"""

import torch
from src.day15_pfn_prior import pendulum_delta_omega


def test_default_scalar_value():
    out = pendulum_delta_omega(
        torch.tensor(0.5), torch.tensor(1.0), torch.tensor(15.0), torch.tensor(3.0)
    )
    assert out.shape == torch.Size([])
    torch.testing.assert_close(out, torch.tensor(0.525))


def test_zero_inputs():
    out = pendulum_delta_omega(
        torch.zeros(3, 1), torch.zeros(3, 1), torch.tensor(15.0), torch.tensor(3.0)
    )
    torch.testing.assert_close(out, torch.zeros(3, 1))


def test_task_broadcasting_and_inputs_unchanged():
    sin_theta = torch.tensor([[[0.0], [1.0]], [[0.0], [-1.0]]])
    torque = torch.tensor([[[1.0], [0.0]], [[2.0], [1.0]]])
    a = torch.tensor([15.0, 12.0]).reshape(2, 1, 1)
    b = torch.tensor([3.0, 2.0]).reshape(2, 1, 1)
    inputs = (sin_theta, torque, a, b)
    originals = tuple(t.clone() for t in inputs)
    out = pendulum_delta_omega(*inputs)
    assert out.shape == (2, 2, 1)
    expected = torch.tensor([[[0.15], [0.75]], [[0.20], [-0.50]]])
    torch.testing.assert_close(out, expected)
    for current, original in zip(inputs, originals):
        assert torch.equal(current, original)


def test_custom_dt():
    out = pendulum_delta_omega(
        torch.tensor(0.5), torch.tensor(1.0), torch.tensor(15.0), torch.tensor(3.0), dt=0.1
    )
    torch.testing.assert_close(out, torch.tensor(1.05))
