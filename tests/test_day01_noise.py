"""验证带噪声目标的数据约定，不训练模型。"""

import torch

from src.day01_regression import make_inputs, make_noisy_targets


def test_zero_noise_preserves_clean_targets_and_input():
    inputs = make_inputs([-2.0, 0.0, 2.0])
    original = inputs.clone()
    output = make_noisy_targets(inputs, noise_std=0.0)
    torch.testing.assert_close(output, torch.tensor([[3.0], [1.0], [15.0]]))
    assert output.shape == inputs.shape
    assert output.dtype == torch.float32
    assert torch.equal(inputs, original)


def test_known_noise_is_scaled_and_added(monkeypatch):
    inputs = make_inputs([2.0, 2.0])
    monkeypatch.setattr(
        torch, "randn_like", lambda tensor: tensor.new_tensor([[-1.5], [2.0]])
    )
    output = make_noisy_targets(inputs, noise_std=0.2)
    torch.testing.assert_close(output, torch.tensor([[14.7], [15.4]]))


def test_caller_seed_reproduces_outputs_without_internal_reset():
    inputs = make_inputs([0.0] * 16)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        first = make_noisy_targets(inputs)
        second = make_noisy_targets(inputs)
        torch.manual_seed(42)
        repeated = make_noisy_targets(inputs)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)
