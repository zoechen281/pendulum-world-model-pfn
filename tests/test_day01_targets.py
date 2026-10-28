"""检查回归目标的计算结果与样本对应关系。"""

import torch

from src.day01_regression import make_inputs, make_targets


def test_make_targets_values_shape_dtype_and_input_unchanged():
    inputs = make_inputs([-1.0, 0.0, 2.0])
    original = inputs.clone()
    targets = make_targets(inputs)
    assert targets.shape == (3, 1)
    assert targets.dtype == torch.float32
    assert torch.equal(targets, torch.tensor([[0.0], [1.0], [15.0]]))
    assert torch.equal(inputs, original)
