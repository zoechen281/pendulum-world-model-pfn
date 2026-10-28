"""仅检查当前练习的输入输出契约，不提供实现。"""

import torch

from src.day01_regression import make_inputs


def test_make_inputs_column_shape_dtype_and_values():
    values = [-2.0, -1.0, 0.0, 1.0]
    result = make_inputs(values)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (4, 1)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert result.tolist() == [[-2.0], [-1.0], [0.0], [1.0]]


def test_make_inputs_single_sample():
    result = make_inputs([2.0])
    assert result.shape == (1, 1)
    assert result.dtype == torch.float32
    assert result.tolist() == [[2.0]]
