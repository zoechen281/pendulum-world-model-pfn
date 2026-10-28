"""显式 Euler 单步的物理例子与更新顺序检查。"""

import pytest

from src.day02_dynamics import euler_step


@pytest.mark.parametrize(
    'arguments, expected',
    [
        ((0.0, 1.0, 4.0, 2.0, 0.5), (0.5, 2.0)),
        ((3.0, -2.0, 0.0, 1.0, 0.25), (2.5, -2.0)),
        ((0.0, 0.0, -6.0, 3.0, 0.5), (0.0, -1.0)),
        ((0.0, 1.0, 4.0, 4.0, 0.5), (0.5, 1.5)),
    ],
)
def test_euler_step_physical_examples(arguments, expected):
    result = euler_step(*arguments)
    assert isinstance(result, tuple)
    assert result == pytest.approx(expected)
