"""恒定合力解析解的已知物理结果。"""

import pytest

from src.day02_dynamics import exact_constant_force


@pytest.mark.parametrize(
    'arguments, expected',
    [
        ((3.0, -2.0, 4.0, 2.0, 0.0), (3.0, -2.0)),
        ((0.0, 1.0, 4.0, 2.0, 2.0), (6.0, 5.0)),
        ((3.0, -2.0, 0.0, 1.0, 2.0), (-1.0, -2.0)),
        ((0.0, 1.0, -4.0, 2.0, 2.0), (-2.0, -3.0)),
    ],
)
def test_exact_constant_force(arguments, expected):
    assert exact_constant_force(*arguments) == pytest.approx(expected)
