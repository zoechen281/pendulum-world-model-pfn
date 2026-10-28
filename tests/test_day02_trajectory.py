"""轨迹边界、状态递推和固定总时长的误差检查。"""

import pytest

from src.day02_dynamics import simulate_euler


def test_zero_steps_contains_only_initial_state():
    assert simulate_euler(3.0, -1.0, 4.0, 2.0, 0.5, 0) == [(3.0, -1.0)]


def test_two_steps_reuse_updated_state():
    trajectory = simulate_euler(0.0, 1.0, 4.0, 2.0, 0.5, 2)
    assert trajectory == [(0.0, 1.0), (0.5, 2.0), (1.5, 3.0)]


def test_constant_velocity_trajectory():
    trajectory = simulate_euler(3.0, -2.0, 0.0, 1.0, 0.25, 4)
    assert trajectory == [(3.0, -2.0), (2.5, -2.0), (2.0, -2.0), (1.5, -2.0), (1.0, -2.0)]


def test_smaller_steps_at_same_final_time():
    # 恒加速度为 2，初始速度为 1；t=1 时解析位置为 2、速度为 3。
    coarse = simulate_euler(0.0, 1.0, 4.0, 2.0, 0.5, 2)
    fine = simulate_euler(0.0, 1.0, 4.0, 2.0, 0.25, 4)
    assert len(coarse) == 3
    assert len(fine) == 5
    assert coarse[-1] == pytest.approx((1.5, 3.0))
    assert fine[-1] == pytest.approx((1.75, 3.0))
    assert abs(fine[-1][0] - 2.0) == pytest.approx(abs(coarse[-1][0] - 2.0) / 2)
