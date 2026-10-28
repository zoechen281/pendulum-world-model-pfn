"""统一规划辅助的有限检查；无训练/完整闭环实验。"""
from contextlib import ExitStack

import numpy as np
import pytest

from scripts.day10_oracle_mpc import (
    candidate_bank_for_step, load_learned_backend, plan_from_bank,
)
from tests.test_day10_oracle_scoring import make_env


def test_common_bank_is_prefix_paired_and_call_order_independent():
    bank = candidate_bank_for_step(7)
    candidate_bank_for_step(8)
    np.testing.assert_array_equal(bank, candidate_bank_for_step(7))
    assert bank[:20, :5].shape == (20, 5, 1)
    np.testing.assert_array_equal(bank[:20, :5], bank[:200, :30][:20, :5])
    assert bank.dtype == np.float32 and (np.abs(bank) <= 2).all()
    assert not np.array_equal(bank, candidate_bank_for_step(8))


def test_both_backends_share_candidates_first_action_and_real_env_isolation():
    with ExitStack() as stack:
        real, simulation = make_env(stack), make_env(stack)
        observation = real.reset(seed=101)[0]
        model, mean, std, hashes = load_learned_backend()
        assert len(hashes) == 2
        bank = candidate_bank_for_step(0)
        # H=1时两种后端都按相同当前真实观测算即时奖励，无未来模型误差混入。
        # 第2个候选为唯一零力矩，应该选它；其余候选力矩均为1。
        bank[:20, 0, 0] = 1.0
        bank[1, 0, 0] = 0.0
        before = bank.copy()
        actions, scores = {}, {}
        for mode in ("learned", "oracle"):
            actions[mode], scores[mode] = plan_from_bank(
                mode, observation, real, simulation, model, mean, std, bank, 20, 1)
        np.testing.assert_allclose(scores["learned"], scores["oracle"], rtol=0, atol=2e-6)
        np.testing.assert_array_equal(actions["learned"], bank[1, 0])
        np.testing.assert_array_equal(actions["oracle"], bank[1, 0])
        np.testing.assert_array_equal(bank, before)
        assert real._elapsed_steps == 0
        assert all(p.grad is None for p in model.parameters())


def test_same_environment_is_rejected_before_simulation():
    with ExitStack() as stack:
        env = make_env(stack)
        observation = env.reset(seed=100)[0]
        model, mean, std, _ = load_learned_backend()
        with pytest.raises(ValueError, match="独立环境"):
            plan_from_bank("oracle", observation, env, env, model, mean, std,
                           candidate_bank_for_step(0), 20, 5)
        assert env._elapsed_steps == 0
