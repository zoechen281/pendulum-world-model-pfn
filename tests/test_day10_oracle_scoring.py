"""Oracle评分协议测试；只做少量环境step，不训练、不运行控制网格。"""

from contextlib import ExitStack

import gymnasium as gym
import numpy as np

from scripts.day10_oracle_mpc import score_oracle_sequences


def make_env(stack, seed=100):
    env = gym.make("Pendulum-v1")
    stack.callback(env.close)
    env.reset(seed=seed)
    return env


def independent_scores(env, state, candidates):
    """每条候选单独reset，用带包装器的真实step作为参考。"""
    values = []
    for sequence in candidates:
        env.reset(seed=123)
        env.unwrapped.state = state.copy()
        rewards = [env.step(action)[1] for action in sequence]
        values.append(sum(rewards))
    return np.asarray(values, dtype=np.float64)


def test_single_step_reward_uses_pre_action_state():
    with ExitStack() as stack:
        simulation = make_env(stack)
        state = np.array([0.4, -0.7], dtype=np.float64)
        candidates = np.array([[[-2.0]], [[0.0]], [[1.5]]], dtype=np.float32)
        scores = score_oracle_sequences(simulation, state, candidates)
        expected = -(0.4**2 + 0.1 * (-0.7)**2
                     + 0.001 * candidates[:, 0, 0].astype(np.float64)**2)
        assert scores.shape == (3,)
        assert scores.dtype == np.float64
        # 环境内力矩项按float32动作参与运算，解析float64公式允许舍入差异。
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-8)


def test_multistep_matches_independent_simulation_and_candidate_order():
    with ExitStack() as stack:
        simulation = make_env(stack)
        reference = make_env(stack)
        # 非reset初值且靠近速度裁剪边界，包含多步动力学和合法边界动作。
        state = np.array([1.3, 7.95], dtype=np.float64)
        candidates = np.array([
            [[2.0], [2.0], [-2.0], [0.0], [1.0]],
            [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0]],
        ], dtype=np.float32)
        expected = independent_scores(reference, state, candidates)
        scores = score_oracle_sequences(simulation, state, candidates)
        reversed_scores = score_oracle_sequences(simulation, state, candidates[::-1])
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reversed_scores[::-1], scores, rtol=0, atol=1e-12)


def test_inputs_and_real_env_unchanged_and_imagination_bypasses_time_limit():
    with ExitStack() as stack:
        real_env = make_env(stack, seed=101)
        simulation = make_env(stack)
        reference = make_env(stack)
        real_env.step(np.array([0.25], dtype=np.float32))
        real_state_before = real_env.unwrapped.state.copy()
        real_steps_before = real_env._elapsed_steps
        real_last_u = real_env.unwrapped.last_u
        state = real_state_before.copy()
        candidates = np.array([[[1.0]] * 5, [[-1.0]] * 5], dtype=np.float32)
        state_before, candidates_before = state.copy(), candidates.copy()
        # 下一个包装器step原本会触及TimeLimit，评分应只推进底层环境。
        simulation._elapsed_steps = 199
        scores = score_oracle_sequences(simulation, state, candidates)
        np.testing.assert_allclose(scores, independent_scores(reference, state, candidates),
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(state, state_before)
        np.testing.assert_array_equal(candidates, candidates_before)
        np.testing.assert_array_equal(real_env.unwrapped.state, real_state_before)
        assert real_env._elapsed_steps == real_steps_before
        assert real_env.unwrapped.last_u == real_last_u
        assert simulation._elapsed_steps == 199
