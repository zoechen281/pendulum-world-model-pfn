"""单回合采集的结束边界、数据快照与真实环境复现检查。"""

import gymnasium as gym
import numpy as np
import pytest

from src.day03_collection import collect_episode


class TwoStepEnv(gym.Env):
    def __init__(self, ending):
        self.action_space = gym.spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32)
        self.ending = ending

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.buffers = [np.array([0., 0., 0.], dtype=np.float32)]
        return self.buffers[-1], {}

    def step(self, action):
        assert self.steps < 2, 'Step called after episode ended'
        self.steps += 1
        observation = np.full(3, self.steps, dtype=np.float32)
        self.buffers.append(observation)
        done = self.steps == 2
        return observation, -float(self.steps), done and self.ending == 'terminated', done and self.ending == 'truncated', {}


@pytest.mark.parametrize('ending', ['terminated', 'truncated'])
def test_episode_boundary_and_snapshots(ending):
    env = TwoStepEnv(ending)
    records = collect_episode(env, seed=7)
    assert len(records) == 2
    assert records[-1][ending]
    assert [r['reward'] for r in records] == [-1., -2.]
    np.testing.assert_array_equal(records[0]['next_observation'], records[1]['observation'])
    assert not np.shares_memory(records[0]['next_observation'], records[1]['observation'])
    for buffer in env.buffers:
        buffer.fill(99.)
    np.testing.assert_array_equal(records[0]['observation'], np.zeros(3))
    np.testing.assert_array_equal(records[-1]['next_observation'], np.full(3, 2.))


def test_real_pendulum_episode_and_seed_reproducibility():
    env = gym.make('Pendulum-v1')
    try:
        first = collect_episode(env, seed=0)
        repeated = collect_episode(env, seed=0)
        assert len(first) == 200
        assert not any(r['terminated'] for r in first)
        assert not any(r['truncated'] for r in first[:-1])
        assert first[-1]['truncated']
        for i, (record, duplicate) in enumerate(zip(first, repeated)):
            assert set(record) == {'observation', 'action', 'reward', 'next_observation', 'terminated', 'truncated'}
            assert env.observation_space.contains(record['observation'])
            assert env.observation_space.contains(record['next_observation'])
            assert env.action_space.contains(record['action'])
            assert np.isfinite(record['reward'])
            for key in record:
                np.testing.assert_array_equal(record[key], duplicate[key])
            if i:
                np.testing.assert_array_equal(first[i-1]['next_observation'], record['observation'])
    finally:
        env.close()
