"""多回合种子传递、边界保留与真实环境检查。"""

import gymnasium as gym
import numpy as np

import src.day03_collection as collection


def test_seeds_and_whole_episode_append(monkeypatch):
    calls = []
    trajectories = []
    env = object()

    def fake_collect(actual_env, seed):
        assert actual_env is env
        calls.append(seed)
        trajectory = [{'seed': seed}, {'seed': seed, 'last': True}]
        trajectories.append(trajectory)
        return trajectory

    monkeypatch.setattr(collection, 'collect_episode', fake_collect)
    result = collection.collect_episodes(env, num_episodes=3, base_seed=10)
    assert calls == [10, 11, 12]
    assert len(result) == 3
    for actual, expected in zip(result, trajectories):
        assert actual is expected


def test_real_episodes_differ_and_batch_reproduces():
    env = gym.make('Pendulum-v1')
    try:
        episodes = collection.collect_episodes(env, 3, base_seed=0)
        repeated = collection.collect_episodes(env, 3, base_seed=0)
        assert [len(episode) for episode in episodes] == [200, 200, 200]
        for i in range(1, 3):
            assert not np.array_equal(episodes[0][0]['observation'], episodes[i][0]['observation'])
        for episode, duplicate in zip(episodes, repeated):
            assert episode[-1]['truncated']
            for record, other in zip(episode, duplicate):
                for key in record:
                    np.testing.assert_array_equal(record[key], other[key])
    finally:
        env.close()
