"""小型采集入口；NumPy 转 JSON 和环境清理由导师提供。"""

import hashlib
import json
import platform
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.day03_collection import collect_episodes, split_episodes


def encode_numpy(value):
    """把 NumPy 数组/标量转换成 JSON 支持的普通列表/数值。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported value type: {type(value).__name__}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output_path = root / "results/day03_pendulum_dataset.json"
    if output_path.exists():
        raise FileExistsError(f"保留已有数据集，不覆盖：{output_path}")

    env = gym.make("Pendulum-v1")
    try:
        # TODO 1：调用 collect_episodes，采集 10 条轨迹，base_seed=0，保存为 episodes。
        episodes = collect_episodes(env, num_episodes=10, base_seed=0)
        # TODO 2：调用 split_episodes，训练 6 条、验证 2 条，其余测试，保存为 splits。
        splits = split_episodes(episodes, train_count=6, val_count=2)
        # TODO 3：将 splits 作为 dataset 的 "splits" 字段，构建完整的 dataset。
        dataset = {
            "metadata": {
                "env_id": "Pendulum-v1",
                "gymnasium_version": gym.__version__,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "base_seed": 0,
                "episode_seeds": {
                    "train": list(range(6)),
                    "val": [6, 7],
                    "test": [8, 9],
                },
                "policy": "uniform random actions via seeded action_space.sample",
                "dt": env.unwrapped.dt,
                "max_episode_steps": env.spec.max_episode_steps,
                "source_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in [
                        "src/day03_collection.py",
                        "scripts/day03_save_dataset.py",
                    ]
                },
            },
            "splits": splits,
        }
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(
                dataset,
                handle,
                default=encode_numpy,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        print(output_path)
        for name, trajectories in splits.items():
            print(
                f"{name}: {len(trajectories)} episodes, {sum(len(t) for t in trajectories)} transitions"
            )
    finally:
        env.close()
