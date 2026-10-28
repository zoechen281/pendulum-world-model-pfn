"""已有模型的单回合 MPC/随机策略对照；不训练或保存权重。"""
import argparse
import hashlib
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from src.day04_world_model import DynamicsMLP
from src.day06_mpc import plan_random_shooting

def run_random_check(env_seed: int, action_seed: int, max_steps: int) -> float:
    """独立环境中运行随机动作策略，返回真实累计奖励。

    环境种子和步数与 MPC 一致；动作随机数单独设种子以复现。
    这只是单初始状态的短对照，不构成多种子完整控制评价。
    """
    random_env = gym.make('Pendulum-v1')
    try:
        observation, info = random_env.reset(seed=env_seed)
        random_env.action_space.seed(action_seed)
        random_total_reward = 0.0
        steps_executed = 0
        for step in range(max_steps):
            action = random_env.action_space.sample()
            assert random_env.action_space.contains(action)
            next_observation, reward, terminated, truncated, info = random_env.step(action)
            random_total_reward += reward
            steps_executed += 1
            print(f'第 {step + 1} 步：action={action[0]:.6f}，reward={reward:.6f}')
            if terminated or truncated:
                break
        assert 1 <= steps_executed <= max_steps
        assert np.isfinite(random_total_reward)
        return random_total_reward
    finally:
        random_env.close()

def main(env_seed: int=100):
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / 'results/day04_delta_normalized_run.json'
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    record = json.loads(checkpoint_bytes.decode('utf-8'))
    torch.set_num_threads(1)
    model = DynamicsMLP()
    model.load_state_dict({name: torch.tensor(values, dtype=torch.float32) for name, values in record['final_state_dict'].items()})
    model.eval()
    parameters_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    input_mean = torch.tensor(record['input_mean'], dtype=torch.float32)
    input_std = torch.tensor(record['input_std'], dtype=torch.float32)
    generator = torch.Generator(device='cpu').manual_seed(0)
    num_candidates = 20
    horizon = 3
    max_steps = 200
    print(f'环境种子={env_seed}，候选数={num_candidates}，规划步数={horizon}，最多执行={max_steps}步')
    env = gym.make('Pendulum-v1')
    try:
        observation, info = env.reset(seed=env_seed)
        total_reward = 0.0
        steps_executed = 0
        for step in range(max_steps):
            initial_observation = torch.tensor(observation, dtype=torch.float32).reshape(1, 3)
            best_action = plan_random_shooting(model, initial_observation, input_mean, input_std, num_candidates, horizon, generator)
            action = best_action.detach().cpu().numpy()
            assert env.action_space.contains(action)
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            observation = next_observation
            steps_executed += 1
            print(f'第 {step + 1} 步：action={action[0]:.6f}，reward={reward:.6f}')
            if terminated or truncated:
                break
        assert 1 <= steps_executed <= max_steps
        assert np.isfinite(total_reward)
        assert all((torch.equal(p, parameters_before[name]) for name, p in model.named_parameters()))
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
        print(f'执行 {steps_executed} 步，真实累计奖励={total_reward:.6f}')
        print('仅为单初始状态的200步完整回合对照，不能代表其他种子的表现。')
    finally:
        env.close()
    random_total_reward = run_random_check(env_seed, 0, max_steps)
    print(f'随机策略真实累计奖励={random_total_reward:.6f}')
    print(f'MPC 累计奖励减随机累计奖励={total_reward - random_total_reward:.6f}')
    return dict(mpc_return=float(total_reward), random_return=float(random_total_reward), mpc_steps=int(steps_executed))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行一个环境种子的 MPC/随机策略对照')
    parser.add_argument('--env-seed', type=int, default=100, help='非负环境种子，默认 100')
    args = parser.parse_args()
    if args.env_seed < 0:
        parser.error('--env-seed 必须为非负整数')
    main(env_seed=args.env_seed)