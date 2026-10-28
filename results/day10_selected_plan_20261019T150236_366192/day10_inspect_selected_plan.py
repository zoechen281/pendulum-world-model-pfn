"""Day10：同一初始状态、同一选中动作序列的预测/真实轨迹诊断。"""

import hashlib
import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import gymnasium as gym
import torch

from src.day04_world_model import DynamicsMLP, rollout_predictions, trajectory_step_mse
from src.day06_mpc import pendulum_rewards, score_action_sequences


def main(env_seed: int = 100):
    # 导师提供的恢复与采样辅助；沿用Day6模型和H5配置，不训练。
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / "results/day04_delta_normalized_run.json"
    original_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(original_bytes).hexdigest()
    record = json.loads(original_bytes.decode("utf-8"))
    torch.set_num_threads(1)
    model = DynamicsMLP()
    model.load_state_dict(
        {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in record["final_state_dict"].items()
        }
    )
    model.eval()
    parameters_before = {name: value.clone() for name, value in model.state_dict().items()}
    input_mean = torch.tensor(record["input_mean"], dtype=torch.float32)
    input_std = torch.tensor(record["input_std"], dtype=torch.float32)
    horizon = 5
    num_candidates = 20
    generator = torch.Generator().manual_seed(0)
    env = gym.make("Pendulum-v1")
    try:
        observation, info = env.reset(seed=env_seed)
        initial_observation = torch.tensor(observation, dtype=torch.float32).reshape(
            1, 3
        )
        candidate_actions = (
            4 * torch.rand(num_candidates, horizon, 1, generator=generator) - 2
        )
        with torch.no_grad():
            # TODO 1（当前步骤）：调用score_action_sequences，将分数保存为scores (20,)。
            # 参数依次为model、initial_observation、candidate_actions、input_mean、input_std。
            scores = score_action_sequences(
                model, initial_observation, candidate_actions, input_mean, input_std
            )
            best_index = torch.argmax(scores)
            selected_actions = candidate_actions[best_index]
            actions = [action.reshape(1, 1) for action in selected_actions]
            trajectory = rollout_predictions(
                model, initial_observation, actions, input_mean, input_std
            )
            predicted_trajectory = torch.cat(trajectory, dim=0)
            predicted_return = scores[best_index].item()

            # 后续：argmax找到best_index；取整条selected_actions (5,1)。
            # 注意是完整序列，而不是MPC实际只执行的第一个动作。
            # 将每步动作保持为(1,1)，调用rollout_predictions，拼成predicted_trajectory (6,3)。
            # predicted_return取scores[best_index]，本项目规划评分不折扣。
        true_observations = [initial_observation.clone()]
        true_rewards = []
        # 模型推演不能推进真实环境，保证预测/实际从同一状态出发。
        assert env._elapsed_steps == 0
        assert torch.equal(torch.tensor(env.unwrapped._get_obs()).reshape(1, 3), initial_observation)
        assert selected_actions.shape == (5, 1) and scores.shape == (20,)
        for action in selected_actions:
            assert env.action_space.contains(action.numpy())
            next_observation, reward, terminated, truncated, info = env.step(
                action.numpy()
            )
            true_observations.append(
                torch.tensor(next_observation, dtype=torch.float32).reshape(1, 3)
            )
            true_rewards.append(reward)
            if terminated or truncated:
                raise RuntimeError("5步诊断发生提前结束，不能按完整5步解释")
        # 后续真实诊断：环境仍停在同一初始状态；按selected_actions执行5步。
        # 保存初始及各next_observation共6项、每步真实reward共5项，检查动作合法及结束标记。
        # 这是一次性执行选中序列的开环诊断，期间不重新规划；不替代Day6的MPC闭环。
        # 比较逐步MSE、预测累计奖励与真实累计奖励。
        # gap=预测回报-真实回报；正值才表示本方案被高估。
        # 局部偏差或gap不能单独证明排序错误、模型误差被系统利用或整个回合失败的因果。
        # 后续只检查env_seed=100和101两个案例；不为挑显著失败而无限搜索。
        true_trajectory = torch.cat(true_observations, dim=0)
        step_mse = trajectory_step_mse(predicted_trajectory, true_trajectory)
        true_return = sum(true_rewards)
        gap = predicted_return - true_return
        print(predicted_return, true_return, gap)
        print(step_mse.tolist())
        # 导师提供的有限协议检查；不改学习者核心计算。
        assert predicted_trajectory.shape == true_trajectory.shape == (6, 3)
        assert step_mse.shape == (6,) and step_mse[0].item() == 0.0
        assert len(true_rewards) == env._elapsed_steps == 5
        assert torch.isfinite(step_mse).all() and not predicted_trajectory.requires_grad
        assert torch.isfinite(predicted_trajectory).all() and torch.isfinite(true_trajectory).all()
        assert all(torch.equal(v, parameters_before[k]) for k, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        predicted_rewards = pendulum_rewards(predicted_trajectory[:-1], selected_actions)
        assert abs(predicted_rewards.sum().item() - predicted_return) < 1e-4
        assert (
            hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == checkpoint_hash
        )
        return dict(
            env_seed=env_seed, horizon=horizon, num_candidates=num_candidates, sampling_seed=0,
            checkpoint=str(checkpoint_path.relative_to(root)), checkpoint_sha256=checkpoint_hash,
            best_index=int(best_index), scores=scores.tolist(), selected_actions=selected_actions.tolist(),
            input_mean=input_mean.tolist(), input_std=input_std.tolist(),
            predicted_trajectory=predicted_trajectory.tolist(), true_trajectory=true_trajectory.tolist(),
            step_mse=step_mse.tolist(), predicted_rewards=predicted_rewards.tolist(),
            true_rewards=[float(x) for x in true_rewards],
            predicted_return=predicted_return, true_return=float(true_return), gap=float(gap),
            real_steps=5, model_parameters_unchanged=True, checkpoint_unchanged=True,
            training_performed=False, test_evaluated=False,
            protocol="Same initial state and selected sequence; open-loop 5 steps, no replanning.",
        )
    finally:
        env.close()


def archive_cases():
    """导师归档辅助：仅固定100/101两个案例，不追加搜索。"""
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "src/day04_world_model.py",
               root / "src/day06_mpc.py", root / "results/day04_delta_normalized_run.json"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output = root / "results" / ("day10_selected_plan_" + datetime.now(
        timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(exist_ok=False)
    for p in sources[:3]:
        (output / p.name).write_bytes(p.read_bytes())
    def write_json(name, data):
        with (output / name).open("x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    write_json("config.json", dict(env_seeds=[100, 101], source_sha256=hashes,
                                  checkpoint="results/day04_delta_normalized_run.json",
                                  horizon=5, num_candidates=20, sampling_seed=0,
                                  training_performed=False, test_evaluated=False,
                                  versions=dict(torch=torch.__version__, gymnasium=gym.__version__)))
    capture = io.StringIO()
    reports = []
    started = time.perf_counter()
    try:
        for seed in (100, 101):
            with redirect_stdout(capture):
                print(f"env_seed={seed}")
                report = main(seed)
            write_json(f"case_seed{seed}.json", report)
            reports.append(report)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))]
                   for p in sources)
        write_json("comparison.json", dict(cases=reports, source_sha256=hashes,
                                           historical_files_unchanged=True,
                                           elapsed_seconds=time.perf_counter()-started))
        rows = ["# Day10：选中序列预测/真实有限案例", "",
                "原Day6旧SGD200模型、H5、20候选、torch采样seed0；同起点同整条动作序列，开环5步，不重新规划。",
                "", "|环境种子|预测回报|真实回报|gap（预测减真实）|第5步MSE|",
                "|---:|---:|---:|---:|---:|"]
        for r in reports:
            rows.append(f"|{r['env_seed']}|{r['predicted_return']:.8f}|{r['true_return']:.8f}|"
                        f"{r['gap']:.8f}|{r['step_mse'][-1]:.8f}|")
        rows += ["", "gap正为高估，负为低估。仅对选中方案做了真实执行，未评价其他候选的真实排序，不能证明选错或系统利用模型误差。",
                 "这是旧模型局部诊断，不是新Adam模型或完整MPC回合结果；不改写P0-2结论。",
                 "奖励按执行动作前状态计算，因此5步回报用状态0至4，第5步末状态误差本身未进入这5个奖励。",
                 "逐步误差/动作/两条轨迹/奖励见各case JSON；参数及来源哈希未变，未训练、未用test。"]
        with (output / "SUMMARY.md").open("x", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
    except (Exception, KeyboardInterrupt) as error:
        import traceback
        write_json("failed.json", dict(error=repr(error), completed_seeds=[r["env_seed"] for r in reports]))
        capture.write(traceback.format_exc())
        raise
    finally:
        with (output / "stdout.txt").open("x", encoding="utf-8") as f:
            f.write(capture.getvalue())
        print(capture.getvalue(), end="")
        print(f"ARCHIVE: {output}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="旧Day6模型的固定有限案例，不训练。")
    parser.add_argument("--archive-cases", action="store_true")
    args = parser.parse_args()
    if args.archive_cases:
        archive_cases()
    else:
        main()
