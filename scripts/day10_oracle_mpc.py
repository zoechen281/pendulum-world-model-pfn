"""P0-2：评分核心由学习者实现，共同候选和统一选择由导师接入。

实验矩阵/成功标准/判读规则见docs/oracle_protocol.md。
完整网格尚未运行，默认入口只说明状态。
"""

from dataclasses import asdict, dataclass
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch

from src.day04_world_model import DynamicsMLP
from src.day06_mpc import score_action_sequences


@dataclass(frozen=True)
class OracleConfig:
    horizons: tuple[int, ...] = (5, 15, 30)
    candidate_counts: tuple[int, ...] = (20, 200)
    environment_seeds: tuple[int, ...] = (100, 101, 102)
    sampling_seed: int = 0
    random_action_seed: int = 0
    max_real_steps: int = 200
    upright_angle: float = 0.2
    upright_speed: float = 1.0
    hold_steps: int = 20


OUTPUT_PREFIX = "results/day10_oracle_"
LEARNED_CHECKPOINT = "results/day10_convergence_20261017T221844_116308"


def score_oracle_sequences(
    simulation_env,
    initial_state: np.ndarray,
    candidate_actions: np.ndarray,
) -> np.ndarray:
    """学习者写评分；独立Gymnasium仿真环境，不传真实执行env。

    initial_state: (2,)精确内部[theta,omega]副本；
    candidate_actions: (N,H,1)，float32力矩[-2,2]；返回(N,)累计真实模拟奖励。
    同一候选从相同state开始，reward用执行动作前状态，不折扣、不附加终端价值。
    """
    num_candidates = candidate_actions.shape[0]
    scores = np.zeros(num_candidates, dtype=np.float64)
    for candidate_index in range(num_candidates):
        simulation_env.unwrapped.state = initial_state.copy()
        for action in candidate_actions[candidate_index]:
            reward = simulation_env.unwrapped.step(action)[1]
            scores[candidate_index] += reward

    # TODO（学习者核心）：每个候选恢复独立仿真状态，再按该动作序列模拟step。
    # TODO：累计原环境奖励；不能修改initial_state/candidate_actions或真实执行env。
    # TODO：不能在不同候选之间接续上个末状态，不能靠重设初始seed替代当前state。
    # 导师辅助后续提供：环境隔离/恢复测试、最大bank前缀配对、统一规划器和归档。
    return scores


def candidate_bank_for_step(real_step: int, sampling_seed: int = 0) -> np.ndarray:
    """导师辅助：同step/seed始终同bank，不依赖评分次数或模式调用顺序。"""
    if real_step < 0 or sampling_seed < 0:
        raise ValueError("step和seed须非负")
    rng = np.random.default_rng(np.random.SeedSequence([sampling_seed, real_step]))
    return rng.uniform(-2.0, 2.0, size=(200, 30, 1)).astype(np.float32)


def load_learned_backend():
    """冻结为P0-1最佳Adam模型及对应全train统计量，不按oracle结果选权重。"""
    folder = Path(__file__).resolve().parents[1] / LEARNED_CHECKPOINT
    weight_path, config_path = folder / "best_weights.json", folder / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights = json.loads(weight_path.read_text(encoding="utf-8"))
    model = DynamicsMLP()
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()})
    model.eval()
    mean = torch.tensor(config["input_mean"], dtype=torch.float32)
    std = torch.tensor(config["input_std"], dtype=torch.float32)
    hashes = {str(p.relative_to(folder.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (weight_path, config_path)}
    return model, mean, std, hashes


@torch.no_grad()
def plan_from_bank(mode, observation, real_env, simulation_env, model,
                   input_mean, input_std, candidate_bank, num_candidates, horizon):
    """统一评分分派、argmax及首动作；环境创建/真实step由外层负责。

    两后端取同bank前缀，仅评分方法不同；返回(action (1,), scores (N,))。
    oracle只访问真实state副本，在独立无渲染模拟环境内step。
    """
    if mode not in ("learned", "oracle"):
        raise ValueError(f"未知模式：{mode}")
    if real_env.unwrapped is simulation_env.unwrapped:
        raise ValueError("真实执行和候选模拟必须是两个独立环境")
    if not (1 <= num_candidates <= 200 and 1 <= horizon <= 30):
        raise ValueError("候选数/长度超出共同bank")
    assert candidate_bank.shape == (200, 30, 1) and candidate_bank.dtype == np.float32
    assert np.isfinite(candidate_bank).all() and (np.abs(candidate_bank) <= 2).all()
    assert simulation_env.unwrapped.render_mode is None
    assert not model.training
    state_before = real_env.unwrapped.state.copy()
    steps_before = real_env._elapsed_steps
    last_u_before = real_env.unwrapped.last_u
    # 已由学习者核对的切片；显式copy隔离输入bank。
    candidate_actions = candidate_bank[:num_candidates, :horizon].copy()
    if mode == "oracle":
        scores = score_oracle_sequences(simulation_env, state_before.copy(), candidate_actions)
    else:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, 3)
        scores = score_action_sequences(model, obs, torch.from_numpy(candidate_actions),
                                        input_mean, input_std).cpu().numpy().astype(np.float64)
    assert scores.shape == (num_candidates,) and np.isfinite(scores).all()
    np.testing.assert_array_equal(real_env.unwrapped.state, state_before)
    assert real_env._elapsed_steps == steps_before and real_env.unwrapped.last_u == last_u_before
    # 复用原random shooting的argmax和首动作规则，两种后端不另写选择逻辑。
    best_index = int(np.argmax(scores))
    return candidate_actions[best_index, 0].copy(), scores.copy()


def summarize_real_states(post_action_states, config):
    """导师记录辅助：仅统计执行后的真实状态，区分经过直立与稳定保持。"""
    states = np.asarray(post_action_states, dtype=np.float64)
    assert states.ndim == 2 and states.shape[1] == 2 and len(states) > 0
    angles = np.arctan2(np.sin(states[:, 0]), np.cos(states[:, 0]))
    reached = np.abs(angles) < config.upright_angle
    stable = reached & (np.abs(states[:, 1]) < config.upright_speed)
    streak = longest = 0
    for flag in stable:
        streak = streak + 1 if flag else 0
        longest = max(longest, streak)
    return dict(
        reached_upright=bool(reached.any()),
        first_upright_step=int(np.flatnonzero(reached)[0]) + 1 if reached.any() else None,
        longest_hold_steps=longest, success=longest >= config.hold_steps,
        final_hold_window_stable=bool(len(stable) >= config.hold_steps
                                      and stable[-config.hold_steps:].all()),
        stable_mask=stable.tolist(),
        metric_scope="post-action real states only; initial reset state excluded",
    )


@torch.no_grad()
def run_episode(mode, env_seed, num_candidates, horizon, backend, config=OracleConfig()):
    """导师闭环/日志辅助；两MPC模式共用本流程，random仅替换动作来源。

    backend为固定恢复的(model, mean, std, checkpoint_hashes)。不创建优化器，
    每步仅执行最佳序列首动作，以环境返回的新观测重新规划。
    """
    if mode not in ("learned", "oracle", "random"):
        raise ValueError(f"未知模式：{mode}")
    if not 1 <= config.max_real_steps <= 200:
        raise ValueError("真实回合步数应在1至200之间")
    model, mean, std, checkpoint_hashes = backend
    assert not model.training
    parameters_before = {k: v.clone() for k, v in model.state_dict().items()}
    observations, states, actions, rewards, planning_times, plan_records = [], [], [], [], [], []
    started = time.perf_counter()
    with ExitStack() as stack:
        real = gym.make("Pendulum-v1")
        stack.callback(real.close)
        simulation = gym.make("Pendulum-v1")
        stack.callback(simulation.close)
        observation, _ = real.reset(seed=env_seed)
        simulation.reset(seed=0)
        real.action_space.seed(config.random_action_seed)
        observations.append(observation.tolist())
        states.append(real.unwrapped.state.copy().tolist())
        dt = float(real.unwrapped.dt)
        for step in range(config.max_real_steps):
            planning_started = time.perf_counter()
            if mode == "random":
                action = real.action_space.sample()
            else:
                bank = candidate_bank_for_step(step, config.sampling_seed)
                action, scores = plan_from_bank(mode, observation, real, simulation,
                                                model, mean, std, bank, num_candidates, horizon)
                plan_records.append(dict(
                    real_step=step,
                    bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),
                    candidate_prefix_sha256=hashlib.sha256(
                        bank[:num_candidates, :horizon].tobytes()).hexdigest(),
                    selected_index=int(np.argmax(scores)),
                    selected_predicted_return=float(np.max(scores)),
                ))
            planning_times.append(time.perf_counter() - planning_started)
            assert real.action_space.contains(action)
            observation, reward, terminated, truncated, _ = real.step(action)
            assert real._elapsed_steps == step + 1
            actions.append(action.tolist())
            rewards.append(float(reward))
            observations.append(observation.tolist())
            states.append(real.unwrapped.state.copy().tolist())
            if terminated or truncated:
                break
    assert len(observations) == len(states) == len(actions) + 1
    assert np.isfinite(states).all() and np.isfinite(rewards).all()
    assert all(torch.equal(v, parameters_before[k]) for k, v in model.state_dict().items())
    metrics = summarize_real_states(states[1:], config)
    return dict(
        mode=mode, env_seed=env_seed, num_candidates=None if mode == "random" else num_candidates,
        horizon=None if mode == "random" else horizon, real_steps=len(actions), dt=dt,
        real_total_reward=float(sum(rewards)), **metrics,
        longest_hold_seconds=metrics["longest_hold_steps"] * dt,
        observations=observations, internal_real_states=states, actions=actions, rewards=rewards,
        planning_seconds_by_step=planning_times, planning_seconds=float(sum(planning_times)),
        total_seconds=time.perf_counter() - started, plans=plan_records,
        candidate_transitions=0 if mode == "random" else len(actions) * num_candidates * horizon,
        oracle_simulated_transitions=len(actions) * num_candidates * horizon if mode == "oracle" else 0,
        checkpoint_sha256=checkpoint_hashes, model_parameters_unchanged=True,
        training_performed=False,
    )


def experiment_cases(config=OracleConfig()):
    """固定调度：每种子只跑一次random，同设定learned/oracle相邻。"""
    cases = []
    for seed in config.environment_seeds:
        cases.append(dict(mode="random", env_seed=seed, num_candidates=0, horizon=0))
        for horizon in config.horizons:
            for count in config.candidate_counts:
                for mode in ("learned", "oracle"):
                    cases.append(dict(mode=mode, env_seed=seed, num_candidates=count, horizon=horizon))
    return cases


def grid_budget(config=OracleConfig()):
    cases = experiment_cases(config)
    return dict(
        episodes=len(cases), real_steps_max=len(cases) * config.max_real_steps,
        oracle_simulated_transitions_max=sum(
            c["num_candidates"] * c["horizon"] * config.max_real_steps
            for c in cases if c["mode"] == "oracle"),
    )


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def run_grid(output_dir=None):
    """导师调度/归档辅助：冻结原网格，逐回合保存，失败保留已完成证据。"""
    config = OracleConfig()
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone(timedelta(hours=8)))
    output = Path(output_dir) if output_dir is not None else root / (
        OUTPUT_PREFIX + stamp.strftime("%Y%m%dT%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    snapshots = output / "snapshots"
    snapshots.mkdir()
    sources = [Path(__file__).resolve(), root / "src/day04_world_model.py",
               root / "src/day06_mpc.py", root / "docs/oracle_protocol.md",
               root / "results/day03_pendulum_dataset.json",
               root / LEARNED_CHECKPOINT / "best_weights.json",
               root / LEARNED_CHECKPOINT / "config.json"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source_hashes = {str(p.relative_to(root)): digest(p) for p in sources}
    for p in sources[:4]:
        (snapshots / p.name).write_bytes(p.read_bytes())
    cases = experiment_cases(config)
    metadata = dict(
        recorded_at=stamp.isoformat(), config=asdict(config), budget=grid_budget(config),
        cases=cases, source_sha256=source_hashes, learned_checkpoint=LEARNED_CHECKPOINT,
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, gymnasium=gym.__version__),
        device="cpu", torch_threads=1, discount=1.0, terminal_value=0.0,
        candidate_pairing="NumPy SeedSequence([sampling_seed, real_step]); shared (200,30,1) bank prefixes",
        success_rule="post-action real states: abs(wrapped theta)<0.2 and abs(omega)<1 for >=20 consecutive steps",
        interpretation_rules=[
            "Oracle failure supports insufficiency of this planning setting/budget in this case, not a unique cause.",
            "Oracle success and learned failure supports dynamics error as a limiting factor for this paired setting.",
            "Both succeed does not establish exact learned dynamics; mixed outcomes must be retained.",
            "Three development seeds and one sampling seed do not prove generalization.",
        ], training_performed=False, test_evaluated=False,
    )
    _write_json(output / "config.json", metadata)
    with (output / "stdout.txt").open("x", encoding="utf-8"):
        pass
    def log(message):
        print(message, flush=True)
        with (output / "stdout.txt").open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    completed = []
    started = time.perf_counter()
    log(f"ARCHIVE: {output}")
    try:
        torch.set_num_threads(1)
        backend = load_learned_backend()
        model, mean, std, _ = backend
        _write_json(output / "frozen_model.json", dict(
            checkpoint_sha256=backend[3], input_mean=mean.tolist(), input_std=std.tolist(),
            state_dict={k: v.tolist() for k, v in model.state_dict().items()}))
        for index, case in enumerate(cases, 1):
            log(f"[{index}/{len(cases)}] start {case}")
            report = run_episode(**case, backend=backend, config=config)
            assert report["real_steps"] == config.max_real_steps
            filename = f"{index:02d}_{case['mode']}_seed{case['env_seed']}_N{case['num_candidates']}_H{case['horizon']}.json"
            # 先落盘整回合，再做配对核对；异常时该回合证据也不丢失。
            _write_json(output / "episodes" / filename, report)
            completed.append(dict(case=case, file="episodes/" + filename, report=report))
            if case["mode"] == "oracle":
                paired = completed[-2]["report"]
                assert paired["mode"] == "learned"
                np.testing.assert_array_equal(paired["internal_real_states"][0], report["internal_real_states"][0])
                for field in ("bank_sha256", "candidate_prefix_sha256"):
                    assert [p[field] for p in paired["plans"]] == [p[field] for p in report["plans"]]
            log(f"[{index}/{len(cases)}] return={report['real_total_reward']:.6f}, "
                  f"hold={report['longest_hold_steps']}, success={report['success']}, "
                  f"seconds={report['total_seconds']:.2f}")
        assert all(digest(p) == source_hashes[str(p.relative_to(root))] for p in sources)
        summary = [{k: item["report"][k] for k in (
            "mode", "env_seed", "num_candidates", "horizon", "real_steps", "real_total_reward",
            "reached_upright", "longest_hold_steps", "success", "final_hold_window_stable",
            "planning_seconds", "total_seconds", "oracle_simulated_transitions")}
                   | dict(file=item["file"]) for item in completed]
        _write_json(output / "comparison.json", dict(
            episodes=summary, elapsed_seconds=time.perf_counter() - started,
            historical_files_unchanged=True, training_performed=False, test_evaluated=False,
            paired_candidates_verified=True, completed_episodes=len(completed)))
        rows = ["# Oracle MPC固定网格结果", "",
                "相同random shooting，只替换评分动力学；learned固定P0-1最佳Adam模型。",
                "", "|seed|模式|N|H|真实回报|最长保持步数|成功|末20步稳定|",
                "|---:|---|---:|---:|---:|---:|---|---|"]
        for r in summary:
            rows.append(f"|{r['env_seed']}|{r['mode']}|{r['num_candidates']}|{r['horizon']}|"
                        f"{r['real_total_reward']:.6f}|{r['longest_hold_steps']}|{r['success']}|{r['final_hold_window_stable']}|")
        rows += ["", "判读规则见运行前config.json；保持成功按执行后的真实状态计算，不把经过直立算保持。",
                 "三开发种子、单采样种子；oracle使用精确内部状态是诊断特权。失败结果全部保留。",
                 "未训练、未评估test、原来源哈希不变；完整轨迹/候选指纹/耗时见各episodes文件。"]
        with (output / "SUMMARY.md").open("x", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        _write_json(output / "status.json", dict(status="complete", completed_episodes=len(completed)))
    except (Exception, KeyboardInterrupt) as error:
        import traceback
        _write_json(output / "status.json", dict(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=repr(error), traceback=traceback.format_exc(),
            completed_files=[item["file"] for item in completed],
            elapsed_seconds=time.perf_counter() - started))
        raise
    return output


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="固定Oracle MPC网格；默认不运行。")
    parser.add_argument("--run-grid", action="store_true")
    args = parser.parse_args()
    if args.run_grid:
        run_grid()
    else:
        print(f"NOT RUN：完整网格预算{grid_budget()}；显式--run-grid才运行，无训练。")


if __name__ == "__main__":
    main()
