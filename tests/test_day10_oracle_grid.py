"""调度/归档测试；回合全部用替身，输出只在pytest临时目录，无网格仿真。"""

import json

import pytest

from scripts import day10_oracle_mpc as runner


def fake_episode(mode, env_seed, num_candidates, horizon, backend, config):
    """仅供归档测试的假回合，禁止用于实验报告。"""
    return dict(
        mode=mode, env_seed=env_seed,
        num_candidates=None if mode == "random" else num_candidates,
        horizon=None if mode == "random" else horizon,
        real_steps=200, real_total_reward=-123.0, reached_upright=False,
        longest_hold_steps=0, success=False, final_hold_window_stable=False,
        planning_seconds=0.0, total_seconds=0.0,
        oracle_simulated_transitions=200*num_candidates*horizon if mode == "oracle" else 0,
        internal_real_states=[[0.5, 0.0]] * 201,
        plans=[] if mode == "random" else [dict(bank_sha256="mock-bank", candidate_prefix_sha256="mock-prefix")]*200,
    )


def test_fixed_matrix_and_budget_and_random_once_per_seed():
    cases = runner.experiment_cases()
    assert len(cases) == 39
    assert [c["env_seed"] for c in cases if c["mode"] == "random"] == [100, 101, 102]
    assert sum(c["mode"] == "oracle" for c in cases) == 18
    for i, case in enumerate(cases):
        if case["mode"] == "oracle":
            assert cases[i-1] == dict(case, mode="learned")
    assert runner.grid_budget() == dict(episodes=39, real_steps_max=7800,
                                       oracle_simulated_transitions_max=6600000)


def test_archive_completes_with_mock_episodes_without_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_episode", fake_episode)
    output = runner.run_grid(tmp_path / "MOCK_ONLY_NOT_EXPERIMENT")
    summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert summary["completed_episodes"] == 39
    assert len(list((output / "episodes").glob("*.json"))) == 39
    assert (output / "SUMMARY.md").exists() and (output / "stdout.txt").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        runner.run_grid(output)


def test_failure_preserves_finished_mock_episode_and_protocol(tmp_path, monkeypatch):
    calls = []
    def fail_second(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("intentional mock interruption")
        return fake_episode(**kwargs)
    monkeypatch.setattr(runner, "run_episode", fail_second)
    output = tmp_path / "MOCK_FAILURE_NOT_EXPERIMENT"
    with pytest.raises(RuntimeError, match="intentional mock"):
        runner.run_grid(output)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and len(status["completed_files"]) == 1
    assert (output / status["completed_files"][0]).exists()
    assert (output / "config.json").exists() and not (output / "comparison.json").exists()
