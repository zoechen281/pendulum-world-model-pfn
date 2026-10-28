"""闭环与成功判据的有限测试；每种模式仅3步，不训练、不跑网格。"""

from dataclasses import replace

import numpy as np

from scripts import day10_oracle_mpc as runner


def test_reaching_upright_is_distinct_from_holding_and_final_stability():
    config = runner.OracleConfig()
    fast = runner.summarize_real_states([[0, 2]] * 25, config)
    assert fast["reached_upright"] and not fast["success"]
    assert fast["longest_hold_steps"] == 0
    brief = runner.summarize_real_states([[0, 0]] * 19, config)
    assert not brief["success"] and not brief["final_hold_window_stable"]
    held_then_left = runner.summarize_real_states([[0, 0]] * 20 + [[1, 0]], config)
    assert held_then_left["success"] and held_then_left["longest_hold_steps"] == 20
    assert not held_then_left["final_hold_window_stable"]
    held_at_end = runner.summarize_real_states([[1, 0]] + [[0, 0]] * 20, config)
    assert held_at_end["success"] and held_at_end["final_hold_window_stable"]
    boundary_speed = runner.summarize_real_states([[0, 1]] * 20, config)
    assert not boundary_speed["success"]


def test_three_step_closed_loop_replans_from_real_feedback(monkeypatch):
    backend = runner.load_learned_backend()
    config = replace(runner.OracleConfig(), max_real_steps=3)
    original = runner.plan_from_bank
    seen = []

    def observe_plan(mode, observation, real, *args):
        # 传入下一次规划的是环境当前真实反馈，而非模型预测状态。
        np.testing.assert_array_equal(observation, real.unwrapped._get_obs())
        seen.append((mode, real._elapsed_steps))
        return original(mode, observation, real, *args)

    monkeypatch.setattr(runner, "plan_from_bank", observe_plan)
    reports = {mode: runner.run_episode(mode, 100, 2, 2, backend, config)
               for mode in ("learned", "oracle", "random")}
    assert seen == [("learned", i) for i in range(3)] + [("oracle", i) for i in range(3)]
    for mode, report in reports.items():
        assert report["real_steps"] == 3 and len(report["observations"]) == 4
        assert len(report["rewards"]) == len(report["actions"]) == 3
        assert report["real_total_reward"] == sum(report["rewards"])
        assert report["model_parameters_unchanged"] and not report["training_performed"]
        assert report["longest_hold_steps"] <= 3 and not report["success"]
    for key in ("bank_sha256", "candidate_prefix_sha256"):
        assert [p[key] for p in reports["learned"]["plans"]] == [p[key] for p in reports["oracle"]["plans"]]
    assert reports["oracle"]["oracle_simulated_transitions"] == 12
    assert reports["random"]["candidate_transitions"] == 0
    repeated = runner.run_episode("random", 100, 2, 2, backend, config)
    assert repeated["actions"] == reports["random"]["actions"]
    assert repeated["rewards"] == reports["random"]["rewards"]
