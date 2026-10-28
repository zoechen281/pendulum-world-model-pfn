"""仅用合成指标检查学习者的早停/快照语句；不前向、不反传、不更新参数。"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace


class StateOnlyModel:
    def __init__(self):
        self.state = {"tick": [0]}

    def state_dict(self):
        return self.state


def build_control_flow_check():
    source = Path(__file__).resolve().parents[1] / "scripts/day10_convergence_check.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "train_adam_with_early_stopping")
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    evaluation = next(n for n in loop.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Compare))
    start = next(i for i, n in enumerate(evaluation.body)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "append")
    names = {"history", "best_val_mse", "best_train_mse", "best_step", "best_state_dict",
             "patience_reference", "no_improvement_checks", "stop_reason"}
    initial = [n for n in fn.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    harness = ast.parse("def check(model, config, checkpoints):\n    pass").body[0]
    synthetic_loop = ast.parse(
        "for step, train_mse, val_mse in checkpoints:\n    model.state['tick'][0] = step"
    ).body[0]
    synthetic_loop.body.extend(copy.deepcopy(evaluation.body[start:]))
    harness.body = copy.deepcopy(initial) + [synthetic_loop] + copy.deepcopy(fn.body[fn.body.index(loop) + 1:])
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {"copy": copy}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["check"]


def test_plateau_respects_minimum_updates_and_keeps_independent_best_snapshot():
    check = build_control_flow_check()
    model = StateOnlyModel()
    config = SimpleNamespace(min_delta=0.01, patience_checks=2, min_updates=5)
    result = check(model, config, [(step, 0.5, 1.0) for step in range(1, 9)])
    assert result["stop_step"] == 5 and result["stop_reason"] == "early_stopping"
    assert result["best_step"] == 1 and result["best_train_mse"] == 0.5
    assert result["best_state_dict"]["tick"] == [1]
    assert result["final_state_dict"]["tick"] == [5]
    model.state["tick"][0] = 999
    assert result["best_state_dict"]["tick"] == [1]
    assert result["final_state_dict"]["tick"] == [5]


def test_cumulative_small_improvement_resets_patience_separately_from_best():
    check = build_control_flow_check()
    config = SimpleNamespace(min_delta=0.1, patience_checks=3, min_updates=1)
    values = [1.0, 0.96, 0.92, 0.88, 0.85, 0.84, 0.83, 0.82]
    result = check(StateOnlyModel(), config, [(i + 1, value / 2, value) for i, value in enumerate(values)])
    assert result["stop_step"] == 7 and result["stop_reason"] == "early_stopping"
    assert result["best_step"] == 7 and result["best_val_mse"] == 0.83
    assert len(result["history"]) == 7


def test_budget_exhaustion_returns_last_and_best_at_their_actual_steps():
    check = build_control_flow_check()
    config = SimpleNamespace(min_delta=0.01, patience_checks=10, min_updates=1)
    result = check(StateOnlyModel(), config, [(50, 0.5, 1.0), (100, 0.3, 0.8), (150, 0.4, 0.9)])
    assert result["stop_reason"] == "max_updates" and result["stop_step"] == 150
    assert result["best_step"] == 100 and result["best_val_mse"] == 0.8
    assert result["best_state_dict"]["tick"] == [100]
    assert result["final_state_dict"]["tick"] == [150]
