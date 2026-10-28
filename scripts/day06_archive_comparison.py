"""导师提供的归档辅助：复用学习者闭环代码，保存六组日志、源码和精确结果。"""

import ast
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import gymnasium
import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    shanghai = timezone(timedelta(hours=8), name="Asia/Shanghai")
    stamp = datetime.now(shanghai).strftime("%Y%m%dT%H%M%S_%f")
    output = root / "results" / f"day06_horizon_comparison_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    runner = root / "scripts/day06_run_mpc_check.py"
    checkpoint = root / "results/day04_delta_normalized_run.json"
    paths = [runner, root / "src/day06_mpc.py", root / "src/day04_world_model.py", Path(__file__)]
    hashes = {str(p.relative_to(root)): sha256(p) for p in paths}
    for path in paths:
        (output / path.name).write_bytes(path.read_bytes())
    checkpoint_before = sha256(checkpoint)
    source = runner.read_text(encoding="utf-8")
    runs = []
    for horizon in (3, 5):
        # 仅在内存中更改规划长度，并追加返回统计量；不修改原脚本和核心算法。
        tree = ast.parse(source)
        main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        assignments = [n for n in ast.walk(main_node) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "horizon" for t in n.targets)]
        assert len(assignments) == 1
        assignments[0].value = ast.Constant(horizon)
        main_node.body.append(ast.parse(
            "return dict(mpc_return=float(total_reward), random_return=float(random_total_reward), "
            "mpc_steps=int(steps_executed))"
        ).body[0])
        executed_source = ast.unparse(ast.fix_missing_locations(tree))
        executed_path = output / f"executed_runner_h{horizon}.py"
        executed_path.write_text(executed_source, encoding="utf-8")
        namespace = {"__name__": "archived_day06_runner", "__file__": str(runner)}
        exec(compile(executed_source, str(runner), "exec"), namespace)
        for seed in (100, 101, 102):
            capture = io.StringIO()
            started = time.perf_counter()
            with contextlib.redirect_stdout(capture):
                result = namespace["main"](env_seed=seed)
            elapsed = time.perf_counter() - started
            log = capture.getvalue()
            assert log.count("第 200 步：") == 2
            assert result["mpc_steps"] == 200
            assert np.isfinite([result["mpc_return"], result["random_return"]]).all()
            log_name = f"h{horizon}_env{seed}_stdout.txt"
            (output / log_name).write_text(log, encoding="utf-8")
            result.update(horizon=horizon, env_seed=seed,
                          delta_mpc_minus_random=result["mpc_return"] - result["random_return"],
                          random_steps=200, elapsed_seconds=elapsed, stdout=log_name,
                          executed_source_sha256=sha256(executed_path))
            runs.append(result)
            print(f"H={horizon}, seed={seed}: MPC={result['mpc_return']:.6f}, "
                  f"random={result['random_return']:.6f}, delta={result['delta_mpc_minus_random']:+.6f}")
    assert sha256(checkpoint) == checkpoint_before
    assert all(sha256(root / name) == value for name, value in hashes.items())
    report = dict(
        recorded_at=datetime.now(shanghai).isoformat(),
        source_sha256=hashes, checkpoint_sha256=checkpoint_before,
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      numpy=np.__version__, gymnasium=gymnasium.__version__),
        protocol=dict(env="Pendulum-v1", env_seeds=[100, 101, 102], horizons=[3, 5],
                      num_candidates=20, max_steps=200, planner_sampling_seed=0,
                      random_action_seed=0, torch_threads=1, device="cpu",
                      reward="sum of actual environment rewards", discount=1.0),
        execution="Copied runner AST; changed horizon constant and appended statistics return only; original __file__ retained for checkpoint lookup.",
        checkpoint_unchanged=True, training_performed=False, test_dataset_evaluated=False,
        limitations=["Three development environment seeds only; one action sampling seed.",
                     "Different horizons change sampled sequences and computation budget.",
                     "Wall times include both policies and logging; not an isolated latency benchmark."],
        runs=runs,
    )
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ARCHIVE: {output}")


if __name__ == "__main__":
    main()
