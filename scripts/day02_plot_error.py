"""读取已保存的步长实验结果，绘制最终位置误差，不重新模拟。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
record = json.loads(
    (project_root / "results/day02_compare_steps_run.json").read_text(encoding="utf-8")
)
rows = sorted(record["rows"], key=lambda row: row["dt"])
step_sizes = [row["dt"] for row in rows]
position_errors = [row["position_error"] for row in rows]
output_path = project_root / "results/day02_euler_position_error.png"

plt.figure()
# TODO 1：以 step_sizes 为横坐标、position_errors 为纵坐标画折线。
plt.plot(step_sizes, position_errors, marker="o")
# TODO 2：横轴标签为 "Step size (s)"。
plt.xlabel("Step size (s)")
# TODO 3：纵轴标签为 "Final position absolute error (m)"。
plt.ylabel("Final position absolute error (m)")

plt.tight_layout()
plt.savefig(output_path)
plt.close()
print(output_path)
