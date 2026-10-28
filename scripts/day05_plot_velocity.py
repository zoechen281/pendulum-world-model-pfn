"""读取验证首轨迹前20步的已保存预测，比较角速度，不重新训练。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
record = json.loads(
    (root / "results/day04_rollout_20.json").read_text(encoding="utf-8")
)
predicted_trajectory = np.asarray(record["predicted_trajectory"], dtype=float)
true_trajectory = np.asarray(record["true_trajectory"], dtype=float)
assert predicted_trajectory.shape == true_trajectory.shape == (21, 3)
times = np.arange(record["horizon"] + 1) * 0.05
output_path = root / "results/day05_velocity_comparison.png"

# TODO 1：取 predicted_trajectory 第三列，保存为 predicted_velocity。
predicted_velocity = predicted_trajectory[:, 2]
# TODO 2：取 true_trajectory 第三列，保存为 true_velocity。
true_velocity = true_trajectory[:, 2]

plt.figure()
# TODO 3：画真实曲线，label="True"。
plt.plot(times, true_velocity, label="True")
# TODO 4：画预测曲线，label="Predicted"。
plt.plot(times, predicted_velocity, label="Predicted")
# TODO 5：调用 plt.legend() 显示图例。
plt.legend()

plt.xlabel("Prediction time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Validation episode 0: angular velocity")
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()
print(output_path)
