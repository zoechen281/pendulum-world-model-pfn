"""读取已有验证窗口误差记录绘图，不重新训练或评估。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
record = json.loads(
    (root / "results/day04_rollout_windows.json").read_text(encoding="utf-8")
)
mean_step_mse = np.asarray(record["mean_step_mse"], dtype=float)
# Pendulum-v1 每步0.05秒；记录包含初始时刻。
times = np.arange(record["horizon"] + 1) * 0.05
assert times.shape == mean_step_mse.shape
assert np.isfinite(mean_step_mse).all()
output_path = root / "results/day05_rollout_mean_error.png"

plt.figure()
# 在这里填写刚才练习的四行：plot、xlabel、ylabel、title。
plt.plot(times, mean_step_mse)
plt.xlabel("Prediction time (s)")
plt.ylabel("Mean rollout MSE")
plt.title("Validation rollout error")
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()
print(output_path)
