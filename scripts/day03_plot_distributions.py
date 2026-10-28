"""只读取训练集，检查状态和动作的经验分布，不重新采集。"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
dataset = json.loads(
    (root / "results/day03_pendulum_dataset.json").read_text(encoding="utf-8")
)
records = [record for episode in dataset["splits"]["train"] for record in episode]
observations = np.asarray([r["observation"] for r in records], dtype=np.float32)
actions = np.asarray([r["action"] for r in records], dtype=np.float32)

# arctan2(sin(theta), cos(theta)) 将朝向恢复为 [-pi,pi] 范围的角度。
angles = np.arctan2(observations[:, 1], observations[:, 0])
angular_velocities = observations[:, 2]
torques = actions[:, 0]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
# TODO 1：axes[0].hist 画 angles，bins=20，range=(-np.pi, np.pi)。
axes[0].hist(angles, bins=20, range=(-np.pi, np.pi))
# TODO 2：axes[1].hist 画 angular_velocities，bins=20，range=(-8, 8)。
axes[1].hist(angular_velocities, bins=20, range=(-8, 8))
# TODO 3：axes[2].hist 画 torques，bins=20，range=(-2, 2)。
axes[2].hist(torques, bins=20, range=(-2, 2))

for axis, label in zip(
    axes, ["Angle (rad)", "Angular velocity (rad/s)", "Torque (N m)"]
):
    axis.set_xlabel(label)
    axis.set_ylabel("Transition count")
fig.suptitle("Training data only: 6 episodes, 1200 transitions")
fig.tight_layout()
output_path = root / "results/day03_training_distributions.png"
fig.savefig(output_path)
plt.close(fig)
print(output_path)

summary = {
    "split": "train",
    "episodes": len(dataset["splits"]["train"]),
    "transitions": len(records),
}
for name, values, bounds in [
    ("angle", angles, (-np.pi, np.pi)),
    ("angular_velocity", angular_velocities, (-8, 8)),
    ("torque", torques, (-2, 2)),
]:
    counts, edges = np.histogram(values, bins=20, range=bounds)
    summary[name] = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "bin_counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }
    print(f"{name}: min={values.min():.4f}, max={values.max():.4f}")
(root / "results/day03_training_distribution_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
