"""Day 1：读取真实训练记录，由学习者完成损失曲线绘制。"""

import json
from pathlib import Path

import matplotlib

# 直接保存图片，不启动交互窗口。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 文件定位和读取属于工程准备，当前练习只填写绘图部分。
project_dir = Path(__file__).resolve().parents[1]
record_path = project_dir / "results" / "day01_mlp_lr001_run.json"
record = json.loads(record_path.read_text(encoding="utf-8"))
losses = record["losses_before_each_update"]
output_path = project_dir / "results" / "day01_mlp_lr001_loss.png"

# 请亲手完成：
# 1. 用 plt.plot 绘制 losses；默认横坐标依次为 0、1、...、199。
# 2. 用 plt.xlabel 设置横轴标签 "Update"。
# 3. 用 plt.ylabel 设置纵轴标签 "Training MSE"。
# 4. 用 plt.savefig 保存到 output_path。
# 5. 用 plt.close 关闭当前图。
# 完成后删除占位异常。
plt.plot(losses)
plt.xlabel("Update")
plt.ylabel("Training MSE")
plt.savefig(output_path)
plt.close()
