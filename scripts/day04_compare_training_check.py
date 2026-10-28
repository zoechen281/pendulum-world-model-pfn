"""A/B 单窗口续训与验证练习；不保存或覆盖权重。

当前任务：完成 BASELINE TODO 1~3，再统一检查和运行。
范围：每组在同一训练窗口更新一次，在同一验证窗口做两步递推评估。
这只是单窗口检查，不能据此得出整体泛化或多步训练普遍更好的结论。
"""

import copy
import hashlib
import json
from pathlib import Path

import torch

from src.day04_world_model import (
    DynamicsMLP,
    episode_observations,
    multistep_loss,
    normalize_inputs,
    rollout_predictions,
)


def window_loss(model, true_trajectory, actions, input_mean, input_std, mode):
    """两步窗口：(3,3) 真实轨迹、两个 (1,1) 动作；返回有梯度的标量。"""
    if mode == "A":
        # A 组：两个真实当前观测分别作为输入。
        # 1. 取当前观测 s0,s1 与目标观测 s1,s2，均为 (2,3)。
        current_observations = true_trajectory[:2]
        next_observations = true_trajectory[1:3]
        # 2. 拼接两个动作得到 action_batch (2,1)。
        action_batch = torch.cat(actions, dim=0)
        # 3. 拼接当前观测与动作得到 inputs (2,4)。
        inputs = torch.cat([current_observations, action_batch], dim=1)
        # 4. 使用传入的训练集统计量标准化 inputs。
        inputs_normalized = normalize_inputs(inputs, input_mean, input_std)
        # 5. 模型输出 predicted_deltas (2,3)，加回原尺度当前观测。
        predicted_deltas = model(inputs_normalized)
        # 6. 返回预测下一观测与真实下一观测的平均 MSE。
        return torch.nn.functional.mse_loss(
            predicted_deltas + current_observations, next_observations
        )
    elif mode == "B":
        initial_observation = true_trajectory[:1]
        trajectory = rollout_predictions(
            model, initial_observation, actions, input_mean, input_std
        )
        predicted_trajectory = torch.cat(trajectory, dim=0)
        # B 组：第二步使用第一步预测，保留跨步梯度。
        return multistep_loss(predicted_trajectory, true_trajectory)
    else:
        raise ValueError(f"未知组别：{mode}")


def main():
    # 以下是导师提供的数据恢复与对照检查辅助。
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "results/day04_delta_normalized_run.json").read_text(encoding="utf-8")
    )
    dataset_path = root / "results/day03_pendulum_dataset.json"
    assert (
        hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        == record["dataset_sha256"]
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    torch.set_num_threads(1)

    baseline = DynamicsMLP()
    baseline.load_state_dict(
        {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in record["final_state_dict"].items()
        }
    )
    models = {mode: copy.deepcopy(baseline) for mode in ("A", "B")}
    assert all(
        torch.equal(models["A"].state_dict()[name], value)
        for name, value in models["B"].state_dict().items()
    )
    optimizers = {
        mode: torch.optim.SGD(model.parameters(), lr=0.01)
        for mode, model in models.items()
    }
    input_mean = torch.tensor(record["input_mean"], dtype=torch.float32)
    input_std = torch.tensor(record["input_std"], dtype=torch.float32)
    window = dataset["splits"]["train"][0][:2]
    true_trajectory = episode_observations(window)
    actions = [
        torch.tensor(t["action"], dtype=torch.float32).reshape(1, 1) for t in window
    ]

    val_window = dataset["splits"]["val"][0][:2]
    val_true_trajectory = episode_observations(val_window)
    val_actions = [
        torch.tensor(t["action"], dtype=torch.float32).reshape(1, 1) for t in val_window
    ]

    # BASELINE TODO 1：在 A/B 更新前，评估未续训的 baseline。
    # 放在此处、for 循环外，只做一次；baseline 不交给优化器、不更新参数。
    # a. 调用 baseline.eval()。
    baseline.eval()
    # b. 在 torch.no_grad() 内调用 rollout_predictions：
    #    模型用 baseline；初始观测用 val_true_trajectory[:1]；
    #    动作用 val_actions；统计量用原有 input_mean、input_std。
    #    将列表保存为 baseline_trajectory。
    with torch.no_grad():
        baseline_trajectory = rollout_predictions(
            baseline, val_true_trajectory[:1], val_actions, input_mean, input_std
        )
    # c. 用 torch.cat 沿 dim=0 拼成 baseline_predicted_trajectory (3,3)。
    baseline_predicted_trajectory = torch.cat(baseline_trajectory, dim=0)
    # d. 用 multistep_loss 与 val_true_trajectory 比较，得到 baseline_val_loss。
    baseline_val_loss = multistep_loss(
        baseline_predicted_trajectory, val_true_trajectory
    )
    # e. 检查预测及真实轨迹形状均为 (3,3)，损失为有限标量，
    #    损失与预测均不需要梯度。可以参照下方已经完成的验证检查。
    assert baseline_predicted_trajectory.shape == val_true_trajectory.shape == (3, 3)
    assert baseline_val_loss.ndim == 0
    assert torch.isfinite(baseline_val_loss)
    assert not baseline_val_loss.requires_grad
    assert not baseline_predicted_trajectory.requires_grad
    # f. 在 no_grad 块外打印“未续训 baseline、验证首窗口、两步递推 MSE”，
    #    数值用 baseline_val_loss.item()，保留六位小数。
    print(f"未续训 baseline、验证首窗口、两步递推 MSE = {baseline_val_loss.item():.6f}")
    # 注意：baseline 指已有的标准化残差模型，并非随机初始化模型。

    for mode, model in models.items():
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        model.train()
        optimizer = optimizers[mode]
        optimizer.zero_grad()
        loss = window_loss(model, true_trajectory, actions, input_mean, input_std, mode)
        assert loss.ndim == 0 and loss.requires_grad and torch.isfinite(loss)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        optimizer.step()
        assert any(
            not torch.equal(p, before[name]) for name, p in model.named_parameters()
        )
        print(f"{mode}: 更新前训练目标 loss={loss.item():.6f}；梯度有限，完成一次更新")
        # 切换为评估模式
        model.eval()
        with torch.no_grad():
            val_trajectory = rollout_predictions(
                model, val_true_trajectory[:1], val_actions, input_mean, input_std
            )
            val_predicted_trajectory = torch.cat(val_trajectory, dim=0)

            # 已完成：计算验证两步平均误差，保存为 val_loss。
            # 调用 multistep_loss，参数依次为：
            #   val_predicted_trajectory：预测轨迹 (3,3)，含初始观测；
            #   val_true_trajectory：真实轨迹 (3,3)，含初始观测。
            # 输出应为标量 Tensor；该函数会排除初始项。
            # 两组验证都使用这里的递推结果，不按 mode 切换验证方式。
            val_loss = multistep_loss(val_predicted_trajectory, val_true_trajectory)

            # 已完成：用 assert 检查验证结果。
            # a. val_predicted_trajectory.shape 与 val_true_trajectory.shape
            #    都等于 (3,3)。
            # b. val_loss.ndim 等于 0。
            # c. torch.isfinite(val_loss) 为真。
            # d. val_loss.requires_grad 为假，验证预测也不应需要梯度。
            # 此处不调用 backward、step，也不重新拟合标准化统计量。
            assert val_predicted_trajectory.shape == val_true_trajectory.shape == (3, 3)
            assert val_loss.ndim == 0
            assert torch.isfinite(val_loss)
            assert not val_loss.requires_grad
            assert not val_predicted_trajectory.requires_grad

        # 已完成：在这里打印本组的验证结果。
        # 保持在 for 循环内，与 with 对齐；输出包括 mode、
        # “更新后、验证首窗口、两步递推 MSE”等含义及 val_loss.item()。
        # 可参考前面的 f-string，将误差保留六位小数。
        print(f"{mode}: 更新后、验证首窗口、两步递推 MSE = {val_loss.item():.6f}")

        # BASELINE TODO 2：在 for 循环内、with 块外，计算并打印差值。
        # delta_val_mse = 本组更新后验证 MSE 减去 baseline 验证 MSE。
        # 两个 Tensor 都先用 .item() 转为普通数值；结果存为 delta_val_mse。
        # 打印 mode 和差值，建议格式 {delta_val_mse:+.6f} 保留正负号。
        # 负值表示本窗口误差下降，正值表示上升；不要用训练 loss 做减法。
        delta_val_mse = val_loss.item() - baseline_val_loss.item()
        print(f"{mode}: 更新后验证 MSE 与 baseline 差值 = {delta_val_mse:+.6f}")

    # BASELINE TODO 3：更新下面已经过时的结果范围说明。
    # 已评估 baseline 后，删去“尚未评估 baseline”的旧说法，改为：
    # 仅比较一次训练窗口更新后，在一个验证窗口上的变化；
    # 不代表全部验证轨迹、多次更新或其他种子的结果。
    # 本次不扩展全部窗口，不使用测试集，不保存或覆盖历史权重。
    print(
        "注意：这里只比较一次训练窗口更新后，在一个验证窗口上的误差变化；"
        "不能代表全部验证轨迹、多次更新或其他随机种子的表现。"
    )


if __name__ == "__main__":
    main()
