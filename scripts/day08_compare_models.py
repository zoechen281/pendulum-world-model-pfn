"""Day8受控单步训练练习；不恢复旧模型，不覆盖历史权重/数据。"""

import json
from pathlib import Path

import torch

from src.day04_world_model import (
    DynamicsMLP,
    episodes_to_tensors,
    fit_input_mean,
    fit_input_std,
    predict_next_observation,
)
from src.day08_physics_model import PhysicsStructuredMLP


def predict_for_comparison(model, mode, observations, actions, input_mean, input_std):
    """两组统一返回原尺度下一观测 (N,3)，调用方控制梯度。"""
    if mode == "baseline":
        # 复用学习者已有的残差预测函数；它也支持当前按行独立的批量计算。
        return predict_next_observation(
            model, observations, actions, input_mean, input_std
        )
    if mode == "structured":
        # 当前任务：调用结构模型，并return其最终下一观测预测。
        # 参数依次为 observations、actions、input_mean、input_std。
        return model(observations, actions, input_mean, input_std)
    raise ValueError(f"未知组别：{mode}")


def main():
    # 导师提供的工程准备；不读取test集合做评估，不更新验证数据上的参数。
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads(
        (root / "results/day03_pendulum_dataset.json").read_text(encoding="utf-8")
    )
    train_inputs, train_targets = episodes_to_tensors(dataset["splits"]["train"])
    val_inputs, val_targets = episodes_to_tensors(dataset["splits"]["val"])
    input_mean = fit_input_mean(train_inputs)
    input_std = fit_input_std(train_inputs)
    observations = train_inputs[:, :3]
    actions = train_inputs[:, 3:4]

    torch.set_num_threads(1)
    torch.manual_seed(0)
    baseline = DynamicsMLP()
    torch.manual_seed(0)
    structured = PhysicsStructuredMLP()
    # 同架构隐藏层共享相同初值；输出头形状不同，不能逐参数全相同。
    for layer_index in (0, 2):
        for name, parameter in baseline.network[layer_index].named_parameters():
            assert torch.equal(
                parameter,
                dict(structured.network[layer_index].named_parameters())[name],
            )
    models = {"baseline": baseline, "structured": structured}
    epochs = 200
    learning_rate = 0.01
    for mode, model in models.items():
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f"{mode}: parameters={sum(p.numel() for p in model.parameters())}")
        model.train()
        for epoch in range(epochs):
            # 后续训练任务：清梯度；调用predict_for_comparison得到predictions；
            # 和train_targets计算最终观测MSE；backward；step。
            # 两组均200次全批量更新；不把更新前loss误称最终指标。
            optimizer.zero_grad()
            predictions = predict_for_comparison(
                model, mode, observations, actions, input_mean, input_std
            )
            loss = torch.mean((predictions - train_targets) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = predict_for_comparison(
                model, mode, observations, actions, input_mean, input_std
            )
            mse = torch.mean((predictions - train_targets) ** 2).item()
            print(f"{mode}: final training MSE={mse:.6f}")
            predictions_val = predict_for_comparison(
                model,
                mode,
                val_inputs[:, :3],
                val_inputs[:, 3:4],
                input_mean,
                input_std,
            )
            mse_val = torch.mean((predictions_val - val_targets) ** 2).item()
            print(f"{mode}: final validation MSE={mse_val:.6f}")
            component_mse = ((predictions_val - val_targets) ** 2).mean(dim=0)
            print(f"{mode}: component MSE={component_mse.tolist()}")
            circle_errors = torch.abs(
                predictions_val[:, 0] ** 2 + predictions_val[:, 1] ** 2 - 1
            )
            print(f"{mode}: mean circle error={circle_errors.mean().item():.6f}")
        # 后续评估任务：eval + no_grad；重新计算最终训练/验证下一观测MSE，
        # 及验证各分量MSE、单位圆误差；两组使用相同统计量和数据。
        # 单训练种子、小预算，不以必胜为验收标准。多步/控制不在本次必做比较内。


if __name__ == "__main__":
    main()
