"""Day8受控单步训练练习；不恢复旧模型，不覆盖历史权重/数据。"""
import json
from pathlib import Path
import torch
from src.day04_world_model import DynamicsMLP, episodes_to_tensors, fit_input_mean, fit_input_std, predict_next_observation
from src.day08_physics_model import PhysicsStructuredMLP

def predict_for_comparison(model, mode, observations, actions, input_mean, input_std):
    """两组统一返回原尺度下一观测 (N,3)，调用方控制梯度。"""
    if mode == 'baseline':
        return predict_next_observation(model, observations, actions, input_mean, input_std)
    if mode == 'structured':
        return model(observations, actions, input_mean, input_std)
    raise ValueError(f'未知组别：{mode}')

def main():
    root = Path(__file__).resolve().parents[1]
    dataset = json.loads((root / 'results/day03_pendulum_dataset.json').read_text(encoding='utf-8'))
    train_inputs, train_targets = episodes_to_tensors(dataset['splits']['train'])
    val_inputs, val_targets = episodes_to_tensors(dataset['splits']['val'])
    input_mean = fit_input_mean(train_inputs)
    input_std = fit_input_std(train_inputs)
    observations = train_inputs[:, :3]
    actions = train_inputs[:, 3:4]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    baseline = DynamicsMLP()
    torch.manual_seed(0)
    structured = PhysicsStructuredMLP()
    for layer_index in (0, 2):
        for name, parameter in baseline.network[layer_index].named_parameters():
            assert torch.equal(parameter, dict(structured.network[layer_index].named_parameters())[name])
    models = {'baseline': baseline, 'structured': structured}
    epochs = 200
    learning_rate = 0.01
    archive_groups = {}
    for mode, model in models.items():
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f'{mode}: parameters={sum((p.numel() for p in model.parameters()))}')
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = predict_for_comparison(model, mode, observations, actions, input_mean, input_std)
            loss = torch.mean((predictions - train_targets) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = predict_for_comparison(model, mode, observations, actions, input_mean, input_std)
            mse = torch.mean((predictions - train_targets) ** 2).item()
            print(f'{mode}: final training MSE={mse:.6f}')
            predictions_val = predict_for_comparison(model, mode, val_inputs[:, :3], val_inputs[:, 3:4], input_mean, input_std)
            mse_val = torch.mean((predictions_val - val_targets) ** 2).item()
            print(f'{mode}: final validation MSE={mse_val:.6f}')
            component_mse = ((predictions_val - val_targets) ** 2).mean(dim=0)
            print(f'{mode}: component MSE={component_mse.tolist()}')
            circle_errors = torch.abs(predictions_val[:, 0] ** 2 + predictions_val[:, 1] ** 2 - 1)
            print(f'{mode}: mean circle error={circle_errors.mean().item():.6f}')
        assert predictions_val.shape == val_targets.shape == (400, 3)
        assert not predictions_val.requires_grad and (not model.training)
        assert torch.isfinite(predictions_val).all() and torch.isfinite(component_mse).all()
        assert all((torch.isfinite(p).all() and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        archive_groups[mode] = dict(train_mse=mse, validation_mse=mse_val, component_mse=component_mse.tolist(), mean_circle_error=circle_errors.mean().item(), max_circle_error=circle_errors.max().item(), parameters=sum((p.numel() for p in model.parameters())), final_state_dict={name: value.detach().tolist() for name, value in model.state_dict().items()})
    return dict(groups=archive_groups, input_mean=input_mean.tolist(), input_std=input_std.tolist(), train_count=len(train_inputs), val_count=len(val_inputs), epochs=epochs, learning_rate=learning_rate)
if __name__ == '__main__':
    main()