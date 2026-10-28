"""检查Day15 PFN模型结构与NLL；只用合成小数据，不训练真实任务。"""

import math

import torch
from src.day15_pfn_model import ContextPFN, gaussian_nll
from src.day15_pfn_prior import sample_tasks


def _model():
    torch.manual_seed(0)
    model = ContextPFN()
    model.eval()
    return model


def test_nll_matches_manual_formula():
    mean = torch.tensor([[[1.0]], [[2.0]]])
    log_var = torch.zeros_like(mean)
    y = torch.tensor([[[1.1]], [[2.3]]])
    expected = (
        0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2)
    ).mean()
    torch.testing.assert_close(gaussian_nll(mean, log_var, y), expected)


def test_nll_decreases_as_prediction_approaches():
    mean = torch.tensor([[[1.0]]])
    log_var = torch.zeros_like(mean)
    good = gaussian_nll(mean, log_var, torch.tensor([[[1.0]]]))
    bad = gaussian_nll(mean, log_var, torch.tensor([[[1.5]]]))
    assert good < bad


def test_forward_shapes_and_no_leakage_between_queries():
    # 预测是推理时计算；train/eval 模式下 PyTorch 注意力走不同路径，
    # 置换等变性以 eval+no_grad 语义为准。
    model = _model()
    task = sample_tasks(3, 5, 4, torch.Generator().manual_seed(1))
    with torch.no_grad():
        mean, log_var = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
        assert mean.shape == (3, 4, 1) and log_var.shape == (3, 4, 1)
        flipped = task["x_qry"].flip(dims=[1])
        mean2, _ = model(task["x_ctx"], task["y_ctx"], flipped)
    torch.testing.assert_close(mean.flip(dims=[1]), mean2, atol=1e-6, rtol=1e-5)


def test_context_set_order_invariant_for_queries():
    model = _model()
    task = sample_tasks(2, 6, 2, torch.Generator().manual_seed(2))
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        a, _ = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
        b, _ = model(task["x_ctx"][:, perm], task["y_ctx"][:, perm], task["x_qry"])
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_zero_context_uses_prior_token_and_is_finite():
    model = _model()
    task = sample_tasks(2, 0, 3, torch.Generator().manual_seed(3))
    with torch.no_grad():
        mean, log_var = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
    assert mean.shape == (2, 3, 1)
    assert torch.isfinite(mean).all() and torch.isfinite(log_var).all()


def test_nll_backward_reaches_model_parameters():
    # K=0 让 prior_token 也参与计算，否则它天然无梯度
    model = _model()
    model.train()
    task = sample_tasks(2, 0, 2, torch.Generator().manual_seed(4))
    mean, log_var = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
    loss = gaussian_nll(mean, log_var, task["y_qry"])
    loss.backward()
    grads = {p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()}
    assert all(grads)
