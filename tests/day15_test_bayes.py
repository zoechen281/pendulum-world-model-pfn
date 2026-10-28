"""检查Day15贝叶斯后验预测；只用合成数据，不训练。"""

import torch
from src.day15_bayes_oracle import posterior_predictive
from src.day15_pfn_prior import pendulum_delta_omega, sample_tasks

DT, SIGMA = 0.05, 0.02


def _reference(x_ctx, y_ctx, x_qry):
    """逐任务、逐点累加的独立写法（float64），用于对照。"""
    x_ctx, y_ctx, x_qry = x_ctx.double(), y_ctx.double(), x_qry.double()
    means, variances = [], []
    for i in range(x_ctx.shape[0]):
        precision = torch.diag(torch.tensor([1 / 9.0, 1 / 0.5625], dtype=torch.float64))
        rhs = precision @ torch.tensor([15.0, 3.0], dtype=torch.float64)
        for k in range(x_ctx.shape[1]):
            phi = DT * x_ctx[i, k]
            precision = precision + torch.outer(phi, phi) / SIGMA**2
            rhs = rhs + phi * y_ctx[i, k, 0] / SIGMA**2
        cov = torch.linalg.inv(precision)
        mu = cov @ rhs
        phi_q = DT * x_qry[i]
        means.append((phi_q @ mu).unsqueeze(-1))
        variances.append(((phi_q @ cov) * phi_q).sum(-1, keepdim=True) + SIGMA**2)
    return torch.stack(means), torch.stack(variances)


def test_shapes_positive_variance_and_inputs_unchanged():
    task = sample_tasks(3, 4, 5, torch.Generator().manual_seed(0))
    inputs = (task["x_ctx"], task["y_ctx"], task["x_qry"])
    originals = tuple(t.clone() for t in inputs)
    mean, var = posterior_predictive(*inputs)
    assert mean.shape == (3, 5, 1) and var.shape == (3, 5, 1)
    assert (var > 0).all()
    for current, original in zip(inputs, originals):
        assert torch.equal(current, original)


def test_zero_context_equals_prior_predictive():
    task = sample_tasks(2, 0, 6, torch.Generator().manual_seed(1))
    mean, var = posterior_predictive(task["x_ctx"], task["y_ctx"], task["x_qry"])
    s, u = task["x_qry"][..., :1], task["x_qry"][..., 1:]
    torch.testing.assert_close(mean, DT * (15.0 * s + 3.0 * u))
    torch.testing.assert_close(var, DT**2 * (9.0 * s**2 + 0.5625 * u**2) + SIGMA**2)


def test_matches_independent_reference():
    task = sample_tasks(4, 7, 3, torch.Generator().manual_seed(2))
    mean, var = posterior_predictive(task["x_ctx"], task["y_ctx"], task["x_qry"])
    ref_mean, ref_var = _reference(task["x_ctx"], task["y_ctx"], task["x_qry"])
    torch.testing.assert_close(mean.double(), ref_mean, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(var.double(), ref_var, atol=1e-7, rtol=1e-4)


def test_context_order_does_not_matter():
    task = sample_tasks(2, 6, 3, torch.Generator().manual_seed(3))
    perm = torch.tensor([5, 2, 0, 4, 1, 3])
    a = posterior_predictive(task["x_ctx"], task["y_ctx"], task["x_qry"])
    b = posterior_predictive(task["x_ctx"][:, perm], task["y_ctx"][:, perm], task["x_qry"])
    torch.testing.assert_close(a[0], b[0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(a[1], b[1], atol=1e-7, rtol=1e-4)


def test_many_context_points_concentrate_on_truth():
    task = sample_tasks(5, 400, 10, torch.Generator().manual_seed(4))
    mean, var = posterior_predictive(task["x_ctx"], task["y_ctx"], task["x_qry"])
    a = task["params"][:, 0].reshape(-1, 1, 1)
    b = task["params"][:, 1].reshape(-1, 1, 1)
    x = task["x_qry"]
    truth = pendulum_delta_omega(x[..., :1], x[..., 1:], a, b)
    assert (mean - truth).abs().max() < 0.02
    assert (var - SIGMA**2).max() < 0.1 * SIGMA**2
