"""检查Day15任务采样；不执行训练或读取真实数据。"""

import math

import torch
from src.day15_pfn_prior import pendulum_delta_omega, sample_tasks

KEYS = {"x_ctx", "y_ctx", "x_qry", "y_qry", "params", "obs_qry"}


def _sample(seed=0, B=4, K=3, Q=5, **kwargs):
    return sample_tasks(B, K, Q, torch.Generator().manual_seed(seed), **kwargs)


def test_keys_shapes_dtypes():
    out = _sample()
    assert set(out) == KEYS
    expected = {
        "x_ctx": (4, 3, 2), "y_ctx": (4, 3, 1), "x_qry": (4, 5, 2),
        "y_qry": (4, 5, 1), "params": (4, 2), "obs_qry": (4, 5, 3),
    }
    for key, shape in expected.items():
        assert out[key].shape == shape, key
        assert out[key].dtype == torch.float32, key


def test_zero_context_allowed():
    out = _sample(K=0)
    assert out["x_ctx"].shape == (4, 0, 2)
    assert out["y_ctx"].shape == (4, 0, 1)


def test_same_seed_reproducible_and_different_seed_differs():
    a, b, c = _sample(seed=1), _sample(seed=1), _sample(seed=2)
    for key in KEYS:
        assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a["params"], c["params"])


def test_noise_free_labels_follow_shared_task_params():
    out = _sample(noise_std=0.0)
    a = out["params"][:, 0].reshape(-1, 1, 1)
    b = out["params"][:, 1].reshape(-1, 1, 1)
    for x, y in ((out["x_ctx"], out["y_ctx"]), (out["x_qry"], out["y_qry"])):
        expected = pendulum_delta_omega(x[..., :1], x[..., 1:], a, b)
        torch.testing.assert_close(y, expected)


def test_query_observation_consistent_with_inputs_and_ranges():
    out = _sample(B=64, K=8, Q=16)
    obs, x = out["obs_qry"], out["x_qry"]
    torch.testing.assert_close(obs[..., 1], x[..., 0])
    torch.testing.assert_close(obs[..., 0] ** 2 + obs[..., 1] ** 2, torch.ones_like(obs[..., 0]))
    assert obs[..., 2].abs().max() <= 6.0
    assert out["x_ctx"][..., 1].abs().max() <= 2.0 and x[..., 1].abs().max() <= 2.0


def test_noise_level_and_prior_moments_roughly_match():
    out = _sample(B=20000, K=1, Q=1, noise_std=0.02)
    params = out["params"]
    torch.testing.assert_close(params.mean(0), torch.tensor([15.0, 3.0]), atol=0.1, rtol=0)
    torch.testing.assert_close(params.std(0), torch.tensor([3.0, 0.75]), atol=0.1, rtol=0)
    a = params[:, 0].reshape(-1, 1, 1)
    b = params[:, 1].reshape(-1, 1, 1)
    x = out["x_qry"]
    residual = out["y_qry"] - pendulum_delta_omega(x[..., :1], x[..., 1:], a, b)
    assert math.isclose(residual.std().item(), 0.02, rel_tol=0.05)
