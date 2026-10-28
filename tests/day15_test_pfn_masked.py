"""掩码语义和真正的查询独立性检查；合成小数据，不更新参数。"""

import pytest
import torch

from src.day15_pfn_model import gaussian_nll
from src.day15_pfn_model_masked import MaskedContextPFN, build_attention_mask
from src.day15_pfn_prior import sample_tasks


def make_model():
    torch.manual_seed(0)
    return MaskedContextPFN().eval()


def test_mask_matches_two_context_two_query_example():
    mask = build_attention_mask(2, 2)
    expected = torch.tensor([
        [False, False, True, True],
        [False, False, True, True],
        [False, False, True, True],
        [False, False, True, True],
    ])
    assert mask.dtype == torch.bool
    assert mask.shape == (4, 4)
    assert torch.equal(mask, expected)


def test_prior_token_mask_has_one_visible_column():
    mask = build_attention_mask(1, 3)
    assert mask.shape == (4, 4)
    assert not mask[:, 0].any()
    assert mask[:, 1:].all()


@pytest.mark.parametrize("k", [0, 2])
def test_changing_another_query_does_not_change_first_prediction(k):
    model = make_model()
    task = sample_tasks(2, k, 2, torch.Generator().manual_seed(10))
    changed = task["x_qry"].clone()
    changed[:, 1, :] = torch.tensor([0.95, -1.9])
    with torch.no_grad():
        before = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
        after = model(task["x_ctx"], task["y_ctx"], changed)
    for a, b in zip(before, after):
        assert a.shape == (2, 2, 1)
        torch.testing.assert_close(a[:, :1], b[:, :1], atol=1e-6, rtol=0)


def test_context_permutation_does_not_change_predictions():
    model = make_model()
    task = sample_tasks(2, 4, 3, torch.Generator().manual_seed(11))
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        before = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
        after = model(task["x_ctx"][:, permutation], task["y_ctx"][:, permutation], task["x_qry"])
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_no_context_has_finite_outputs():
    model = make_model()
    task = sample_tasks(2, 0, 3, torch.Generator().manual_seed(12))
    with torch.no_grad():
        outputs = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
    for output in outputs:
        assert output.shape == (2, 3, 1)
        assert torch.isfinite(output).all()


def test_backward_reaches_every_parameter_without_optimizer_step():
    model = make_model().train()
    task = sample_tasks(2, 0, 2, torch.Generator().manual_seed(13))
    mean, log_var = model(task["x_ctx"], task["y_ctx"], task["x_qry"])
    gaussian_nll(mean, log_var, task["y_qry"]).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
