"""单条监督学习数据转换的顺序、目标和独立存储检查。"""

from copy import deepcopy

import torch

from src.day04_world_model import transition_to_tensors


def test_feature_order_and_direct_target():
    record = {
        'observation': [0.6, 0.8, 2.0], 'action': [-0.5],
        'next_observation': [0.0, 1.0, 2.5],
        'reward': -100.0, 'terminated': False, 'truncated': True,
    }
    original = deepcopy(record)
    inputs, targets = transition_to_tensors(record)
    assert inputs.shape == (4,)
    assert targets.shape == (3,)
    assert inputs.dtype == targets.dtype == torch.float32
    torch.testing.assert_close(inputs, torch.tensor([0.6, 0.8, 2.0, -0.5]))
    torch.testing.assert_close(targets, torch.tensor([0.0, 1.0, 2.5]))
    assert record == original


def test_tensor_mutation_does_not_modify_json_lists():
    record = {'observation': [1., 0., 0.], 'action': [2.], 'next_observation': [0.8, 0.6, 1.]}
    original = deepcopy(record)
    inputs, targets = transition_to_tensors(record)
    inputs.fill_(99.)
    targets.fill_(99.)
    assert record == original
