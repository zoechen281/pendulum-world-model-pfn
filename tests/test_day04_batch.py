"""按轨迹与时间顺序生成成对监督数据。"""

from copy import deepcopy

import torch

from src.day04_world_model import episodes_to_tensors


def record(value):
    return {'observation': [1., 0., value], 'action': [-value], 'next_observation': [0., 1., value+1]}


def test_order_and_row_pairing_across_unequal_episodes():
    episodes = [[record(1.), record(2.)], [record(3.)]]
    original = deepcopy(episodes)
    inputs, targets = episodes_to_tensors(episodes)
    assert inputs.shape == (3, 4)
    assert targets.shape == (3, 3)
    assert inputs.dtype == targets.dtype == torch.float32
    torch.testing.assert_close(inputs, torch.tensor([[1.,0.,1.,-1.], [1.,0.,2.,-2.], [1.,0.,3.,-3.]]))
    torch.testing.assert_close(targets, torch.tensor([[0.,1.,2.], [0.,1.,3.], [0.,1.,4.]]))
    assert episodes == original


def test_single_transition_keeps_batch_axis():
    inputs, targets = episodes_to_tensors([[record(0.)]])
    assert inputs.shape == (1, 4)
    assert targets.shape == (1, 3)
