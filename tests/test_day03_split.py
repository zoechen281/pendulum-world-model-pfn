"""完整轨迹切分必须不重不漏，且不修改输入。"""

import pytest

from src.day03_collection import split_episodes


@pytest.mark.parametrize('size,train_count,val_count', [(10, 6, 2), (3, 1, 1), (7, 2, 3)])
def test_whole_episode_partition(size, train_count, val_count):
    episodes = [[{'episode_id': i, 'step': j} for j in range(i + 1)] for i in range(size)]
    original_ids = [id(episode) for episode in episodes]
    result = split_episodes(episodes, train_count, val_count)
    assert set(result) == {'train', 'val', 'test'}
    assert [len(result[k]) for k in ('train', 'val', 'test')] == [train_count, val_count, size-train_count-val_count]
    flattened = result['train'] + result['val'] + result['test']
    assert [id(episode) for episode in flattened] == original_ids
    assert [id(episode) for episode in episodes] == original_ids
    groups = [{id(episode) for episode in result[key]} for key in ('train', 'val', 'test')]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
