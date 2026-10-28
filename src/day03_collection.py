"""Day 3：采集真实环境转移；核心循环由学习者填写。"""

import gymnasium as gym


def collect_episode(env: gym.Env, seed: int) -> list[dict]:
    """从 reset 开始，用随机动作采集至 terminated 或 truncated。

    env 约定为 gym.make 创建且含时间限制的 Pendulum-v1。
    返回列表，每项包含 observation、action、reward、next_observation、
    terminated、truncated 六个字段。数组字段用 copy 保存快照。
    环境由调用方创建和关闭，本函数不关闭环境。
    本阶段只采集一个 episode，不保存文件、不划分数据。
    """
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed)
    transitions = []

    while True:
        # TODO 1：用 env.action_space.sample() 生成 action。
        action = env.action_space.sample()
        # TODO 2：用 env.step(action) 取得五个返回值。
        next_observation, reward, terminated, truncated, info = env.step(action)
        # TODO 3：将六字段字典 append 到 transitions，数组字段保存 copy。
        transitions.append(
            {
                "observation": observation.copy(),
                "action": action.copy(),
                "reward": reward,
                "next_observation": next_observation.copy(),
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        # TODO 4：把 observation 更新为 next_observation。
        observation = next_observation
        # TODO 5：若 terminated 或 truncated，则 break。
        if terminated or truncated:
            break

    return transitions


def collect_episodes(
    env: gym.Env,
    num_episodes: int,
    base_seed: int = 0,
) -> list[list[dict]]:
    """采集多个 episode，保持每条轨迹独立，不打散或展平。

    当前约定 num_episodes 为正整数，base_seed 为非负整数。
    第 episode_id 个 episode 使用 base_seed + episode_id 作为种子。
    返回 episodes：外层每项是一条轨迹，内层每项是一条转移。
    环境由调用方创建和关闭；本函数不保存文件。
    """
    # TODO 1：创建空列表 episodes。
    episodes = []
    # TODO 2：遍历 range(num_episodes)，循环变量命名 episode_id。
    for episode_id in range(num_episodes):
        # TODO 3：计算 episode_seed，调用 collect_episode 得到 transitions。
        episode_seed = base_seed + episode_id
        transitions = collect_episode(env, seed=episode_seed)
        # TODO 4：将整条 transitions 作为一项 append 到 episodes。
        episodes.append(transitions)
    # TODO 5：循环结束后返回 episodes。
    return episodes


def split_episodes(
    episodes: list[list[dict]],
    train_count: int,
    val_count: int,
) -> dict[str, list[list[dict]]]:
    """按完整 episode 顺序划分，剩余 episode 全部作为测试集。

    当前约定 train_count、val_count 为正整数，且二者之和小于
    len(episodes)，保证训练、验证、测试三部分都非空。
    返回键为 train、val、test，不修改输入、不打散每条轨迹。
    本阶段采用可复现的顺序划分，不要求随机排列轨迹。
    """
    # TODO 1：用切片取前 train_count 条轨迹，保存为 train_episodes。
    train_episodes = episodes[:train_count]
    # TODO 2：取接下来 val_count 条轨迹，保存为 val_episodes。
    val_episodes = episodes[train_count : train_count + val_count]
    # TODO 3：取剩余轨迹，保存为 test_episodes。
    test_episodes = episodes[train_count + val_count :]
    # TODO 4：将三部分放入 train、val、test 三键字典并返回。
    return {
        "train": train_episodes,
        "val": val_episodes,
        "test": test_episodes,
    }
