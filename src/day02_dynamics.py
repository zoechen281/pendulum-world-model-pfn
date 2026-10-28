"""Day 2：理想一维质点的动力学，每次只实现一个小功能。"""


def euler_step(
    position: float,
    velocity: float,
    force: float,
    mass: float,
    dt: float,
) -> tuple[float, float]:
    """用显式 Euler 方法推进一步，返回 (next_position, next_velocity)。

    单位依次为 m、m/s、N、kg、s。force 是本步保持不变的合力。
    当前约定输入均有限、mass > 0、dt > 0，暂不扩展异常处理。
    位置更新使用传入的旧 velocity；不包含摩擦或其他额外力。
    示例：输入 (0.0, 1.0, 4.0, 2.0, 0.5)，应返回 (0.5, 2.0)。
    """
    # TODO 1：由合力和质量计算 acceleration。
    acceleration = force / mass
    # TODO 2：用旧速度计算 next_position。
    next_position = position + velocity * dt
    # TODO 3：计算 next_velocity。
    next_velocity = velocity + acceleration * dt
    # TODO 4：按 (位置, 速度) 顺序返回两个结果。
    return next_position, next_velocity


def simulate_euler(
    position: float,
    velocity: float,
    force: float,
    mass: float,
    dt: float,
    steps: int,
) -> list[tuple[float, float]]:
    """在恒定合力下连续推进 steps 步，返回包含初始状态的轨迹。

    沿用 euler_step 的输入约定，steps 为非负整数。
    每项为 (位置, 速度)，列表长度为 steps + 1。
    steps=0 时只返回初始状态。不重置每一步的位置和速度。
    示例：(0.0, 1.0, 4.0, 2.0, 0.5, 2) 应返回
    [(0.0, 1.0), (0.5, 2.0), (1.5, 3.0)]。
    """
    # TODO 1：创建 trajectory 列表，先放入初始 (position, velocity)。
    trajectory = [(position, velocity)]
    # TODO 2：循环 steps 次，每次调用 euler_step，更新 position、velocity。
    for _ in range(steps):
        position, velocity = euler_step(position, velocity, force, mass, dt)
        trajectory.append((position, velocity))
    # TODO 3：每次更新后，将新状态元组 append 到 trajectory。
    # TODO 4：在循环结束后返回 trajectory。
    return trajectory


def exact_constant_force(
    position: float,
    velocity: float,
    force: float,
    mass: float,
    time: float,
) -> tuple[float, float]:
    """返回恒定合力下经过 time 秒的解析状态 (位置, 速度)。

    position、velocity 为初始状态，不是逐步更新的中间状态。
    输入均有限，mass > 0，time >= 0；力在整个时间区间恒定。
    不调用 Euler 方法，不使用循环，不包含其他额外力。
    示例：(0.0, 1.0, 4.0, 2.0, 2.0) 应返回 (6.0, 5.0)。
    """
    # TODO 1：计算 acceleration。
    acceleration = force / mass
    # TODO 2：使用恒加速度解析公式计算 final_position。
    final_position = position + velocity * time + 0.5 * acceleration * time**2
    # TODO 3：计算 final_velocity。
    final_velocity = velocity + acceleration * time
    # TODO 4：按 (位置, 速度) 顺序返回结果。
    return final_position, final_velocity
