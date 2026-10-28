"""Day 2：固定模拟总时长，比较 Euler 的最终位置误差。"""

from src.day02_dynamics import exact_constant_force, simulate_euler

initial_position = 0.0
initial_velocity = 1.0
force = 4.0
mass = 2.0
total_time = 2.0

exact_position, exact_velocity = exact_constant_force(
    initial_position, initial_velocity, force, mass, total_time
)

for dt in [0.5, 0.25, 0.125]:
    # 本例的步长均能整除总时长，不要求处理不整除情况。
    # TODO 1：计算 steps（需要整数）。
    steps = int(total_time / dt)
    # TODO 2：调用 simulate_euler，将轨迹保存为 trajectory。
    trajectory = simulate_euler(
        initial_position, initial_velocity, force, mass, dt, steps
    )
    # TODO 3：从 trajectory 最后一项取出 final_position、final_velocity。
    final_position, final_velocity = trajectory[-1]
    # TODO 4：用 abs 计算位置绝对误差 position_error。
    position_error = abs(final_position - exact_position)
    print(f"dt={dt}, steps={steps}, x={final_position}, error={position_error}")
