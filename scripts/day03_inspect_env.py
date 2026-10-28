import gymnasium as gym
import numpy as np

env = gym.make("Pendulum-v1")
observation, info = env.reset(seed=0)

print(observation)
print(observation.shape)
action = np.array([1.0], dtype=np.float32)
next_observation, reward, terminated, truncated, info = env.step(action)
print(action)
print(next_observation)
print(reward)
env.close()
