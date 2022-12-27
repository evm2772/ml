import gym
import tensorflow as tf
import wandb
from warnings import filterwarnings
# filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
import torch
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

# повторять 10 раз
for i in range(10):
    # предпринять случайное действие
    obj, reward, done, truncated, info = env.step(env.action_space.sample())

    # нарисовать состояние игры
    env.render()
    if done or truncated:
        print(obj, reward, done, truncated, info)
        break
# закрыть окружающую среду
input("Press Enter key to close...")
env.close()
