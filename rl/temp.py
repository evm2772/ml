# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])
# print (model.f)

from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # установить режим обучения

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        # print(type(state[0]))
        x = torch.from_numpy(state.astype(np.float32))  # преобразование в тензор
        pdparam = self.forward(x)  # прямой проход
        pd = Categorical(logits=pdparam)  # вероятностное распределение
        action = pd.sample()  # pi(a|s) выбор действия по распределению pd
        log_prob = pd.log_prob(action)  # логарифм вероятности pi(a|s)
        self.log_probs.append(log_prob)  # сохраняем для обучения
        return action.item()


def train(pi, optimizer):
    # Внутренний цикл градиентного восхождения в алгоритме REINFORCE
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)  # отдачи
    future_ret = 0.0
    # эффективное вычисление отдачи
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets  # член градиента; знак минуса для максимизации
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()  # обратное распространение, вычисление градиентов
    optimizer.step()  # градиентное восхождение, обновление весов
    return loss


def main():
    #env = gym.make("CartPole-v1", render_mode="human")
    max_steps = 300
    env = gym.make("CartPole-v1",max_episode_steps=max_steps)

    in_dim = env.observation_space.shape[0]  # 4
    out_dim = env.action_space.n  # 2
    print(out_dim)
    pi = Pi(in_dim, out_dim)  # стратегия pi_theta для REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(5000):
        state, _ = env.reset()
        for t in range(max_steps):  # 200 — максимальное количество шагов в cartpole
            action = pi.act(state)
            state, reward, done, truncated, _ = env.step(action)
            pi.rewards.append(reward)
            #env.render()
            if done:
                break
        loss = train(pi, optimizer)  # обучение в эпизоде
        total_reward = sum(pi.rewards)
        solved = total_reward > (max_steps - 5)
        pi.onpolicy_reset()  # обучение по актуальному опыту: очистить память после обучения
        print(f'Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()
