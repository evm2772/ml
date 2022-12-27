from torch.distributions import Categorical
import torchvision.models as models
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
gamma = 0.99
USE_WANDB = False

if USE_WANDB:
    import wandb
    wandb.login()
    wandb.init(
            # Set the project where this run will be logged
            project="basic-intro",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_1",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.02,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 10,
            })

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
        #print(pdparam)
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
    LOAD = False
    env = gym.make("CartPole-v1")
    #env = gym.make("CartPole-v1", render_mode="human")
    in_dim = env.observation_space.shape[0]  # 4
    out_dim = env.action_space.n  # 2
    pi = Pi(in_dim, out_dim)  # стратегия pi_theta для REINFORCE
    if Path("CartPole.pth").exists() and LOAD:
        pi.load_state_dict(torch.load("CartPole.pth"))

    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(10000):
        state, _ = env.reset()
        for t in range(200):  # 200 — максимальное количество шагов в cartpole
            action = pi.act(state)
            state, reward, done, truncated, _ = env.step(action)
            pi.rewards.append(reward)
            if done:
                break
        loss = train(pi, optimizer)  # обучение в эпизоде
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()  # обучение по актуальному опыту: очистить память после обучения

        if USE_WANDB:
            wandb.log({"total_reward ": total_reward, "loss": loss})
        else:
            print(f'Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}')

    if USE_WANDB:
        wandb.finish()
    #if not LOAD:
        #torch.save(pi.state_dict(), 'CartPole.pth')



if __name__ == '__main__':
    main()
