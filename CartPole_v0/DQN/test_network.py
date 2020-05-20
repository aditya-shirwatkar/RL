import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from matplotlib import pyplot as plt
from matplotlib import animation as anime

plt.style.use('fivethirtyeight')
xval = []
yval = []

plt.ylabel('Reward')
plt.xlabel('episode')

def animate(i):
    plt.cla()
    plt.plot(xval, yval)

class DqnAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 572)
        self.fc2 = nn.Linear(572, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        return x

run = torch.load('4_572_256_2')
run.eval()
env = gym.make('CartPole-v0')
EPISODES = 1000

for eps in range(EPISODES):
    batch_reward = 0
    done = False
    state = (torch.Tensor(env.reset()))
    while not done:
        output = run(state)
        action = torch.argmax(output).detach().numpy()
        new_state, reward, done, _ = env.step(action)
        new_state = torch.Tensor(new_state)
        state = new_state
        env.render()
        batch_reward += 1

    xval.append(eps)
    yval.append(batch_reward)
    ani = anime.FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

env.close()
plt.close()
plt.ylabel('Reward')
plt.xlabel('episode')
plt.plot(xval, yval)