import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
from matplotlib import pyplot as plt
import random
from collections import namedtuple
from matplotlib import animation as anime

xval = []
yval = []


def animate(i):
    plt.cla()
    plt.plot(xval, yval, 'r*')


env = gym.make('CartPole-v0')


# print(env.observation_space.n)

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(env.observation_space.high), 128)
        # self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, env.action_space.n)

    def forward(self, x):
        # print(x)
        x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
        # return x

#
# class LearnAcrobot:
#     def __init__(self, d_size, episodes):
#         self.D_SIZE = d_size
#         self.EPISODES = episodes
#         self.model = Agent()
#         self.optim = optim.SGD(self.model.parameters(), lr=8e-3)
#         self.trajectory = []
#         self.log_probs = []
#         self.rewards = []
#
#     def predict(self, s):
#         # if isinstance(s, list):
#         #     return self.model(torch.tensor(s).float().unsqueeze(0))
#         # else:
#         ss = self.model(torch.as_tensor(s).float())
#         # torch.clamp(ss, 0, 1)
#         # print(ss)
#         return torch.distributions.Categorical(ss)
#
def takeAction(s):
    s = model(torch.as_tensor(s).float().unsqueeze(0))
    prob = torch.distributions.Categorical(s)
    act = prob.sample()
    # print(act)
    # print(prob.log_prob(act))
    # self.log_probs[indx].append(prob.log_prob(act))
    return act.item(), prob.log_prob(act)
#
#     def train(self):
#         # print(self.rewards)
#         returns = torch.tensor(self.rewards)
#
#         # print(returns.size(0), returns.size(1))
#         # for jj in range(returns.size(0)):
#         ret = 0
#         for jjj in range(returns.size(0)):
#             ret += returns[jjj]
#             returns[jjj] = ret
#         g = returns[-1].item()
#
#         if returns.std() > 0:
#             # print(returns.mean(dim=1).unsqueeze(1), returns.std(dim=1).unsqueeze(1))
#             returns = (returns - returns.mean()) / (returns.std())
#         else:
#             returns = returns - returns.mean()
#         # print(returns)
#         # print(self.log_probs)
#         log_prob = torch.tensor(self.log_probs, requires_grad=True)
#         # print(log_prob)
#         loss = (-log_prob * returns).sum()
#         # print(loss)
#         self.optim.zero_grad()
#         loss.backward()
#         self.optim.step()
#         self.log_probs = []
#         self.rewards = []
#         # batch_states, batch_rewards, batch_actions = self.extractTensors(self.trajectory)
#         # Return = batch_rewards.sum()
#         # logp = self.predict(batch_states).log_prob(batch_actions)
#         # loss = (-(logp * Return).mean())
#         # loss.backward()
#         # # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
#         # self.optim.step()
#         # self.batch = []
#         # self.trajectory = []
#         return g, loss
#
#     def makeTrajectory(self, traj):
#         self.trajectory.append(traj)
#
#     def extractTensors(self, b):
#         # bb = Batch(*zip(*b))
#         # print(bb)
#         batch = Trajectory(*zip(*b))
#         t1 = torch.tensor(batch.state).float()
#         t3 = torch.tensor(batch.action).int()
#         t2 = torch.tensor(batch.reward).float()
#         return t1, t2, t3


# acro = LearnAcrobot(15, 100000)
Trajectory = namedtuple('Trajectory', ('state', 'action', 'reward'))
# Batch = namedtuple('BatReturnch', 'Trajectory')
model = torch.load('420-210')
model.eval()
win_count = 0
EPISODES = 100
for i in range(EPISODES):
    done = False
    state = env.reset()
    while not done:
        # print(state)
        # for state in states:
        action, log_prob = takeAction(state)
        new_state, reward, done, _ = env.step(action)
        # if done and reward == 0:
        #     reward = -5.0
        # elif not done and reward == 0:
        #     reward = 0
        state = new_state

        # if state[0] >= env.goal_position:
        #     win_count += 1
        #     print('we made it')

        # if i%10000 == 0:
        env.render()
        # acro.rewards.append(reward)
        # acro.log_probs.append(log_prob)

    # if i%100 == 0:
    #     print(loss)
    # g, loss = acro.train()
    # print(g)
    # xval.append(i % 100)
    # yval.append(g)
    # # if i%1000 == 0:
    # print(f'Loss at {i} is {loss}')
    # if i%1000 == 0:
    #     # ani = anime.FuncAnimation(plt.gcf(), animate, interval=1000)
    #     plt.plot(xval, yval, 'r')
    #     plt.tight_layout()
    #     plt.draw()
    #     plt.pause(0.001)
    #     plt.clf()
    #     xval = []
    #     yval = []
# torch.save(acro.model, '420-210(1)')
print(win_count)
env.close()
# plt.plot(xval, yval, 'r')
# plt.show()
plt.close()

# plt.plot(xval, yval, 'b', xval, zval , 'r')
# plt.show()
#
# l = []
# # print(rewards.numel(), rewards.sum().item())
# # print( distri)
# for j in range(len(self.trajectory)):
#     distri = self.trajectory[j][0]
#     rewards = self.trajectory[j][1]
#     actions = self.trajectory[j][2]
#     # print(actions[j])
#     l.append(distri.log_prob(actions) * rewards)
# loss = -sum(l)
# loss.backward()
# self.trajectory = []
# # loss = -(distri.log_prob(actions) * r)
# # loss.backward()
# self.optim.step()
# return loss

