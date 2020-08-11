
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import numpy as np
from matplotlib import pyplot as plt
import random
from collections import namedtuple

# initialise reward graphs
plt.style.use('fivethirtyeight')
xval = []
yval = []

# make a named tuple for replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


# Make a neural network architecture, the below one gave decent results
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

# make a class for our cartpole for easy functionality
class LearnCartPole:
    def __init__(self, mem_size, batch_size, interval, train_length):
        self.run = DqnAgent()
        self.teach = DqnAgent()
        self.teach.load_state_dict(self.run.state_dict())
        self.teach.eval()
        self.mem_size = mem_size
        self.env = gym.make('CartPole-v0')
        self.EPISODES = train_length
        self.replay = []
        self.interval = interval
        self.batch_size = batch_size

        # you can tweak the below values to improve the results
        self.optimiser = optim.Adam(self.run.parameters(), lr=1e-3)
        self.DISCOUNT = 0.99
        self.EPSILON = 1
        self.DECAY_RATE = 1.5
        self.MIN_EPSILON = 0.01

# A function to store the experience tuple
    def store_mem(self, experiance):
        if len(self.replay) < self.mem_size:
            self.replay.append(experiance)
        else:
            self.replay[np.random.randint(0, len(self.replay))] = experiance
        random.shuffle(self.replay)

# A function to get np array from replay memory
    def extract_tensors(self, experiences):
        batch = Experience(*zip(*experiences))
        t1 = np.asarray(batch.state)
        t2 = np.asarray(batch.action)
        t3 = np.asarray(batch.reward)
        t4 = np.asarray(batch.next_state)
        t5 = np.asarray(batch.done)
        return t1, t2, t3, t4, t5

# To clear memory if needed
    def delete_mem(self):
        self.replay = []

# Forward pass
    def predict(self, net, data):
        return net(torch.Tensor(data))

# A function for epsilon greedy policy
    def predict_action(self, qvalue):
        if np.random.random() >= self.EPSILON:
            return torch.argmax(qvalue).detach().numpy()
        else:
            return self.env.action_space.sample()

# A function for getting current q values (of runner agent)
    def get_qvalues(self, agent, s, a):
        aa = (torch.tensor(a, requires_grad=False)).long()
        return self.predict(agent, s).gather(dim=1, index=aa.unsqueeze(-1))

# A function for getting future q values (of teacher agent)
    def get_nextqvalues(self, agent, ns):
        q = self.predict(agent, ns)
        a_next = q.argmax(dim=-1)
        return q.gather(dim=1, index=a_next.unsqueeze(-1))

# Train your runner agent based with the reference of teacher agent
    def train(self, br, k):
        sample = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, terminal_states = self.extract_tensors(sample)
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        terminal_states = torch.tensor(terminal_states, dtype=torch.float).unsqueeze(dim=-1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(dim=-1)

        run_predict = self.get_qvalues(self.run, states, actions)
        teach_predict = self.get_nextqvalues(self.teach, next_states)
        target_qvalues = teach_predict * (1 - terminal_states) * self.DISCOUNT + rewards

        loss = F.mse_loss(run_predict, target_qvalues)
        loss.backward()
        self.optimiser.step()

        if k % self.interval == 0:
            print(loss, br, self.EPSILON)

        self.optimiser.zero_grad()
        # self.run.zero_grad()


# you can tweak the below values to improve the results
# mem_size = 2048, batch_size = 572, interval = 64, episodes = 600
cart = LearnCartPole(2048, 572, 64, 600)

# To record the behavior in mp4 format, uncomment below line
# cart.env = gym.wrappers.Monitor(cart.env, "vid", video_callable=lambda episode_id: episode_id%(cart.interval) == 0, force=True)


for eps in range(cart.EPISODES):
    done = False
    state = cart.env.reset()
    batch_reward = 0

    # main loop for traversing the environment for each episode
    while not done:
        output = cart.predict(cart.run, state)
        action = cart.predict_action(output)
        new_state, reward, done, _ = cart.env.step(action)

        cart.store_mem(Experience(state, action, reward, new_state, done))
        state = new_state
        batch_reward += reward

        if eps % cart.interval == 0:
            cart.env.render()

    if len(cart.replay) >= cart.batch_size:
        cart.train(batch_reward, eps)

    if eps % cart.interval == 0:
        # decay your epsilon
        if cart.EPSILON >= cart.MIN_EPSILON:
            cart.EPSILON /= cart.DECAY_RATE
        # update your teacher agent
        cart.teach.load_state_dict(cart.run.state_dict())

    xval.append(eps)
    yval.append(batch_reward)

# uncomment below to save your model
# torch.save(cart.run, '4_572_256_2')

cart.env.close()

# plot reward vs episode garph
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.plot(xval, yval)
plt.show()
