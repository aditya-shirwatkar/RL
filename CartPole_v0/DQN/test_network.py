import torch
import gym
from matplotlib import pyplot as plt
from matplotlib import animation as anime

plt.style.use('fivethirtyeight')
xval = []
yval = []

def animate(i):
    plt.cla()
    plt.plot(xval, yval)

run = torch.load('4_572_256_2')
run.eval()
env = gym.make('CartPole-v0')
EPISODES = 1000

for eps in range(EPISODES):
    k = 0
    done = False
    state = (torch.Tensor(env.reset()))
    while not done:
        # print(k)
        output = run(state)
        action = torch.argmax(output).detach().numpy()
        # print(action)
        new_state,reward,done,_ = env.step(action)
        new_state = torch.Tensor(new_state)
        # replay_mem.append((state,action,reward,new_state,output))
        state = new_state
        env.render()
        k += 1

    xval.append(eps)
    yval.append(batch_reward)
    ani = anime.FuncAnimation(plt.gcf(), animate, interval=1000)
    if eps%cart.SHOW_EVERY == 0:
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        # cart.train(batch_reward)

env.close()
