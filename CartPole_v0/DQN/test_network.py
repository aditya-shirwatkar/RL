from dqn import *

run = torch.load('final_new2')
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

env.close()
