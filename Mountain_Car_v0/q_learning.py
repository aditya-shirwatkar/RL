import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('MountainCar-v0')
N = 20
#since our observation space is huge, we approximate it into 20 states
DISCRETE_SIZE = (N,N,env.action_space.n) #20 -> position, 20 -> velocity, env.actionSpace.n -> actions
# print([N,N] + [env.action_space.n])
LEARNING_RATE = 0.5
DISCOUNT = 0.99
EPISODES = 10000
EPSILON = 1
DECAY_RATE = 1.1

def get_discrete_state(state):
    discrete_state = ((state - env.observation_space.low)*N)/(env.observation_space.high - env.observation_space.low)
    return tuple(discrete_state.astype(np.int)) #since look q_table pass this directly

q_table = np.random.uniform(low = -3, high=3,size=DISCRETE_SIZE)

for eps in range(EPISODES):
    episode_reward = 0
    done = False
    state = get_discrete_state(env.reset())
    # print((q_table[state]))
    # action = np.argmax(q_table[state])
    if eps%500 == 0:
        render = True
    else:
        render = False

    while not done:
        if np.random.random() >= EPSILON:
            action = np.argmax(q_table[state])
        else:
            rand = np.random.randint(0, env.action_space.n)
            while rand is np.argmax(q_table[state]):
                rand = np.random.randint(0, env.action_space.n)
            action = rand

        new_state, reward, done, info = env.step(action)
        new_dstate = get_discrete_state(new_state)

        if not done:
            q_target = LEARNING_RATE*(reward + (DISCOUNT*np.max(q_table[new_dstate])) - q_table[state][action])
            q_table[state][action] += q_target
            # print(info)
        elif new_state[0] >= env.goal_position:
            print(f"Yay we reached at {eps} with {new_state[0]} from {env.goal_position}")
            q_table[state][action] = 0

        if render == True:
            env.render()
        # action = new_action
        state = new_dstate
        episode_reward += reward
    EPSILON /= DECAY_RATE
    # print(episode_reward)
env.close()