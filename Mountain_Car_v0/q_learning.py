import gym
import numpy as np

env = gym.make('MountainCar-v0')
N = 20
# since our observation space is huge, we approximate it into N states for quick learning
DISCRETE_SIZE = (N,N,env.action_space.n) # N -> position, N -> velocity, env.actionSpace.n -> actions
LEARNING_RATE = 0.5
DISCOUNT = 0.99
EPISODES = 10000
EPSILON = 1
DECAY_RATE = 1.001

def get_discrete_state(state):
    # transcription into N states
    discrete_state = ((state - env.observation_space.low)*N)/(env.observation_space.high - env.observation_space.low)
    return tuple(discrete_state.astype(np.int)) #since look q_table pass this directly

# initialize our q_table randomly
q_table = np.random.uniform(-3,3,DISCRETE_SIZE)

# make our agent learn 'EPISODES' times
for eps in range(EPISODES):
    done = False # represents whether our agent has finished learning in 'eps' episode
    state = get_discrete_state(env.reset()) # reseting and getting base position everytime our agent starts learning

    if eps%500 == 0: # since rendering slows down the program, we'll render every 500th episode
        render = True
    else:
        render = False

    while not done:
        # here we'll follow an epsilon-greedy policy
        if np.random.random() >= EPSILON:
            action = np.argmax(q_table[state])
        else:
            rand = np.random.randint(0, env.action_space.n)
            while rand is np.argmax(q_table[state]):
                rand = np.random.randint(0, env.action_space.n)
            action = rand

        # take a step into the environment
        new_state, reward, done, info = env.step(action)
        new_dis_state = get_discrete_state(new_state)

        # update our q table using Q-Learning
        if not done:
            q_target = LEARNING_RATE*(reward + (DISCOUNT*np.max(q_table[new_dis_state])) - q_table[state][action])
            q_table[state][action] += q_target
        elif new_state[0] >= env.goal_position:
            print(f"Yay we reached our goal {eps}th episode")
            q_table[state][action] = 0 # our terminal state

        if render == True:
            env.render()
        # action = new_action
        state = new_dis_state

    EPSILON /= DECAY_RATE
    # now we save our q_table
    if eps%500 == 0:
        np.save(f"qtables/{eps}-qtable.npy", q_table)
env.close()