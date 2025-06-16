import random
import gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
observation_space_size = env.observation_space.n
print(observation_space_size)
action_space_size = env.action_space.n
print(action_space_size)
q_table = np.zeros((observation_space_size, action_space_size))
print(q_table)

total_episodes = 200  # Total episodes 训练次数
learning_rate = 0.8  # Learning rate 学习率
max_steps = 50  # Max steps per episode 一次训练中最多决策次数
gamma = 0.95  # Discounting rate 折扣率，对未来收益的折扣

# Exploration parameters
epsilon = 0.7

# For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    state = state[0]  # 我们需要第一个未知的参数state,prob暂时不用
    step = 0
    done = False

    for step in range(max_steps):
        # Choose an action a in the current world state (s)
        # First we randomize a number

        # 大概率根据Q表行动，也有一定概率随机行动
        if random.uniform(0, 1) < epsilon:
            # 根据Q值最大原则选取action，如果有多个action的Q相同且最大，则随机选取一个
            state_all = q_table[state, :]
            max_indices = np.argwhere(state_all == np.max(state_all)).flatten()
            action = np.random.choice(max_indices)
        else:
            # 随机选取一个行动
            action = env.action_space.sample()

        # 利用step函数求新状态、奖励、结果等，这里需要5个参数，可以看看源文档
        new_state, reward, done, truncated, info = env.step(action)
        # Q表更新，也就是Q-learning的核心，具体原理不再赘述，在别的博客有很多
        q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        # 这里自行修改Q值，加快运行速度，这里看似“作弊”，但是不影响Q-learning的原理
        # 比如靠左墙还左转，那么将Q值减为负值，下次直接跳过这个选择
        if state == new_state:
            q_table[state, action] = -1
        # 掉进河里也将Q值减为负值
        if done:
            if reward == 0:
                q_table[state, action] = -1
            break
        # Our new state is state
        state = new_state

    # Reduce epsilon (because we need less and less exploration) 随着智能体对环境熟悉程度增加，可以减少对环境的探索
    if epsilon < 0.95:
        epsilon = epsilon + 0.001