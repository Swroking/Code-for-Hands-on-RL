import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt

# 本文的思想确实让人眼前一亮，如有时间可以读一下原文
# 我的理解是 UVFA( HER 的前身) 是 简单的 基于目标 构建 新的奖励，HER 在此基础上 对 目标 做了 动态转换，并构建了 伪奖励

# 构建二维环境
class WorldEnv:
    def __init__(self):
        self.distance_threshold = 0.15
        self.action_bound = 1

    def reset(self):  # 重置环境
        # 生成 1 个目标状态
        # 当然也可以是多个 只要每个 episode 采样前先随机选择 其中的一个目标 即可
        # 坐标范围是 [3.5-4.5, 3.5-4.5]
        self.goal = np.array([4 + random.uniform(-0.5, 0.5), 4 + random.uniform(-0.5, 0.5)])

        self.state = np.array([0, 0])  # 初始状态
        self.count = 0
        return np.hstack((self.state, self.goal))
        # [0.         0.         3.76993187 4.39183987]

    def step(self, action):
        action = np.clip(action, -self.action_bound, self.action_bound)
        x = max(0, min(5, self.state[0] + action[0]))
        y = max(0, min(5, self.state[1] + action[1]))
        self.state = np.array([x, y])
        self.count += 1

        dis = np.sqrt(np.sum(np.square(self.state - self.goal))) # 计算欧几里得距离
        reward = -1.0 if dis > self.distance_threshold else 0 # 设定奖励

        if dis <= self.distance_threshold or self.count == 50:
            done = True
        else:
            done = False

        return np.hstack((self.state, self.goal)), reward, done

# 定义网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return torch.tanh(self.fc3(x)) * self.action_bound

class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc2(F.relu(self.fc1(cat))))
        return self.fc3(x)

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim,  action_dim).to(device)

        # 初始化目标价值网络并使其参数和价值网络一样
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并使其参数和策略网络一样
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为 0
        self.tau = tau  # 目标网络软更新参数

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0] # 与 DDPG 不同，item 只适用于 1 维 tensor
        # print(self.actor(state).detach().cpu().numpy().shape) # [1,2]
        # print(self.actor(state).detach().cpu().numpy()[0]) # [-0.32608196 -0.07984246]

        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # MSE损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 策略网络就是为了使 Q 值最大化
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

class Trajectory:
    ''' 用来记录一条完整轨迹 '''
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1
        # states 有 T + 1个，因为每个 action 导致一个 next_state
        # actions / rewards / dones 有 T 个

class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 创建队列，先进先出
        # self.buffer ≈ [traj_1, traj_2, traj_3, ..., traj_n]

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, dis_threshold=0.15, her_ratio=0.8):
        # use_her: 是否使用
        # dis_threshold: 成功的判定阈值，即距离小于多少算成功
        # her_ratio: 使用 HER 的比例

        batch = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])

        # 从 random batch_size 条 轨迹 中 采样 random batch_size 个 数据
        for _ in range(batch_size):
            # 随机采样一条 trajectory
            traj = random.sample(self.buffer, 1)[0] # sample 返回的是一个列表 list[traj_1] ，但是这个列表只包含 1 条轨迹
            # 这里的 [0] 是指 取出那条轨迹对象本身

            # print(random.sample(self.buffer, 1)) # [<__main__.Trajectory object at 0x00000134D6D4B6A0>]
            # print(traj) #< __main__.Trajectory object at 0x000001B0FDEAB160 >
            # print(traj.length) # 50
            # print(len(traj.states)) # 51
            # print(len(traj.actions)) # 50

            # 从这条轨迹中 随机采样一个 step
            step_state = np.random.randint(traj.length)

            # 读取状态转移数据
            state = traj.states[step_state]
            next_state = traj.states[step_state + 1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.dones[step_state]

            # 启用 HER
            if use_her and np.random.uniform() <= her_ratio:
                # 随机选择未来状态的索引值
                step_goal = np.random.randint(step_state + 1, traj.length + 1)

                # step_goal = traj.length # 当然也可以选择 final 状态作为 goal 但是效果好像不太好

                # 使用 HER 算法的 future 方案设置目标
                goal = traj.states[step_goal][:2] # 获取 未来状态 的 前两个维度 作为 目标坐标（位置），后两个维度是 原始的 未来状态的 目标
                # print(traj.states[step_goal]) # [4.01540966 3.34925844 3.96925781 3.88597348]

                dis = np.sqrt(np.sum(np.square(next_state[:2] - goal)))
                reward = -1.0 if dis > dis_threshold else 0 # 重新计算奖励值
                done = False if dis > dis_threshold else True # 更新 done

                # 重写 state 和 next_state 的 goal 部分，形成新的 伪经验
                state = np.hstack((state[:2], goal))
                next_state = np.hstack((next_state[:2], goal))

            # 写入状态转移数据
            batch['states'].append(state)
            batch['next_states'].append(next_state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        # 将原本的 Python 列表转换为 NumPy 数组 方便神经网络的训练

        # print(type(batch['states'])) # <class 'list'>
        batch['states'] = np.array(batch['states'])  # shape: (batch_size, state_dim)
        # print(type(batch['states'])) # <class 'numpy.ndarray'>
        # print(batch['states'].shape) # (256, 4)

        batch['next_states'] = np.array(batch['next_states'])
        batch['actions'] = np.array(batch['actions'])

        return batch

actor_lr = 1e-3
critic_lr = 1e-3
hidden_dim = 128
state_dim = 4
action_dim = 2
action_bound = 1
sigma = 0.1
tau = 0.005
gamma = 0.98
num_episodes = 2000
n_train = 20
batch_size = 256
minimal_episodes = 200
buffer_size = 10000
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
env = WorldEnv()
replay_buffer = ReplayBuffer_Trajectory(buffer_size)
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, tau, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            traj = Trajectory(state) # 初始化轨迹
            done = False

            # 持续与环境交互直到 episode 结束
            while not done:
                action = agent.take_action(state)
                state, reward, done = env.step(action) # 👈 这里的 state 是 next_state，初始化时 self.states = [init_state] 已经占了一格
                episode_return += reward
                traj.store_step(action, state, reward, done)

            # 每条完整轨迹（trajectory）被存入轨迹经验池
            replay_buffer.add_trajectory(traj)
            # 存储 episode 的总奖励，供后续画图
            return_list.append(episode_return)

            # 如果 数据足够 则 学习更新
            if replay_buffer.size() >= minimal_episodes:
                # 每个 episode 后进行 n_train = 20 次更新
                for _ in range(n_train):
                    transition_dict = replay_buffer.sample(batch_size, True)
                    agent.update(transition_dict)

            # 打印和更新进度
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG with HER on {}'.format('GridWorld'))
plt.show()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
env = WorldEnv()
replay_buffer = ReplayBuffer_Trajectory(buffer_size)
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, tau, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            traj = Trajectory(state)
            done = False
            while not done:
                action = agent.take_action(state)
                state, reward, done = env.step(action)
                episode_return += reward
                traj.store_step(action, state, reward, done)
            replay_buffer.add_trajectory(traj)
            return_list.append(episode_return)
            if replay_buffer.size() >= minimal_episodes:
                for _ in range(n_train):
                    # 和使用HER训练的唯一区别
                    transition_dict = replay_buffer.sample(batch_size, False)
                    agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG without HER on {}'.format('GridWorld'))
plt.show()

# future
# Iteration 0: 100%|██████████| 200/200 [00:11<00:00, 18.15it/s, episode=200, return=-100.000]
# Iteration 1: 100%|██████████| 200/200 [01:05<00:00,  3.03it/s, episode=400, return=-84.400]
# Iteration 2: 100%|██████████| 200/200 [01:05<00:00,  3.07it/s, episode=600, return=-90.700]
# Iteration 3: 100%|██████████| 200/200 [01:08<00:00,  2.94it/s, episode=800, return=-71.800]
# Iteration 4: 100%|██████████| 200/200 [01:19<00:00,  2.53it/s, episode=1000, return=-71.700]
# Iteration 5: 100%|██████████| 200/200 [01:04<00:00,  3.10it/s, episode=1200, return=-72.300]
# Iteration 6: 100%|██████████| 200/200 [01:03<00:00,  3.15it/s, episode=1400, return=-90.500]
# Iteration 7: 100%|██████████| 200/200 [01:04<00:00,  3.09it/s, episode=1600, return=-62.000]
# Iteration 8: 100%|██████████| 200/200 [01:03<00:00,  3.15it/s, episode=1800, return=-71.300]
# Iteration 9: 100%|██████████| 200/200 [01:05<00:00,  3.06it/s, episode=2000, return=-43.800]
# Iteration 0: 100%|██████████| 200/200 [00:10<00:00, 18.65it/s, episode=200, return=-100.000]

# final
