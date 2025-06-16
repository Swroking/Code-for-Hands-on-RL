import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import rl_utils


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用于训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)

        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断

            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
env.reset(seed=0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)

# 采样专家数据
def sample_expert_data(n_episode):
    states = []
    actions = []
    for episode in range(n_episode):
        state, _ = env.reset()
        done = False
        while not done:
            action = ppo_agent.take_action(state)
            states.append(state)
            actions.append(action)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            if truncated:
                break
    return np.array(states), np.array(actions)

env.reset(seed=0)
torch.manual_seed(0)
random.seed(0)
n_episode = 5 # 此处 1 改为 5
expert_s, expert_a = sample_expert_data(n_episode)
# print(expert_s.shape, expert_a.shape) # 采样的专家数据的size大小******************(200, 4) (200,)

n_samples = 200  # 采样30个数据 # 原始数据为 30 ，在 V1 版本下能学习得很好，V0 则不行，return约为 70~80，教材说因为数据量少易导致过拟合，故改为 128，return 变为 110~120，最后改为 200

# shape[0] 专家数据是二维数据[batch_size,state]，因此用[0]来取行数
# sample 随机采样 30 个数据
# range 的作用是创建一个长度与 expert_s 相等的 从 0 开始的列表，列表的采样值可以作为索引从 expert_s 中取值，实现采样

random_index = random.sample(range(expert_s.shape[0]), n_samples) # 可能导致遗漏重要的决策样本******************私以为应当改变采样方式，基于完整的专家数据进行训练
# 如采用 随机洗牌（shuffle）+ 小批量训练，比如 torch.utils.data.DataLoader 来更稳定地采样数据。 # 或对专家数据进行限制 return > 150 # 不过如何实现呢？ # 另，采取的动作是基于 random 进行随机的，或许可以添加噪声，如 ε-greedy

# 总结******************在 n_episode = 5 扩大了专家数据的容量 及 n_samples = 200 扩大了采样数避免过拟合 之后，取得了 return 在 60% 进度达到 181.82 在完成时取得了 return 达到 196.78，但是在最后 20 个 进度里，return 只提升了 4 左右
# 为了验证是否是由于数据不过充分（1、覆盖程度不足 2、训练量不足）最后又对 n_episode 及 n_samples 做了加倍处理，在 50% 进度左右就以取得了 190 以上的 return，66% 的 return 为197 。遗憾的是直至训练结束都没能到达 200 的汇报，甚至性能还出现了下降
# 此处也可以一次训练不同的值后汇总到一张图上做对比
# 最后一次，将迭代次数 n_iterations 减小了 200，采样专家数据次数 扩充了 10 倍 (== 50)，采样数恢复至 200，效果不错，70% 进度可以到达 199 以上的 return，80% 后短暂到达了 200 并往返于 (199, 200]

expert_s = expert_s[random_index]
expert_a = expert_a[random_index]
# print(expert_s.shape, expert_a.shape) # 经 sample 之后的专家数据的size大小******************(30, 4) (30,)

# # BC 算法实现
# class BehaviorClone:
#     def __init__(self, state_dim, hidden_dim, action_dim, lr):
#         self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
#         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
#
#     def learn(self, states, actions):
#         states = torch.tensor(states, dtype=torch.float).to(device)
#
#         actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(device) # 这里必须把 actions 变成 int64 格式才行，即 dtype=torch.long
#         # 在 PyTorch 中，gather(dim, index) 需要 index 是 int64(torch.long) 类型，否则会报 Expected dtype int64 for index 错误。
#         # 而 numpy 数组默认的数据类型可能是 int32 或 float32，所以需要手动转换。
#
#         log_probs = torch.log(self.policy(states).gather(1, actions))
#         bc_loss = torch.mean(-log_probs)  # 最大似然估计
#
#         self.optimizer.zero_grad()
#         bc_loss.backward()
#         self.optimizer.step()
#
#     def take_action(self, state):
#         state = torch.tensor([state], dtype=torch.float).to(device)
#         probs = self.policy(state)
#         action_dist = torch.distributions.Categorical(probs)
#         action = action_dist.sample()
#         return action.item()
#
#
# def test_agent(agent, env, n_episode):
#     return_list = []
#     for episode in range(n_episode):
#         episode_return = 0
#         state, _ = env.reset()
#         done = False
#         while not done:
#             action = agent.take_action(state)
#             next_state, reward, done, truncated, _ = env.step(action)
#             state = next_state
#             episode_return += reward
#             if truncated:
#                 break
#         return_list.append(episode_return)
#     return np.mean(return_list)
#
#
# env.reset(seed=0)
# torch.manual_seed(0)
# np.random.seed(0)
#
# lr = 1e-3
# bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr)
# n_iterations = 1000 - 200 # 迭代次数太多也会过拟合
# batch_size = 64
# test_returns = []
#
# with tqdm(total=n_iterations, desc="进度条") as pbar:
#     for i in range(n_iterations):
#         sample_indices = np.random.randint(low=0,
#                                            high=expert_s.shape[0],
#                                            size=batch_size)
#         # print(sample_indices) # 查看 sample 采样的数据的具体情况，是否重复******************有重复，因为 batch_size = 64 > n_samples = 30 # 因为效果不好，故改变了采样数量
#
#         bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices]) # 监督学习
#         # print(type(expert_a[sample_indices])) # 查看 expert_a[sample_indices] 的数据类型******************<class 'numpy.ndarray'>
#         # print(expert_a[sample_indices].dtype)# 查看具体的数据类型******************int32 # 注意 dtype 是属性，不能加 () 用函数来调用
#
#         current_return = test_agent(bc_agent, env, 5) # 计算 5 回合 BC 的平均回报
#         test_returns.append(current_return)
#
#         if (i + 1) % 10 == 0:
#             pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
#         pbar.update(1)
#
# iteration_list = list(range(len(test_returns)))
# plt.plot(iteration_list, test_returns)
# plt.xlabel('Iterations')
# plt.ylabel('Returns')
# plt.title('BC on {}'.format(env_name))
# plt.show()

# GAIL 算法实现 # 因 BC 未能达到最优策略，故提出 GAIL 应对 # BC 最大的劣势在于没法完全覆盖真实的情况，因为没和环境交互
class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))

class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d):
        self.discriminator = Discriminator(state_dim, hidden_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.agent = agent

    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)
        expert_actions = torch.tensor(expert_a, dtype=torch.long).to(device) # F.one_hot 的输入 必须是整数（索引张量），但 expert_actions 或 agent_actions 可能不是整数类型（torch.int64）。

        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)
        agent_actions = torch.tensor(agent_a, dtype=torch.long).to(device)

        # print(agent_actions) # 若不处理的动作值输出******************tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1], device='cuda:0')
        expert_actions = F.one_hot(expert_actions, num_classes=2).float() # float 转换是为了满足 Linear 层的输入
        agent_actions = F.one_hot(agent_actions, num_classes=2).float() # 独热编码操作是为了将 action 的数值转换为类别，避免网络单纯的比较 action 的数值大小 # num_classes=2 表示动作类别为两种 # 此处是因为动作是离散的，故而需要独热编码处理

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)

        # BCELoss 计算二元交叉熵损失，前项为概率，后项为标签
        discriminator_loss = (nn.BCELoss()(agent_prob, torch.ones_like(agent_prob)) +
                              nn.BCELoss()(expert_prob, torch.zeros_like(expert_prob)))

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        rewards = -torch.log(agent_prob).detach().cpu().numpy()
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        # 先更新判别器，再更新生成器
        self.agent.update(transition_dict)


env.reset(seed=0)
torch.manual_seed(0)
lr_d = 1e-3
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d)
n_episode = 500
return_list = []

with tqdm(total=n_episode, desc="进度条") as pbar:
    for i in range(n_episode):
        episode_return = 0
        state, _ = env.reset()
        done = False
        state_list = []
        action_list = []
        next_state_list = []
        done_list = []
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, truncated, _ = env.step(action)  # 此处的 reward 被省略了
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            done_list.append(done)
            state = next_state
            episode_return += reward
            if truncated:
                break
        return_list.append(episode_return)
        gail.learn(expert_s, expert_a, state_list, action_list, next_state_list, done_list)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)

iteration_list = list(range(len(return_list)))
plt.plot(iteration_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GAIL on {}'.format(env_name))
plt.show()

# 没什么好说的，速度杠杠的，开了眼了 # 训练进度至半时 return 已达最高



# 不过似乎采样专家数据次数太多对 GAIL 不太有利，次数越多其训练越不稳定
# 对比效果见图