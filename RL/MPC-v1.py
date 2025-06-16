import numpy as np
from scipy.stats import truncnorm
import gymnasium as gym
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt

# 通过采样、评估和更新，优化 MPC 的动作选择策略，并返回一个动作均值
class CEM:
    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound, lower_bound):
        self.n_sequence = n_sequence  # 采样的动作序列条数 50
        self.elite_ratio = elite_ratio  # 保留累积奖励最高的比例 0.2
        self.upper_bound = upper_bound  # 动作最大值 2
        self.lower_bound = lower_bound  # 动作最小值 -2
        self.fake_env = fake_env  # 训练的环境模型 (Ensemble Model)

    # 传入 s, mean, var 返回一个较可靠的 mean 作为动作的均值
    def optimize(self, state, init_mean, init_var):
        # 以下注释以第一次传入的数据为例
        mean, var = init_mean, init_var # shape(25,) # 均值 0 ,方差 1
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var)) # 截断标准正态分布
        # 由 lower = -2 ,upper = 2 # 得到取值范围 [μ + lower * σ, μ + upper * σ] # 即 [-2, 2]

        state = np.tile(state, (self.n_sequence, 1))  # 每条序列都从相同 state 开始 # shape (3, ) to (50, 3)

        # 进行 5 轮优化
        for _ in range(5):
            # 计算约束方差
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean # 确定均值离上下界的距离
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var) # 约束 var 确保采样范围不会超过 mean ± 2σ
            # 生成 50 条动作序列，每条序列步长为 25
            action_sequences = [X.rvs() for _ in range(self.n_sequence)] * np.sqrt(constrained_var) + mean
            # X.rvs() 从截断标准正态分布中采样，转换为 a ~ N(mean, constrained_var) # shape (50, 25)

            # 计算每条动作序列的累积奖励
            returns = self.fake_env.propagate(state, action_sequences)[:, 0] #  # shape (50,1) to (50, )，应该是为了便于对 return 排序

            # 选取累积奖励最高的若干条动作序列
            elites = action_sequences[np.argsort(returns)] # 按 return 升序排列 动作序列
            # print(elites.shape) # (50, 25)
            elites = elites[-int(self.elite_ratio * self.n_sequence):] # 截断最高 return 的动作序列，共 50 * 0.2 = 10 条

            new_mean = np.mean(elites, axis=0) # 计算第一个动作的均值 # shape (25,)
            new_var = np.var(elites, axis=0)
            # 平滑更新动作序列分布
            mean = 0.1 * mean + 0.9 * new_mean
            var = 0.1 * var + 0.9 * new_var
            # 用精英样本的均值 new_mean 作为新的动作均值，保证后续采样的动作更倾向于高回报区域
            # 用精英样本的方差 new_var 作为新的动作方差，逐步收缩搜索空间，减少无效探索

        return mean

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 一个激活函数
class Swish(nn.Module):
    ''' Swish激活函数 '''
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# 用于初始化 EnsembleModel 里的 FCLayer 层的参数
def init_weights(m):
    ''' 初始化模型权重 '''
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std) # 从给定均值和标准差的正态分布 N(mean, std) 中生成值，填充输入的张量，默认 μ = 0， σ = 1
        while True:
            cond = (t < mean - 2 * std) | (t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape, device=device), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, FCLayer): # 检查 m 是否是 nn.Linear 层或者自定义的 FCLayer 层。如果是，则执行权重初始化
        # 根据输入维度 m._input_dim 计算标准差
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0) # m.bias是该层的偏置参数，.data表示直接操作其数值，fill_(0.0)将所有偏置值设置为 0

# 这一步的目的在于构建一个全连接层，定义运算 并 得到结果 output = X * W + b
class FCLayer(nn.Module):
    ''' 集成之后的全连接层 '''
    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self._input_dim, self._output_dim = input_dim, output_dim
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, input_dim, output_dim).to(device)) # 可训练的参数
        self._activation = activation
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, output_dim).to(device)) # 可训练的参数

    def forward(self, x):
        # 以下注释以 **第一次** 传入的数据为例
        # x.shape = (5, 64, 4) # 第一次传入的数据形状
        # 第一次的 weight shape 为 (5, 4, 200) # self.bias.shape 为 (5, 200)

        a = torch.bmm(x, self.weight) # torch.Size([5, 64, 200])
        # print(a.shape, self.weight.shape, self.bias.shape) # torch.Size([5, 64, 200]) torch.Size([5, 4, 200]) torch.Size([5, 200])

        return self._activation(torch.add(a, self.bias[:, None, :])) # 返回 (5, batch_size, output_dim)

        # return self._activation(torch.add(torch.bmm(x, self.weight), self.bias[:, None, :]))
        # torch.bmm 矩阵乘法 ，参数 1 (batch_size, dim_a, dim_b)，参数 2 (batch_size, input_dim, output_dim) # 返回 (batch_size, dim_a, output_dim)

# 定义一个模型，可以调用该模型进行前向传播，并训练该模型
class EnsembleModel(nn.Module):
    # 集成 (Ensemble) 指的是多个模型同时训练，并通过平均或投票的方式提高泛化能力。
    # 在 EnsembleModel 里： ensemble_size = 5，意味着 5 个独立的神经网络在训练。
    # 每个 FCLayer 其实是 5 组参数的集合，每个网络成员独立学习但共享输入。
    ''' 环境模型集成 '''
    def __init__(self, state_dim, action_dim, ensemble_size=5, learning_rate=1e-3):
        # __init__ 函数 在 创建实例 的时候 就 执行该函数，因此以下的语句只会执行一次
        super(EnsembleModel, self).__init__()
        # 输出包括均值和方差,因此是状态与奖励维度之和的两倍
        self._output_dim = (state_dim + 1) * 2 # 前半部分是同时预测的状态和奖励，后半部分就是下面计算的方差 # 形状为 (8,)，即 8 维数据

        # 由于模型同时学习方差信息（不确定性），为了防止预测方差过大或过小，设定最大方差和最小方差：
        # self._max_logvar 设为 0.5，防止预测不确定性无限大。 如果方差过大，模型可能会过度自信，导致收敛困难。
        # self._min_logvar 设为 - 10，防止预测不确定性无限小。 如果方差过小，模型可能会忽略不确定性，导致预测不稳定。\

        self._max_logvar = nn.Parameter((torch.ones((1, self._output_dim // 2)).float() / 2).to(device), requires_grad=False)
        # print(self._max_logvar)
        # tensor([[0.5000, 0.5000, 0.5000, 0.5000]], device='cuda:0') # shape(1, 4)
        self._min_logvar = nn.Parameter((-torch.ones( (1, self._output_dim // 2)).float() * 10).to(device), requires_grad=False)

        # 输入的数据形状为 (ensemble_size, batch_size, state_dim + action_dim) 即 (5, 64, 4)
        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size, Swish())
        # ensemble_size 主要用于并行训练多个神经网络，在 FCLayer 中，它意味着这个全连接层包含 ensemble_size 组 **独立** 的权重和偏置
        # 每一 FCLayer 层都有多个网络，每个网络 独立接收 **相同的数据** ，但用自己的参数去计算前向传播。
        # 相比较传统的全连接层，默认是单一网络，每次前向传播的输入形状是 (batch_size, in_features)

        self.layer2 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size, nn.Identity()) # 返回 (5, 64, 8)
        # 这里的 nn.Identity() 主要起到占位作用，表示最后一层不使用激活函数。

        self.apply(init_weights)  # 初始化环境模型中的参数，权重矩阵为随机的截断正态分布，偏置为 0
        # self.apply() 会遍历 EnsembleModel 内的所有 nn.Module 子模块 (如 FCLayer)
        # 每个子模块 (如 FCLayer) 都会执行 init_weights(module)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) # self.parameters() 会返回当前模型的所有可训练参数。
        # 等价于:
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.layer1.parameters()},
        #     {'params': self.layer2.parameters()},
        #     {'params': self.layer3.parameters()},
        #     {'params': self.layer4.parameters()},
        #     {'params': self.layer5.parameters()},
        #     {'params': self._max_logvar},  # 也在优化中 # 应该
        #     {'params': self._min_logvar}
        # ], lr=learning_rate)

    # 此部分在数据输入之后，自动前向传播，输出 均值 对数方差
    def forward(self, x, return_log_var=False):
        # x.shape = (5, 64, 4) # return_log_var = True

        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        # ret.shape = (ensemble_size, batch_size, output_dim) 即 (5, 64, 8)

        mean = ret[:, :, :self._output_dim // 2]

        # 在 PETS 算法中，将方差控制在最小值和最大值之间
        logvar = self._max_logvar - F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:]) # softplus(x) = log(1 + e^x)
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)

        return mean, logvar if return_log_var else torch.exp(logvar) # 返回 均值 和 **对数方差** # (5, 64, 4) (5, 64, 4)

    # 此部分用于计算 前向传播 输出的值 与 label 标签之前的损失 # 返回一个 total_loss 总误差-标量，一个均方误差 mse_loss (5,)
    def loss(self, mean, logvar, labels, use_var_loss=True):
        inverse_var = torch.exp(-logvar) # 方差倒数
        # 输入的形状均为 (5, 64, 4)
        if use_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inverse_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            # (ensemble_size, batch_size, output_dim) to (ensemble_size, ) # (5, 64, 4) to (5,)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    # 此部分用于 反向传播 计算出的 loss 值
    def train(self, loss):
        self.optimizer.zero_grad()
        # 额外添加方差正则项，防止 max_logvar 过大或 min_logvar 过小。
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(self._min_logvar)
        loss.backward()
        self.optimizer.step()

# 该模型可用于训练，并使用该模型预测
# 即 状态转移模型
class EnsembleDynamicsModel:
    ''' 环境模型集成,加入精细化的训练 '''
    def __init__(self, state_dim, action_dim, num_network=5):
        self._num_network = num_network
        self._state_dim, self._action_dim = state_dim, action_dim
        self.model = EnsembleModel(state_dim, action_dim, ensemble_size=num_network) # 创建并初始化实例
        self._epoch_since_last_update = 0

    # 基于当前经验池的数据不断训练模型，直到满足一定条件
    def train(self, inputs, labels, batch_size=64, holdout_ratio=0.1, max_iter=20):
        # inputs 和 label 的 shape = (200 * episode, 4)
        # 200 * episode 是经验池的数据总量 # 以下仍以 200 做注释
        # shape = (200, 4)

        # 设置训练集与验证集
        permutation = np.random.permutation(inputs.shape[0]) # 返回 0-200 之间的随机序列，实现对数据随机排序
        inputs, labels = inputs[permutation], labels[permutation] # **打乱原始数据顺序*** 第 1 次乱序
        # 此次乱序用于划分数据集和测试集，只进行一次，以免下面多次循环训练时测试集混入训练集

        num_holdout = int(inputs.shape[0] * holdout_ratio) # 验证集取 20 个数据
        # 划分 数据集 和 验证集
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:] # (180, 4)
        holdout_inputs, holdout_labels = inputs[: num_holdout], labels[: num_holdout] # (20, 4)
        # print(train_inputs.shape, holdout_inputs.shape)

        # holdout_inputs 和 holdout_labels 被转换为 PyTorch 张量并移动到 device
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)

        # repeat([self._num_network, 1, 1]) 作用是为 多个神经网络副本复制相同的验证数据（通常是多个模型并行训练）。
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_network, 1, 1]) # (20, 4) to (5, 20, 4)
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_network, 1, 1])
        # print(holdout_inputs.shape, holdout_labels.shape)

        # 保留最好的结果
        self._snapshots = {i: (None, 1e10) for i in range(self._num_network)} # 存储当前最优的模型参数和对应的最小损失值。
        # print(self._snapshots) # {0: (None, 10000000000.0), 1: (None, 10000000000.0), 2: (None, 10000000000.0), 3: (None, 10000000000.0), 4: (None, 10000000000.0)}

        # 用当前经验池的数据不断训练模型，知道训练次数达到要求 OR 连续 5 轮训练模型未改善
        for epoch in itertools.count(): # 无限循环，epoch 递增
            # 定义每一个网络的训练数据
            # 这里创建一个(num_network, train_size) 形状的索引矩阵，每一行都是训练数据的随机排列索引，确保 每个网络 的 训练数据 顺序不同
            # *** 第 2 次乱序 # 此次乱序只针对测试集
            train_index = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self._num_network)]) # (5, 180)
            # print(train_index.shape)

            # 训练一次模型，所有真实数据都用来训练，直至经验池数据用完
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size): # 按批次 batch_size=64 取数据 # range(0, 180, 64)
                batch_index = train_index[:, batch_start_pos:batch_start_pos + batch_size] #
                # print(batch_index.shape) # (5, 64) (5, 64) (5, 52) 照此规律循环 # 由 (5, 180) 拆分而来

                # 每次取出 batch_size 个数据用于训练 5 个神经网络
                # 同常见的监督学习
                train_input = torch.from_numpy(train_inputs[batch_index]).float().to(device)
                # print(train_inputs[batch_index].shape)
                # (5, 64, 4) (5, 64, 4) (5, 52, 4) 照此规律循环

                # amazing!!! 形状变化 train_inputs # (180, 4) batch_index # (5, 64) train_inputs[batch_index] # (5, 64, 4)
                train_label = torch.from_numpy(train_labels[batch_index]).float().to(device) # (5, 64, 4)

                # 把数据传入 自定义的 EnsembleModel 训练
                mean, logvar = self.model(train_input, return_log_var=True) # 在 EnsembleModel 模型中进行一次前向传播 # 返回 (5, 64, 4) (5, 64, 4)
                loss, _ = self.model.loss(mean, logvar, train_label) # loss 是一个标量值
                self.model.train(loss) # 反向传播，更新一次参数

            # 经验池数据全部用完之后，测试一次测试集 # 并记录训练信息，判断是否提前结束训练
            with torch.no_grad(): # 关闭梯度计算，避免计算图的构建 # 因为是验证集
                mean, logvar = self.model(holdout_inputs, return_log_var=True) # 在 EnsembleModel 模型中进行一次前向传播 # 此次使用 **测试集**
                _, holdout_losses = self.model.loss(mean, logvar, holdout_labels, use_var_loss=False) # 取每个网络的 MSE 损失，shape = (5,)
                holdout_losses = holdout_losses.cpu() # 从 GPU 移到 CPU，方便处理
                break_condition = self._save_best(epoch, holdout_losses)
                # 连续 5 次损失值不改善 or epoch 迭代次数达到上限就早停
                if break_condition or epoch > max_iter:  # 结束训练
                    break

    # 检查当前模型是否比之前的最佳模型有足够的改进，并决定是否早停
    def _save_best(self, epoch, losses, threshold=0.1):
        # threshold = 0.1：改进阈值，默认设为 10 %（即损失相对于最优值减少 10 % 才算显著提升）
        updated = False
        for i in range(len(losses)):
            # print(len(losses)) # 5
            current = losses[i] # 第 i 个网络的损失值
            _, best = self._snapshots[i] # 取出存储的该网络损失值
            improvement = (best - current) / best # 即 current < 0.9 * best 就更新损失值
            if improvement > threshold:
                self._snapshots[i] = (epoch, current)
                # print(self._snapshots)
                updated = True
        # 有一个网络的损失值得到改善就 return false
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5

    # 传入一批样本，返回其预测 mean 和 var
    def predict(self, inputs, batch_size=64):
        mean, var = [], []
        for i in range(0, inputs.shape[0], batch_size):

            input = inputs[i:min(i + batch_size, inputs.shape[0])] # 这里其实就是取全部的输入数据
            # print(input.shape) # (50, 4)
            input = torch.from_numpy(input).float().to(device)

            input = input[None, :, :].repeat([self._num_network, 1, 1]) # 扩充到五维，对应五个神经网络
            # print(input.shape) # (5, 50, 4)

            cur_mean, cur_var = self.model(input, return_log_var=False) # 返回 均值 方差 # 均为 (5, 50, 4)

            mean.append(cur_mean.detach().cpu().numpy())
            var.append(cur_var.detach().cpu().numpy())
        return np.hstack(mean), np.hstack(var) # 在 axis = 1 维拼接 # 不过似乎有些语句没用到

# 计算每条序列的 return，完全在 虚拟环境（状态转移模型）中 走完所有动作序列
class FakeEnv:
    def __init__(self, model):
        self.model = model # 状态转移模型

    # 传入状态和单个动作，返回 状态转移模型 预测 的 reward 和 next_state # s, a to r, s'
    # (50, 3) (50, 1) to (50, 1) (50, 3)
    def step(self, obs, act):
        # (50, 3) (50, 1)
        inputs = np.concatenate((obs, act), axis=-1) # (50, 4) # 50 条轨迹的 s 和 a 拼接作为输入 # 可以理解为 50 个样本

        # 传入状态转移模型预测
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        # print(ensemble_model_means.shape) # (5, 50, 4)

        ensemble_model_means[:, :, 1:] += obs.numpy() # 取出 Δs ，计算 s' = s + Δs # 由 [r, Δs] 变成了 [r, s' ]
        ensemble_model_stds = np.sqrt(ensemble_model_vars) # 计算标准差
        # 生成符合 N(0, std^2) 分布的随机噪声
        ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
        # 从模型预测的分布中进行采样，避免只使用均值预测 # (5, 50, 4)

        num_models, batch_size, _ = ensemble_model_means.shape # 接收参数
        models_to_use = np.random.choice([i for i in range(num_models)], size=batch_size) # 5O 条序列，为每一条序列随机选一个模型
        # [0 1 2 4 3 4 3 0 1 0 3 4 0 1 0 1 4 0 2 3 1 1 0 0 1 1 4 1 4 4 1 1 4 3 1 3 3 2 4 0 2 0 1 1 0 1 1 0 1 0]
        # print(models_to_use.shape) # (50,)

        batch_inds = np.arange(0, batch_size)
        # print(batch_inds)
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        # 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
        samples = ensemble_samples[models_to_use, batch_inds]
        # print(samples.shape) # (50,4)
        # models_to_use[i] 选出第 i 个数据点使用的模型索引，即随机从 5 模型中选择 1 个用于该数据点
        # batch_inds[i] 选出该数据点的位置

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        # 分割 reward (50, 1) 和 next_state (50, 3)
        return rewards, next_obs

    # 输入状态和动作序列，返回 状态转移模型 预测的每条序列的 return
    def propagate(self, obs, actions):
        # (50, 3) (50, 25)
        with torch.no_grad():  # 禁用梯度计算，提高计算效率
            obs = np.copy(obs)  # 复制输入的观测值，避免修改原始数据
            total_reward = np.expand_dims(np.zeros(obs.shape[0]), axis=-1)  # 初始化累积奖励数组，shape 为 (50, 1)
            obs, actions = torch.as_tensor(obs), torch.as_tensor(actions)  # 将 NumPy 数组转换为 PyTorch 张量
            # 计算 return
            for i in range(actions.shape[1]):  # 遍历动作序列的每个时间步 # 25 次循环
                action = torch.unsqueeze(actions[:, i], 1)  # 取出当前时间步的动作，并增加一个维度，shape (50,) to (50, 1)
                rewards, next_obs = self.step(obs, action)  # 通过模型环境预测下一步状态和奖励 # (50, 1) (50, 3)
                total_reward += rewards  # 累积奖励
                obs = torch.as_tensor(next_obs)  # 更新观测值，进入下一步循环
            return total_reward  # 返回所有时间步的累积奖励

# 经验回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        return len(self.buffer)

    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done # state 为啥要 np 呢？？？

class PETS:
    ''' PETS算法 '''
    def __init__(self, env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes):
        # 定义 真实环境 和 经验池
        self._env = env
        self._env_pool = replay_buffer
        # 定义训练模型所需的参数
        obs_dim = env.observation_space.shape[0] # 状态空间的维度
        self._action_dim = env.action_space.shape[0] # 动作空间的维度
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim)
        # 定义虚拟环境和动作值上下界
        self._fake_env = FakeEnv(self._model)
        self.upper_bound = env.action_space.high[0]
        # print('upper_bound', self.upper_bound) # 2
        self.lower_bound = env.action_space.low[0]
        # print('lower_bound', self.lower_bound) # -2

        self._cem = CEM(n_sequence, elite_ratio, self._fake_env, self.upper_bound, self.lower_bound)
        self.plan_horizon = plan_horizon # 规划的时间步长，也就是 MPC 要优化未来多少步的动作。
        self.num_episodes = num_episodes # 模型训练-测试 迭代次数

    # 训练模型
    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = env_samples[0]
        # print(obs.shape) # 随迭代次数每 200 递增 # (200, 3)

        actions = np.array(env_samples[1])
        # print(actions.shape) # (200, 1)

        rewards = np.array(env_samples[2])
        # print(rewards.shape) # (200,)
        rewards = rewards.reshape(-1, 1) # 改变形状便于与 state 拼接
        # print(rewards.shape) # (200, 1)

        next_obs = env_samples[3]
        inputs = np.concatenate((obs, actions), axis=-1) # 行向量延长-拼接 # 与 label 形状均为 (200, 4)
        labels = np.concatenate((rewards, next_obs - obs), axis=-1)
        self._model.train(inputs, labels)

    # 模型预测控制
    # 基于环境模型（FakeEnv）和 CEM 进行规划（Planning），通过优化未来动作来最大化累积奖励。
    def mpc(self):
        # 初始化均值和方差
        mean = np.tile((self.upper_bound + self.lower_bound) / 2.0, self.plan_horizon)
        # 使 mean 的 shape 变成 (plan_horizon, )，对应未来 plan_horizon 步的动作均值。
        # print(mean) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

        var = np.tile(np.square(self.upper_bound - self.lower_bound) / 16, self.plan_horizon)
        # print(var) # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

        obs, _ = self._env.reset()
        done = False
        episode_return = 0
        # 进入虚拟环境
        while not done:
            # 用 CEM 选择未来 plan_horizon 步的最优动作序列，但只执行第一步
            actions = self._cem.optimize(obs, mean, var) # 传入 Fake_env 计算 return 返回一个 动作序列
            action = actions[:self._action_dim]  # 选取第一个动作
            next_obs, reward, done, truncated, _ = self._env.step(action) # 把 动作 传入真实环境
            self._env_pool.add(obs, action, reward, next_obs, done) # 添加数据到经验池
            obs = next_obs
            episode_return += reward
            mean = np.concatenate([
                np.copy(actions)[self._action_dim:],  # 舍弃当前执行的第一步，保留后面的动作
                np.zeros(self._action_dim)  # 用 0 填充最后一步
            ])
            if truncated:
                break
        return episode_return

    # 用于训练之前，先随机探索一条序列
    def explore(self):
        obs, _ = self._env.reset()
        done = False
        episode_return = 0
        while not done:
            action = self._env.action_space.sample()
            next_obs, reward, done, truncated, _ = self._env.step(action)
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            if truncated:
                break
        return episode_return

    # 基于 PETS 算法开始训练并记录值
    def train(self):
        return_list = []
        explore_return = self.explore()  # 先进行随机策略的探索来收集一条序列的数据
        print('episode: initial, return: %d' % explore_return)
        return_list.append(explore_return)

        # 正式进入训练
        for i_episode in range(self.num_episodes):
            self.train_model() # 训练模型
            episode_return = self.mpc() # 基于 CEM 与真实 env 交互直到 done，同时记录数据
            return_list.append(episode_return)
            print('episode: %d, return: %d' % (i_episode + 1, episode_return))
        return return_list

buffer_size = 100000 # 最大经验池数据量
n_sequence = 50 + 25 # 采样的动作序列条数
elite_ratio = 0.2 # 保留的精英序列比例
plan_horizon = 25 - 5 # 规划未来的步长数
num_episodes = 10 + 10 # 迭代次数
env_name = 'Pendulum-v1' # 状态空间 3 动作空间 1
env = gym.make(env_name)

replay_buffer = ReplayBuffer(buffer_size)
pets = PETS(env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes)
return_list = pets.train()

# 用所得 return 绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PETS on {}'.format(env_name))
plt.show()

# 该算法的核心思想是 用一个模型（Fake Env）来拟合真实环境的状态转移
# 即输入 (s, a) 后预测 (r, s')
# 这样，就可以在不与真实环境频繁交互的情况下，基于模型进行采样和优化)
#
# 具体流程如下：
# 1. 训练环境模型：收集 (s, a, r, s') 真实交互数据，不断训练 Fake Env，使其逼近真实环境的状态转移
# 2. 基于模型规划动作序列：
#       在 Fake Env 中采样多个动作序列，计算对应的累计奖励（return）
#       选取 return 最高的动作序列，作为最优规划
# 3. 执行最佳动作并更新数据：
#       只执行当前最优序列的第一步动作
#       真实环境交互得到新的 (s', r)，将其存入经验池，用于继续训练 Fake Env
# 4. 重复上述过程，让模型不断优化，同时在真实环境中收集更多高质量数据，使策略逐渐提升
#
# 本质上是基于模型的强化学习（Model-Based RL），通过不断迭代改进环境模型，实现更高效的决策