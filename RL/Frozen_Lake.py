import gym

env = gym.make("FrozenLake-v1", render_mode="rgb_array")  # 创建环境

# holes = set()
# ends = set()
#
# Gym 的 FrozenLake 环境存储状态转移信息的方式是：
# env.P[state][action] = [(prob, next_state, reward, done)]
#
# for s in env.P:
#     for a in env.P[s]:
#         for s_ in env.P[s][a]:
#             if s_[2] == 1.0:  # 获得奖励为1,代表是目标
#                 ends.add(s_[1])
#             if s_[3] == True:
#                 holes.add(s_[1])
#
# holes = holes - ends
# print("冰洞的索引:", holes)
# print("目标的索引:", ends)
#
# for a in env.P[14]:  # 查看目标左边一格的状态转移信息
#     print(env.P[14][a])

from DP import PolicyIteration, ValueIteration, print_agent

# 这个动作意义是Gym库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])