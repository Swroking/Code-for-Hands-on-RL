# 📘 项目环境说明（适用于[《动手学强化学习》](https://hrl.boyuai.com/chapter/)）教程学习
---

## 📦 安装依赖

本项目主要依赖以下 Python 包：

- `numpy`
- `matplotlib`
- `gym`（需 **版本 ≤ 0.20.0**，因兼容 `ma-gym`）
- `torch`
- `torchvision`
- `torchaudio`
- `pygame`：用于图形化显示训练环境
- `tqdm`：显示训练进度条
- `jupyter`：支持 Notebook 学习环境
- `ma-gym`：用于多智能体环境的研究，依赖特定版本 `gym`
- `SMAC`：星际争霸Ⅱ环境

---

## ⚠️ gym 版本说明与冲突处理

由于 `ma-gym` 对 `gym` 有版本限制（要求 ≤ 0.20.0），在学习 MPC（Model Predictive Control）部分时将 `gym` 降级，导致前面章节（使用新版 `gym` 编写）部分代码可能报错。

### 🔧 建议的解决方案：

#### 方案一：统一使用 **旧版 gym（≤0.20.0）**
- 降低 `gym` 版本：`pip install gym==0.20.0`
- 同时对原始代码进行修改以兼容旧版 API

#### 方案二：继续使用新版 `gym` 或 `gymnasium`
- 安装 `gymnasium` 并使用 `import gymnasium as gym`
- **注意**：新版 API 与旧版存在差异，需修改代码（见下）

---

## ✅ 代码修改参考（新版 gym / gymnasium）

| 原始代码 | 修改后（新版 gym） |
|----------|-------------------|
| `env.seed(0)` | `env.reset(seed=0)` |
| `state = env.reset()` | `state, _ = env.reset()` |
| `next_state, reward, done, _ = env.step(action)` | `next_state, reward, done, truncated, _ = env.step(action)` |
| `if done: break` | `if done or truncated: break` |

### 📌 `done` 与 `truncated` 的区别

- `done`: 环境达到终止条件（如 pole 倒下）
- `truncated`: 达到最大步数（如 `CartPole-v0` 是 200 步，`v1` 是 500 步）

---

## 🧩 环境渲染说明

在部分 `gym` 版本中，`render()` 行为可能有变化。若图形窗口无法正常显示，可尝试如下写法：

```python
env = gym.make('CartPole-v1', render_mode='human')
```

---

## 📁 [项目](https://pan.baidu.com/s/1aHkau6WxLhTGZLzEqKVHfw?pwd=rksq)目录结构

忙活半天还是没能上传GitHub，项目文件放网盘里了🫠项目的目录结构如下：

### 🧠 强化学习基础

* `D:\Myproject\StudOnep\RL`
  👉 单智能体强化学习相关实现

### 🤖 多智能体强化学习

* `D:\Myproject\StudOnep\ma-gym`
  👉 基于 `ma-gym` 环境的 IPPO 算法实现

* `D:\Myproject\StudOnep\multiagent-particle-envs`
  👉 使用粒子环境实现 `MADDPG` 算法

* `D:\Myproject\StudOnep\multi-agent-PPO-on-SMAC-main`
  👉 IPPO 和 MAPPO 在 SMAC 环境上的简单实现

* `D:\Myproject\StudOnep\Mappo-Integration`
  👉 整合后的 MAPPO 实现，并基于 3m 地图完成训练与结果保存

### 📚 教程学习与代码实践

* `D:\Myproject\StudOnep\torch_morvan`
  👉 Morvan Torch 教程相关算法实现（如 `RNN`）

* `D:\Myproject\StudOnep\plot`
  👉 Morvan Matplotlib 教程实践代码

* `D:\Myproject\StudOnep\extra_test`
  👉 Hands-on RL 学习过程中的代码理解与实践

* `D:\Myproject\StudOnep\some_new`
  👉 一些算法补充，详见`some_new.md`文件

---

## 📚 学习资料推荐

为了更好地理解本项目涉及的代码与算法，建议结合以下学习资源一并参考：

### ▶️ Morvan 的教学视频

* 地址：[https://mofanpy.com/learning-steps/](https://mofanpy.com/learning-steps/)
* 主要关注以下内容：

  * Python 基础语法
  * Numpy 数组操作
  * Matplotlib 可视化
  * PyTorch 神经网络
  * 强化学习（RL）[原理](https://www.bilibili.com/video/BV1sd4y167NS/)与实践

---

### 📖 RL 相关博客

* 📁 文件名：`favorites_2025_6_5.html`
* 📌 说明：一些关于强化学习、numpy、plt、PyTorch 的博客和资料以及 Gym、SMAC、[D4RL](https://zhuanlan.zhihu.com/p/11007245238) 的环境配置等，更多环境如 [Atrai、Mujoco and Box2d](https://zhuanlan.zhihu.com/p/667403508) 及相关项目可以参考 `Atrai_Mujoco.zip`
* 🛠 使用方式：

  1. 打开 Edge 浏览器
  2. 进入书签设置 → 导入书签
  3. 加载 `favorites_2025_6_5.html` 文件，即可查看收藏链接

---

### 🔗 建议学习顺序

1. **基础入门**（Python + Numpy + Matplotlib + PyTorch）
2. **强化学习基本概念与单智能体算法**
4. **多智能体强化学习框架（如 `ma-gym`, `SMAC`）**

---

🎓 **祝学习顺利！** 🚀

---
