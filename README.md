# PerfectMahjongCPP — 国标麻将 (MCR) 开天眼强化学习 AI 引擎

> 参考同级项目 [`PerfectDouCPP`](file:///Users/mingming/project/PerfectDouCPP) 的 **完美信息蒸馏（Perfect Information Distillation）+ 同步 PPO 自博弈 + 向听数先知（Shanten Oracle）** 架构，使用纯 C++20 与 LibTorch 从零实现的 **中国国家体育总局《中国麻将竞赛规则》（国标麻将 MCR，81 番种，8 番起和）** 强化学习训练与评测系统，并内置与预训练 **Botzone 国标麻将强基线模型（`TokenTransformer`）** 的 TCP 复式对战评测服务。

---

## 一、主流麻将玩法深度调研（腾讯游戏 vs 世界麻将大赛）

在立项阶段，我们对全球与国内最具代表性的三大竞技与主流麻将规则体系进行了对比调研：

| 维度 | **国标麻将 (MCR)** | **日本立直麻将 (Riichi)** | **腾讯欢乐麻将 — 四川血战到底** |
| :--- | :--- | :--- | :--- |
| **权威赛事 / 平台** | 世界麻将锦标赛 (WMC)、Botzone、腾讯国标 | 世界麻将大赛 (WRC)、雀魂、天凤 | 腾讯《欢乐麻将》日活最高场次 |
| **牌张总数** | 136 张（万/筒/条/风/箭，本引擎采用 Botzone 标准无花 136 张） | 136 张（万/筒/条/风/三元，含赤宝牌） | 108 张（仅万/筒/条，无字牌） |
| **副露规则** | **可吃、可碰、可明杠/暗杠/加杠** | 可吃、可碰、可杠（吃碰破坏门前清立直） | **不可吃**，仅可碰、杠；开局定缺一门 |
| **和牌门槛** | **8 番起和**（81 种番型组合，严禁屁和） | **1 番（役）起和** + 振听限制 | **0 番起和**，一家胡牌后其余玩家继续至 3 家胡完 |
| **计分结算** | 零和博弈：自摸三家各付 `8 + F`；点炮者付 `8 + F`，另两家各付 `8` | 符数 × 番数指数计分 + 立直棒/本场棒 | 番数翻倍（`2^n`）+ 刮风下雨（杠分即时结算） |
| **开源强基线生态** | **`PyMahjongGB` / `mahjong-algorithm` + Botzone 预训练模型** | Mortal / Suphx / Kanachan | 缺乏统一公开的强基线模型权重 |

**本项目定位**：直接聚焦 **国标麻将（Chinese Standard Mahjong, MCR）**，将 81 番种的组合规划、8 番起和约束、四家副露（吃/碰/杠）与出牌序列推理全部纳入纯 C++ 高速仿真与特征工程，并支持与 **Botzone 515 万局人类/强 Bot 对局训练出的 `TokenTransformer` 模型** 进行四座轮换复式评测（Duplicate Evaluation）。

---

## 二、系统架构与目录结构

```text
PerfectMahjongCPP/
├── CMakeLists.txt                  # CMake 构建配置（自动关联 LibTorch 与 GoogleTest）
├── baselines/
│   └── botzone_mcr/
│       ├── botzone_model_bc.pt     # Botzone 5.15M 样本预训练 TokenTransformer 强基线模型
│       └── botzone_model.pt        # PPO 微调版 TokenTransformer 模型
├── src/
│   ├── mahjong_algo/               # 官方 C++ 81 番种算番器与标准/七对/十三幺向听数计算器
│   │   ├── fan_calculator.h / .cpp
│   │   ├── shanten.h / .cpp
│   │   └── stringify.h / .cpp
│   ├── mcr/                        # 国标麻将 (MCR) C++ 核心状态机与特征工程
│   │   ├── tiles.h / .cpp          # 34 种牌编码、副露 Meld、动作空间 Action (1974 唯一抽象 ID < 2048)
│   │   ├── oracle.h / .cpp         # 精确向听数、81 番算番器封装、8 番起和校验、潜在价值函数 Phi(s)
│   │   ├── game.h / .cpp           # 4 人 MCR 状态机（摸/打/吃/碰/杠/抢杠和/8番裁决）与内置 C++ 启发式 AI
│   │   └── features.h / .cpp       # 4 家手牌数 + 4 家副露 + 24 步 LSTM 出牌序列 + 开天眼 Critic 特征
│   ├── algo/                       # 强化学习算法核心
│   │   ├── ppo.h / .cpp            # 多玩家独立轨迹 GAE (trajectoryId = gameId * 4 + seat) 与掩码 Softmax
│   │   ├── rollout.h / .cpp        # 多线程 C++ 锁步批量自博弈采样 + 历史快照池/启发式联赛 + 先知奖励塑形
│   │   └── eval_mcr.h / .cpp       # 四座轮换复式评测器（支持 TCP 连接 Botzone MCR 模型或 C++ 启发式基线）
│   ├── torch_nn/                   # LibTorch 神经网络模块
│   │   ├── models.h / .cpp         # Actor（不完美信息 + 候选动作交互）与 Critic（开天眼全局特征）
│   │   ├── ppo_learner.h / .cpp    # PPO 裁剪目标、解耦梯度范数裁剪、KL 早停与优化器序列化
│   │   └── tensor_bridge.h / .cpp  # C++ Transition 到 LibTorch Batch Tensor 的零拷贝桥接
│   ├── train.cpp                   # 训练主程序 perfectmahjong_train
│   └── eval_mcr.cpp                # 独立评测程序 perfectmahjong_eval
├── tools/
│   ├── mcr_serve.py                # 独立自包含的 Botzone MCR TokenTransformer TCP 评测服务端
│   ├── run_mcr_serve.py            # 一键启动 Botzone MCR TCP 评测服务
│   └── plot_eval.py                # 评测曲线（WP / ADP / 点炮率）实时绘图脚本
└── tests/                          # GoogleTest 单元测试套件
    ├── test_rules.cpp              # 牌编码、81 番算番（大四喜/十三幺/七对/<8番拒和）、状态机与特征测试
    ├── test_algo.cpp               # 掩码 Softmax、多玩家 GAE、多线程 Rollout 与复式评测测试
    └── test_torch_nn.cpp           # LibTorch Actor/Critic 前向反向传播与 Checkpoint 存取测试
```

---

## 三、核心特征工程：四家牌数 + 出牌序列 + 吃碰杠副露 + 开天眼指导

在 [`src/mcr/features.h`](file:///Users/mingming/project/PerfectMahjongCPP/src/mcr/features.h) 与 [`src/mcr/features.cpp`](file:///Users/mingming/project/PerfectMahjongCPP/src/mcr/features.cpp) 中，状态编码严格对齐用户的核心特征需求：

### 1. Actor 不完美信息特征（`1520` 维静态特征 + `24 × 88` 维 LSTM 序列 + `96` 维候选动作特征）
1. **自家暗手牌与刚摸到的牌（Ego Concealed Hand & Drawn Tile）**：
   - **自家暗手牌 `4 × 34 = 136` 维温度计编码（Channels `0..3`）**：对 34 种牌分别用 4 个二进制通道编码自家手里持有 `≥1, ≥2, ≥3, =4` 张（见 [`features.cpp`](file:///Users/mingming/project/PerfectMahjongCPP/src/mcr/features.cpp#L39-L40)）；
   - **刚摸到的牌 `34` 维 One-Hot（Channel `4`）**：编码当前回合刚从牌墙摸入的单张牌（[`features.cpp`](file:///Users/mingming/project/PerfectMahjongCPP/src/mcr/features.cpp#L42-L45)）；
   - **自家听牌与向听数先知特征**：在Channel `42`（`34` 维）与 `scalar[37..43]` 中直接编码自家当前的标准/七对/十三幺向听数、有效进张数、8 番以上听牌位图与最大可和番数。
2. **四家手牌张数（4-Player Concealed Hand Counts）**：
   - 在 `scalar[0..3]` 中按相对座次（自家 `0`、下家 `1`、对家 `2`、上家 `3`）精确编码四家当前暗手牌张数 `handCount[p] / 14.0f`，结合副露杠牌数量实时反映各家手牌紧迫度。
2. **四家副露：吃、碰、明杠、暗杠、加杠（4-Player Melds）**：
   - **空间位图矩阵（16 × 34 = 544 维）**：为四家分别编码「顺子吃牌组、`碰牌组`、`明杠/加杠组`、`暗杠组`」的 34 维牌面分布；
   - **副露统计标量（28 维）**：四家各自的副露总数、吃数、碰数、明杠数、暗杠数、是否门前清（Concealed Flag），以及**喂牌/供牌关系矩阵（Feeding Count）**（记录哪一家给哪一家吃碰了多少次，用于防范国标麻将中的连带包牌风险与染色牌型推断）。
3. **四家弃牌河与 24 步 LSTM 出牌/副露动作序列（4-Player Discard Rivers & Sequence）**：
   - **弃牌河矩阵（16 × 34 = 544 维）**：四家弃牌河的 0..4 张计数热独编码；
   - **LSTM 时序流（`24 × 88` 维）**：最近 24 步公开动作（打牌、吃、碰、明杠、暗杠、加杠），每步 88 维包含：动作玩家相对座次（4 维）、动作类型（6 维）、目标牌 34 维 One-Hot、吃碰杠后打出的弃牌 34 维 One-Hot、手切/摸切标志（`tsumogiri`，1 维）、该牌是否被他家吃碰截走（`claimed`，1 维）、执行动作后**四家最新的暗手牌张数**（4 维）及剩余牌墙比例。
4. **候选动作后验特征（`96` 维 Action Features）**：
   - 对每个合法候选动作（打某张牌、吃、碰、杠、和、过），直接调用 [`simulateActionOnHand`](file:///Users/mingming/project/PerfectMahjongCPP/src/mcr/game.cpp) 推演动作后的手牌与副露，计算**动作后的精确向听数、进张数（Ukeire）、8 番以上听牌数、打该张牌的点炮危险度估计（[`estimateDealInRisk`](file:///Users/mingming/project/PerfectMahjongCPP/src/mcr/game.cpp)）**，与 Actor 隐层做分解式交互打分。

### 2. Critic「开天眼」完美信息特征（额外 `612` 维 Omniscient Oracle Features）
仅在训练阶段输入给 Critic 价值网络（推理与评测时完全移除）：
- **三家对手的真实暗手牌（`12 × 34 = 408` 维）**：下家、对家、上家暗手牌的精确张数位图；
- **牌墙未来 12 张摸牌透视（`4 × 34 = 136` 维）**：牌墙顶未来 12 张即将摸到的牌；
- **三家对手的真实 8 番听牌/点炮危险牌位图（`34` 维）**：直接通过算番器枚举三家对手所有能达到 $\ge 8$ 番的荣和銃牌；
- **四家真实向听数、真实进张数、最大可和番数与有效打点（`34` 维标量）**。

### 3. 向听数与算番先知奖励塑形（Shanten Oracle Reward Shaping）
在 [`oraclePotentialAdvantage`](file:///Users/mingming/project/PerfectMahjongCPP/src/mcr/features.cpp) 中，定义全知状态势函数：
$$\Phi(s, i) = \text{Potential}(s, i) - \frac{1}{3}\sum_{j \neq i} \text{Potential}(s, j)$$
每步奖励加入势差项 $r_t^{\text{oracle}} = \beta \cdot \text{clip}(\Phi(s_{t+1}, i) - \Phi(s_t, i), -1, 1)$，在不改变最优策略不动点的前提下，解决国标麻将 8 番起和导致的早期稀疏奖励难题。

---

## 四、快速开始：编译、测试、训练与 Botzone 强模型评测

### 1. 编译与运行单元测试
```bash
cmake -B build
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

### 2. 启动 Botzone MCR 预训练强模型 TCP 评测服务
我们使用 `/Users/mingming/.venvs/perfectdou-cpp/bin/python` 运行自包含的 [`tools/run_mcr_serve.py`](file:///Users/mingming/project/PerfectMahjongCPP/tools/run_mcr_serve.py)，加载 [`baselines/botzone_mcr/botzone_model_bc.pt`](file:///Users/mingming/project/PerfectMahjongCPP/baselines/botzone_mcr/botzone_model_bc.pt)（515 万局官方 Botzone 国标麻将对局训练的 6 层 `TokenTransformer`）：
```bash
~/.venvs/perfectdou-cpp/bin/python tools/run_mcr_serve.py 18766
```

### 3. 启动同步 PPO + 开天眼训练（自动连接 TCP 评测服务）
```bash
./build/perfectmahjong_train \
    --backend mps \
    --updates 2000 \
    --games 128 \
    --threads 8 \
    --eval-every 50 \
    --eval-decks 25 \
    --eval-mode tcp \
    --eval-port 18766 \
    --out ckpt_mcr
```
- `--backend`: 支持 `mps`（Apple Silicon GPU）、`cuda`（NVIDIA GPU）或 `cpu`；
- `--eval-mode`: 支持 `tcp`（与 `mcr_serve.py` 中的 Botzone `TokenTransformer` 进行四座轮换复式对战）、`heuristic`（与内置 C++ 启发式 AI 对战）或 `auto`（优先连接 TCP，未启动时自动回退至 `heuristic`）。

### 4. 独立评测已保存的模型权重
```bash
./build/perfectmahjong_eval \
    --models ckpt_mcr \
    --decks 25 \
    --mode tcp \
    --port 18766
```
> 注：每副测试牌墙（`deck`）自动执行 **4 局复式轮换（Duplicate Rotation）**，我方模型依次坐东、南、西、北四个座位对战 3 个基线模型，彻底消除麻将发牌运气与座次偏差。
