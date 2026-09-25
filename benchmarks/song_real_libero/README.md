# WEP-VLA：基于软运动先验、点云与 SmolVLA 的真实机器人 / LIBERO 策略

> 项目主文档与快速回顾手册
>
> 文档基线：`wep_vla_v0.4.2`，commit `9fb900c`，2026-07-28
> 如果后续代码行为与本文冲突，以当前代码和 checkpoint 内的 `config.json` 为准，并同步更新本节基线。

## 摘要

WEP-VLA 是建立在 LeRobot SmolVLA 上的视觉—语言—点云动作策略。模型使用冻结或可训练的
SmolVLM 视觉语言前缀、由运动轨迹自动构造的 PointSeg 软监督、LitePT 几何编码器，以及
PointACT 风格的点—动作 token 交互，最终由 Action Expert 通过 flow matching 生成一段
EEF 相对动作轨迹。

项目同时支持真实机器人 HDF5 与 LIBERO 官方 demonstration。两个数据域共享以下核心契约：

- 固定 overview 相机是观测参考系；真实数据不依赖相机到机器人世界的外参。
- 点云在加入虚拟夹爪后转换到当前 EEF 坐标系，再作为模型输入。
- 动作采用 `xyz + rotation-6D + gripper` 的 10 维表示。
- PointSeg 的监督来自机械臂轨迹先验，不使用人工分割、人工光流、手工目标点或 PCA。
- 默认稳定模型不启用 WorldFlow 和额外 SE(3) 动作分支。

本文既是实现说明，也是后续开发时快速恢复上下文的唯一主入口。历史实验审计可继续参考
[docs/experiment-records/LIBERO_EVALUATION_AUDIT.md](docs/experiment-records/LIBERO_EVALUATION_AUDIT.md)，VLM adapter 的旧版专项说明见
[docs/local-notes/README_VLM_ADAPTER.md](docs/local-notes/README_VLM_ADAPTER.md)，但涉及当前行为时以本文为准。

---

## 1. 项目快速回顾

### 1.1 当前稳定配置

| 模块                   | 当前建议 | 说明                                                      |
| ---------------------- | -------: | --------------------------------------------------------- |
| SmolVLM RGB 前缀       |     开启 | adapter 模式加载官方预训练 VLM                            |
| 冻结 VLM               |     开启 | 保留官方图像—语言表征，训练点云与动作相关模块             |
| PointSeg               |     开启 | 使用 cache v7 软运动先验监督                              |
| LitePT                 |     开启 | XYZ 作为 coord，同时 XYZRGB 作为 feat                     |
| Point–Action fusion    |     开启 | 最终 LitePT token 与 action token 联合双向 self-attention |
| WorldFlow              | **关闭** | 当前 cache 通道语义尚未与旧 WorldFlow 角色解释完全统一    |
| 独立 SE(3) flow 动作头 | **关闭** | 实验分支，不是当前稳定结果路径                            |
| PCA / canonical frame  |   不使用 | 已从当前方案中排除                                        |
| 人工分割 / 人工点流    |   不使用 | 所有前景监督由轨迹自动产生                                |

对应核心参数：

```bash
--policy.vla_adapter_enable=true
--policy.vla_adapter_freeze_vlm=true
--policy.pointseg_enable=true
--policy.point_action_fusion_enable=true
--policy.worldflow_enable=false
--policy.worldflow_se3_head_enable=false
--policy.se3_enable=false
--policy.se3_final_correction_enable=false
```

其中 `worldflow_se3_head_enable` 与 `se3_final_correction_enable` 仅为旧命令兼容参数，
当前配置类会提示它们被忽略。

当前模型的关键结构默认值：

| 配置                          |           值 |
| ----------------------------- | -----------: |
| observation history           |            1 |
| action chunk / 默认执行长度   |      32 / 16 |
| state / action 最大维度       |      10 / 10 |
| flow denoising steps          |           10 |
| VLM 保留层数                  |           16 |
| Action Expert 宽度倍率        |         0.75 |
| VLM/Expert attention mode     | `cross_attn` |
| self-attention 插入间隔       |      每 2 层 |
| PointSeg / LitePT feature dim |           64 |
| 前景 LitePT token 上限        |          256 |
| Point–Action attention heads  |            4 |
| robot state prefix            |     默认关闭 |

### 1.2 三条不能破坏的语义契约

1. **坐标契约**：存入 LeRobot dataset 的 `observation.point_cloud` 已经位于当前 EEF
   坐标系；训练和推理必须保持一致。
2. **LIBERO 标签契约**：`observation.state` 是执行后的实际 EEF 位姿，`action` 是原始
   delta OSC action 经控制器重建出的绝对命令目标。两者不应被强制设为相同。
3. **cache 绑定契约**：PointSeg cache 通过点索引引用 dataset 中的点。只要点云内容、
   点顺序、坐标系、episode/frame 映射或采样逻辑变化，就必须重新生成 cache。

### 1.3 最常用入口

| 任务                        | 权威脚本                                                 |
| --------------------------- | -------------------------------------------------------- |
| LIBERO HDF5 → LeRobot       | `scripts/libero_setting/libero_hdf5_to_dataset.py`       |
| 真实 HDF5 → LeRobot         | `scripts/real_setting/real_hdf5_to_dataset.py`           |
| 离线生成 PointSeg cache     | `scripts/song_cache_pointseg_samples.py`                 |
| 项目标准训练                | `scripts/train_song_benchmark.py`                        |
| 只读离线 loss 评测          | `../../src/lerobot/scripts/eval_song.py`                 |
| LIBERO rollout 评测         | `scripts/libero_setting/libero_pointcloud_eval.py`       |
| dataset / pickle / PLY 推理 | `scripts/smolvla_model_inference.py`                     |
| 核心模型                    | `../../src/lerobot/policies/smolvla/modeling_smolvla.py` |
| PointSeg 先验               | `../../src/lerobot/policies/smolvla/song_pointseg.py`    |
| UMI 坐标处理                | `../../src/lerobot/processor/umi_processor.py`           |

历史备份文件位于仓库外的 `useless/source-archives/`，不是当前运行入口，不能据此判断现有模型行为。

---

## 2. 研究目标与设计边界

### 2.1 目标

模型需要从 overview RGB、场景点云、语言任务和当前机器人状态中生成可执行动作，同时尽量：

- 聚焦被操作物体、工具、目标接触区域等任务相关几何；
- 降低机械臂外观、背景变化和异构末端对策略的干扰；
- 使用物体间相对几何，而不是只记住固定轨迹；
- 在真实数据和 LIBERO 中使用同一套模型接口；
- 不依赖真实相机外参和人工对象标签。

### 2.2 当前没有声称解决的问题

- 当前 LitePT 把绝对 XYZ 同时输入 coord 和 feat，仍可能对坐标平移、相机重定位和离群点敏感。
- 当前真实与 LIBERO 数据都把 overview 相机视为固定参考。相机随头部运动时，必须实时获得
  EEF 与相机在同一坐标系中的位姿，否则点云和机器人状态会失配。
- 运动先验是软前景证据，不等价于实例分割或精确场景流。
- 低 flow-matching loss 不保证接触任务成功；闭环误差、控制器语义、数据对齐和动作执行频率
  都可能成为主要瓶颈。

---

## 3. 总体系统

```text
fixed overview RGB ───────────────┐
language task ────────────────────┼─> SmolVLM prefix
                                 │       │
EEF-frame XYZRGB point cloud ─> PointSeg│
                │                 │       │
                ├─ foreground ─> LitePT ─┼─> point prefix tokens
                └─ background ─> pooling │
                                         │
noisy action chunk + time ─> action tokens
                    │                    │
                    └─ final LitePT tokens
                         │
                  Point–Action self-attention
                         │
                  Action Expert + VLM
                         │
                  predicted flow velocity
                         │
                 Euler denoising, 10 steps
                         │
              EEF-relative pose9 + gripper
```

模型存在两条互补的点云信息路径：

- **全局前缀路径**：前景和背景点云各压缩为一个 token，与 RGB、语言共同构成 VLM prefix。
- **细粒度动作路径**：最终 LitePT point tokens 与整个 action chunk token 联合 self-attention，
  让动作在进入 Action Expert 前直接读取局部几何。

第二条路径受 PointACT 思想启发，但它是单层、末端融合模块，不是 PointACT 全部多尺度架构的复现。

---

## 4. 数据与坐标表示

### 4.1 模型输入

#### 点云

```text
observation.point_cloud: (B, N, 6)
channels: [x, y, z, r, g, b]
XYZ: metre
RGB: 0 ... 255
frame: current EEF
```

`observation.point_cloud_is_pad` 为 `True` 的位置是 padding。原始 episode 可以具有不同点数，
DataLoader 会补齐；PointSeg 的候选点选择会进一步构造固定数量的前景和背景点。

#### 图像

dataset 中通常保存 256×256 的 `agentview` 或 `overhead` RGB。策略的官方 SmolVLM processor
会将其 resize/pad 到 512×512，并转换到模型所需数值范围。因此“采集 256×256，VLM 输入
512×512”是有意设计，不是训练—推理尺寸错误。

#### 语言

每帧关联一个 task string。即使真实训练只有单任务，语言 token 仍进入 prefix；单任务条件下
语言分支信息量有限，但不会使点云分支失效。

#### 状态和动作

```text
[x, y, z, r6_0, r6_1, r6_2, r6_3, r6_4, r6_5, gripper]
```

默认 `max_state_dim=max_action_dim=10`，action chunk 长度为 32，部署时通常执行前 16 个或由
评测参数指定的若干动作。

### 4.2 rotation-6D

6D 旋转不是四元数，也不要求每个数位于 `[-1, 1]`。模型输出两条三维向量
\(\mathbf{a}_1,\mathbf{a}_2\)，解码时执行：

\[
\mathbf{b}_1 = \operatorname{normalize}(\mathbf{a}_1)
\]

\[
\mathbf{b}_2 = \operatorname{normalize}
\left(\mathbf{a}_2-(\mathbf{b}_1^\top\mathbf{a}_2)\mathbf{b}_1\right)
\]

\[
\mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2,\qquad
R=[\mathbf{b}_1,\mathbf{b}_2,\mathbf{b}_3].
\]

所以类似 `[0.008, -1.002, -0.015, 1.000, 0.011, 0.023]` 的值可以解码为合法旋转。
单位旋转的 6D 表示为 `[1, 0, 0, 0, 1, 0]`。

### 4.3 UMI 当前末端坐标系

训练处理器以当前真实观测 `observation.state` 为原点，将动作转换到当前 EEF 坐标系。不能再用
`action[0]` 充当观测原点，否则会人为抹掉第一条命令目标，使首个动作错误地接近单位位姿。

当前 UMI 语义是：

- 点云：转换脚本或在线推理包装器已变换到当前 EEF；
- 状态：当前 EEF 在自身坐标系中接近单位位姿；
- 动作：相对于当前实际 EEF 的未来命令目标；
- 推理输出：同样的 EEF-relative 目标，再由执行器还原为控制器目标。

---

## 5. PointSeg：基于轨迹的软前景学习

### 5.1 为什么不用硬标签

机器人任务中的“前景”随阶段变化：抓取前，杯子和杯把重要；搬运时，杯子与夹爪形成共同刚体；
挂杯末端，杯架又必须持续重要。固定实例 mask 或一次性二分类无法完整表达这种关系。

当前先验只输出软概率与置信度：

- 跟随末端共同运动的点获得工具/被操作物体证据；
- 末端沿轨迹逐渐接近的点获得目标证据；
- 即使局部阶段速度很小，只要长期处于接触邻域，仍保留前景概率；
- 远离整段交互轨迹且缺乏其他证据的点获得背景置信度。

### 5.2 双向轨迹窗口

默认时间偏移为：

```text
0, -31, -16, -8, -4, -2, -1, +1, +2, +4, +8, +16, +31
```

它既观察过去也观察未来，适用于离线 cache 和训练时在线计算。由此避免只看单向未来时，在任务
末端失去已接触物体或目标结构。

轨迹优先来自实际 `observation.state`。只有旧数据集无法区分状态与动作时，才回退到 action。

### 5.3 八类几何先验

当前 cache 为每个采样点计算：

1. 当前点到 EEF 的距离；
2. 点到整段 EEF 轨迹的最小距离；
3. 沿时间窗口的接近程度；
4. 与末端共同运动的刚体残差；
5. 静态场景假设下的残差；
6. 共同运动与静态残差之间的差距；
7. 在轨迹邻域中的停留 / 接触持续性；
8. 该点在时间窗口中的可观测上下文。

这些量组合为三个软证据通道：

```text
tool_comotion
trajectory_approach
near_contact
```

最终前景概率使用 soft probabilistic-OR 聚合，而不是阈值硬拼接。背景置信度由“远离轨迹且
前景证据低”产生。训练标签本身保持 ignore，真正监督来自 soft class score 与 sample weight。

### 5.4 PointSeg 网络与损失

`SongPointSegNet` 只把当前帧原始 XYZRGB 作为网络输入。时间窗口先验默认仅用于构建监督，
不会在推理时成为额外输入，所以在线部署只需一帧点云。

PointSeg loss 主要由：

- soft binary cross entropy；
- 体素邻域平滑项；
- 有效点与先验置信度加权；

构成。当前只区分 foreground / background，不要求模型预测“杯子、杯架、夹爪”等人工角色。

`pointseg_use_temporal_priors_as_input` 和 `pointseg_use_pseudo_selection` 目前仅保留在配置与
历史命令中，现行主路径没有读取它们来改变网络输入或选点。它们设为 `false` 可表达当前实验
意图，但不要把效果差异归因于这两个参数。

### 5.5 前景和背景选点

PointSeg 输出每点前景概率。模型随后：

- 前景：按预测分数从高到低选择；
- 背景：优先选择 `score <= 0.5` 且分数最低的点；
- 数量：`max(min_points, ceil(max_valid_points_in_batch × ratio))`；
- 候选不足：循环重复已有候选，形成固定长度张量；
- 无效点：由 pad mask 排除。

因此，若一帧只有 3,072 点而请求 4,000 个前景，模型会保留候选并重复补齐，而不是凭空生成新点。
cache 的 `current-points` / `future-points` 不同：若源点数小于上限，cache 保留全部点，不重复。

---

## 6. LitePT 与点—动作交互

### 6.1 LitePT 当前特征输入

当前 `LitePTTokenizer` 将：

```text
coord = XYZ
feat  = [XYZ, RGB / 255]
```

送入 LitePT。也就是说 XYZ 被使用两次：

- 作为稀疏体素、邻域和序列化所需的空间坐标；
- 作为可学习 feature 的一部分，显式保留几何位置和尺度。

实验证明仅用 RGB 作为 feat 会显著削弱几何目标关注和真实机械臂零样本迁移，因此当前保留
XYZRGB feat。代价是模型不具备严格平移 / SE(3) 不变性；相机或坐标原点变化会改变 feature
分布。这是当前明确的鲁棒性限制。

### 6.2 已移除的 center 二次编码

`LitePTEncoder` 仍实例化了两个 backbone 以兼容历史 checkpoint，但现行 forward 只使用第一个：

```text
LitePT -> self-attention -> scalar alpha -> weighted global pooling
```

旧实现曾根据第一阶段 alpha 预测 center，把点云减去 center 后再次送入第二个 backbone。该路径
对很小的 center 偏差过度敏感：即使 alpha 关注对象几乎不变，XYZ 作为 feat 的整体偏移仍会被
后续 attention 放大，导致 global feature 和动作显著改变。因此该二次 center 路径已经注释停用。

不要简单使用点云几何均值替代预测 center。几何均值会受到背景范围、遮挡和离群点支配，也不能
代表操作对象中心。

### 6.3 Alpha 的含义

LitePT 内部 alpha 是特征聚合注意力，不是 PointSeg 标签。它可以自动聚焦任务细节，但当前不会
反向修改离线先验，也不应被直接当成硬前景监督。两者职责不同：

- PointSeg：从运动先验学习“哪些点与操作相关”；
- LitePT alpha：在选出的前景内部学习“哪些 token 对动作编码更重要”。

### 6.4 PointACT 风格融合

最终前景 LitePT token 与 flow-matching action token 拼接：

```text
[point tokens, action tokens + sinusoidal step embedding]
                         │
              bidirectional self-attention
                         │
                  keep action part
                         │
              FFN + residual action update
```

关键性质：

- fusion 内没有 causal mask，point 与 action token 双向交互；
- action chunk 的每一步具有固定正余弦位置编码；
- `action_is_pad=True` 被正确转换为无效 token，并同时参与 key padding 与输出屏蔽；
- 没有 self / VLM / point / FFN gate；
- action-conditioned point token 不会反馈回 LitePT 的全局 pooling；
- Action Expert 之后仍会通过 RoPE 获得其内部序列位置。

这里使用正余弦位置编码而不是 RoPE，是因为该模块是独立的 PyTorch
`MultiheadAttention`，发生在 Action Expert 的 RoPE 注意力之前。两者分别解决 fusion 层和
VLM/Expert 层中的位置身份问题。

---

## 7. SmolVLM 前缀与 Action Expert

### 7.1 Prefix token

adapter 模式下，prefix 由以下 token 构成：

1. overview RGB image tokens；
2. 前景点云全局 token；
3. 背景点云全局 token；
4. task language tokens；
5. 可选 robot state token，默认配置通常关闭。

图像与语言部分保持官方 SmolVLM 架构和 processor，以便正确加载
SmolVLA 0.45B 的预训练 VLM 权重。点云 token 通过投影进入相同 hidden space。

### 7.2 Adapter 训练范围

推荐设置：

```bash
--policy.vla_adapter_enable=true
--policy.vla_adapter_freeze_vlm=true
--policy.load_vlm_weights=true
```

此时冻结视觉编码器和 VLM Transformer；PointSeg、LitePT、点云投影、Point–Action fusion
与 Action Expert 仍可训练。冻结模块不等于切断梯度：点云 token 经过冻结 VLM 运算时，梯度
仍可回传到点云投影和编码器。

当前版本没有 adapter gate。早期实验中的 `self_gate`、`vlm_gate`、`point_gate`、
`ffn_gate` 不属于现行模型。

### 7.3 Prefix 与 suffix 的注意力

Action Expert 的 suffix 是 noisy action chunk 与 diffusion time 的 embedding。注意力块关系为：

```text
prefix: [RGB, point, language, optional state]
suffix: [action_0, action_1, ..., action_31]
```

- prefix 内部双向可见；
- 每个有效 action token 可以读取全部 prefix；
- action chunk 内部相互双向可见，不是 causal attention；
- prefix 不读取 suffix；
- VLM 与 Expert 的 Q/K 通过 position id 应用 RoPE；
- 默认 `self_attn_every_n_layers=2`，在联合 / 交叉注意力结构中交替传递信息。

该设计适合一次生成完整动作块。它不是自回归 next-token action generation，因此不需要把
action chunk 强制改成 causal attention。

### 7.4 Flow matching 动作生成

训练时：

\[
\epsilon \sim \mathcal{N}(0, 0.1^2I),\qquad
t\sim\operatorname{Beta}(1.5,1)
\]

\[
x_t=(1-t)\epsilon+t a,\qquad
u_t=a-\epsilon.
\]

模型以 \(x_t,t\) 和 prefix 为条件预测速度 \(v_\theta(x_t,t,c)\)，优化：

\[
\mathcal{L}_{action}
=
\frac{\sum m\lVert v_\theta-u_t\rVert_2^2}
{\sum m},
\]

其中 \(m\) 排除 episode 尾部补齐的 action token。

推理从同分布噪声开始，默认使用 10 步 Euler 积分：

\[
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k,c),\qquad \Delta t=0.1.
\]

因此动作生成本身具有随机性。要严格比较两个入口、两次推理或 modality ablation，必须固定
相同 noise seed，并使用相同 batch 构成和数值执行路径。

---

## 8. WorldFlow 与 SE(3) 实验分支

### 8.1 当前实现

`worldflow_enable=true` 时，额外 Dense ObjectFlow 分支：

1. 向量化 top-k / gather 选择最多 2,048 个点；
2. 先在 Ego 点云中选点，再转换到固定 overview/world 参考系；
3. MLP 对每点一次输出 `chunk_size × 3` 的位移；
4. batched weighted Kabsch 通过一次 SVD 拟合每个时刻的刚体变换；
5. 使用 SE(3) 共轭关系把空间变换联系到 body action；
6. 计算 dense flow、rigid、bridge 与 equivariance 辅助损失。

该分支不使用 PCA，不要求真实点流标签，Kabsch 结果默认不反向传播。它当前是训练辅助分支，
不会直接替代默认 flow-matching 推理输出。

### 8.2 当前为什么必须默认关闭

最新版 PointSeg cache 的三个通道是：

```text
tool_comotion / trajectory_approach / near_contact
```

但 WorldFlow 的旧选点和聚合逻辑仍按：

```text
gripper / condition-manipulated-object / target
```

解释同一张量，并会抑制通道 0。两组通道语义并不等价。这会使 WorldFlow 以错误角色解释软证据，
属于功能性风险，而不是单纯超参数问题。

在重新定义角色映射、迁移 cache manifest、补充单元测试并重新训练前，论文主结果和稳定训练必须：

```bash
--policy.worldflow_enable=false
--policy.worldflow_se3_head_enable=false
```

### 8.3 其他 SE(3) 参数

- `se3_enable=true`：独立的实验性 SE(3)-twist flow 动作生成路径，不等于 WorldFlow。
- `worldflow_se3_head_enable`：历史兼容参数，当前忽略。
- `se3_final_correction_enable`：历史兼容参数，当前忽略。

---

## 9. LIBERO 数据转换：正确性优先

### 9.1 旧数据为什么会错

旧转换脚本的核心问题不是“所有 episode 看起来位置完全相同”，而是没有完整恢复每条 demo 的
环境定义。

LIBERO 官方 HDF5 为每条 demonstration 保存独立 `model_file`。其中包含柜体、把手、容器等
fixture 的 XML 布局。flattened simulator state 主要保存动态状态，不能可靠替代整份模型 XML。
旧脚本即使重置 state 后能看到不同物体位置，也可能仍沿用同一 task 环境的其他模型几何，
导致轨迹与柜体 / 把手出现毫米到厘米级错位。接触任务中，这足以让轨迹“差一点”或接触后滑走。

第二个问题是帧对齐。官方 LIBERO v1 文件中，当前观测与 state 存在固定一帧关系。实测：

- 用 `state[i+1]` 重建官方 `obs[i]`，EEF 平均位置误差约 0.304 mm；
- 用 `state[i]` 重建 `obs[i]`，平均误差约 7.48 mm。

但动作标签仍属于 step `i`，不能把所有数组一起简单平移。

第三个问题是动作语义。官方 HDF5 action 是归一化 delta OSC 命令，不是实际 EEF 位姿。将它直接
当绝对目标，或把实际 state 当 action，都会破坏控制目标。

### 9.2 当前修复

当前转换器逐 demo：

1. 恢复该 demo 自己的 `model_file`；
2. 清理并同步 controller goal history；
3. 用 `states[i+1]` 重建输出观测 `obs[i]`；
4. 仍以 `states[i]` 为 action `i` 的控制起点；
5. 调用 controller 的 `set_goal(raw_action[:6])`，保持 `use_delta=True`；
6. 读取控制器生成的绝对 target pose；
7. 保存实际状态与命令目标两种轨迹，供审计和训练分别使用。

当前字段：

| 字段                      | 语义                                         |
| ------------------------- | -------------------------------------------- |
| `observation.state`       | 重建观测时的实际 EEF pose + 实际夹爪宽度     |
| `action`                  | 控制器绝对 target pose + 对应物理夹爪宽度    |
| `world_ee_poses/`         | 实际 EEF 在固定 overview 相机参考系中的 pose |
| `action_target_ee_poses/` | 命令目标在固定 overview 相机参考系中的 pose  |

实际 state 与 action 在接触、跟踪误差或控制延迟下本来就不同。这是正确数据，不应再次对齐为同值。

### 9.3 点云与 RGB

每帧按以下顺序处理：

1. 从 `agentview` 深度反投影，在 overview 相机坐标系生成场景点云；
2. 将仿真世界中的 EEF pose 通过仿真相机外参转换到同一相机坐标系；
3. 在该坐标系添加虚拟夹爪；
4. 保持总点数为 `num_points`，默认可使用 9,500 场景点 + 500 夹爪点；
5. 将合并点云转换到当前 EEF 坐标系；
6. 保存 RGB 图像、点云和动作。

仿真外参仅用于把仿真世界 EEF 转到 overview 相机系。它不意味着真机也需要相机到机器人世界外参。

### 9.4 四套件转换命令

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /path/to/libero_demos \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --num-workers 10 \
  --num-points 10000 \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 1 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --image-camera agentview \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --vis-dir /path/to/libero_4suite_lerobot_dataset/visualizations \
  --overwrite \
  --output-root /path/to/libero_4suite_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud
```

`--num-workers` 是 CPU / MuJoCo 进程并行，不是多 GPU 渲染。单次脚本只由
`MUJOCO_EGL_DEVICE_ID` 选择一张卡。若要使用四卡，最稳妥的方法是把四个 suite 拆为四条命令，
每条命令使用独立 GPU、`output-root` 和临时目录，最后再按确定顺序合并；不能让四个父进程同时
写同一个 LeRobot dataset。

`libero.json` 当前关键默认值是 256×256、10,000 点、500 个夹爪点、20 Hz、
`state_observation_offset=1`、`restore_demo_model=true`。命令行会覆盖 config。相对路径从
`benchmarks/song_real_libero` 解析。

### 9.5 输出与检查

```text
libero_4suite_lerobot_dataset/
├── data/
├── images/
├── meta/
├── point_clouds/episode_XXXXXX.zarr
├── world_ee_poses/episode_XXXXXX.npy
├── action_target_ee_poses/episode_XXXXXX.npy
└── libero_collect_summary.json
```

`--vis-count 0` 明确表示不生成可视化。要得到 PLY / 视频检查结果，使用非零
`--vis-count`，并建议显式传入 `--vis-dir`。转换后至少检查：

- 同一 demo 的重建视频与官方轨迹布局一致；
- 柜体、把手、目标物体没有固定偏移；
- `observation.state` 与 `action` 不再被错误地写成同值；
- 首个 UMI 动作不被强制为单位旋转；
- `libero_collect_summary.json` 中恢复模型和帧偏移均符合预期。

---

## 10. 真实机器人 HDF5 转换

真实 HDF5 已包含 XYZRGB，因此脚本不下载数据、不重播轨迹，也不从深度图重新计算点云。默认：

```text
cloud:       observations/cloud_rgb/<camera>
image:       observations/images/<camera>
pose:        observations/pose_eular
gripper:     observations/eff_angular
timestamp:   timestamp_ms
```

真实数据的末端 pose 默认已位于 overhead 相机坐标系。脚本：

1. 读取相机系 XYZRGB；
2. 在同一相机系添加虚拟夹爪；
3. 把 overhead 相机系视为固定 reference/world；
4. 将合并点云转换到当前 EEF；
5. 从人手示范 pose 构造 state/action；
6. 保存 RGB、时间戳和 sidecar pose。

当前真实采集没有单独记录控制器 command target，因此 state/action 都源自示范 pose。这与修复后的
LIBERO 数据语义不同，混合训练时必须意识到这一 domain 差异。

示例：

```bash
python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
  --input-dir /path/to/humanhand_hdf5 \
  --output-root /path/to/real_adapter_lerobot_dataset \
  --repo-id real_adapter_lerobot_dataset \
  --camera overhead \
  --cloud-frame camera \
  --pose-format auto \
  --timestamp-mode source \
  --num-points 10000 \
  --add-gripper-cloud \
  --gripper-points 500 \
  --gripper-len 0.06 \
  --point-cloud-storage zarr \
  --workers 8 \
  --vis-count 2 \
  --overwrite
```

如果输入 HDF5 已包含夹爪点，改用 `--input-has-gripper-cloud`，不要重复添加。adapter 训练需要
RGB，不能使用 `--image-key none`。图像应按 LeRobot image feature 接口保存为路径；直接把
NumPy image 放入期望路径的统计代码会触发 PIL `seek/read` 错误。

时间戳用于定义 episode 内真实采样时间和数据同步。`--timestamp-mode source` 优先使用 HDF5
时间戳；缺失时才按 `fps` 构造均匀时间。

---

## 11. 生成 PointSeg cache

### 11.1 cache 存储什么

cache 是 index-only 结构，不复制完整点云。每个样本包含：

- 当前 / 未来点索引；
- soft foreground / background score；
- 训练权重与有效性；
- 三个运动证据通道；
- episode / frame 映射；
- manifest 与生成参数。

当源点数不超过 `current-points` 或 `future-points` 时保留全部点；超过时无放回采样到上限。

### 11.2 四卡命令

```bash
export SONG_POINTSEG_REQUIRE_POINTOPS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id=/path/to/lerobot_dataset \
  --output-dir=/path/to/song_pointseg_cache_v7 \
  --current-points=10000 \
  --future-points=10000 \
  --batch-size=24 \
  --num-workers=8 \
  --shard-size=4096 \
  --storage-dtype=float16 \
  --nn-chunk-size=1024 \
  --vis-count=4 \
  --overwrite
```

`SONG_POINTSEG_REQUIRE_POINTOPS=1` 要求 CUDA PointOps KNN，缺少时直接报错，避免无意落入极慢的
PyTorch fallback。调试机器没有 PointOps 时可设为 `0`，但速度会明显下降。

### 11.3 可视化

`--vis-count > 0` 会输出原点云、pseudo soft score 和关键时间百分位结果。benchmark cache
预览使用连续的蓝 → 黄 → 红热力图：蓝色为低前景概率，红色为高前景概率，不做硬阈值。独立
PointSeg 训练脚本生成的 `pred.ply` 与 `pseudo.ply` 则共享“原始 RGB 背景 + 绿色前景覆盖”的
着色规则。两类文件都保存完整原点坐标，不会只保存标签或原点。

episode 的 p25 / p50 / p75 / terminal 可视化用于检查交互前景连续性。例如挂杯任务在 p75 到
terminal 期间，杯子、杯把与杯架都应保留连续软概率。

### 11.4 何时必须重建

以下任一变化都必须重建 cache：

- 重新运行 dataset 转换；
- 修改 `num_points`、点云采样或夹爪点；
- 修改点云坐标系或 RGB / XYZ 通道；
- 修复 demo model、state offset 或 episode/frame 映射；
- 修改运动先验算法、时间偏移或 evidence channel；
- 合并、裁剪或重排 dataset。

当前 strict 检查能验证长度和 episode/frame 索引，但不能为每帧点云做完整内容哈希。因此“程序没有
报错”不代表旧 cache 与新 dataset 语义兼容。

### 11.5 无 cache 在线计算

不提供 `pointseg_sample_cache_dir` 时，benchmark 训练脚本可以按 batch 在线计算同一类双向软先验。
它适合验证算法，不适合大规模重复训练。相关环境变量：

```bash
export SONG_POINTSEG_ONLINE=1
export SONG_POINTSEG_ONLINE_DEVICE=cuda
export SONG_POINTSEG_ONLINE_CURRENT_POINTS=10000
export SONG_POINTSEG_ONLINE_FUTURE_POINTS=10000
export SONG_POINTSEG_ONLINE_NN_CHUNK_SIZE=1024
export SONG_POINTSEG_ONLINE_FUTURE_OFFSETS=1,2,4,8,16,31
```

在线 CUDA prior 与 fork DataLoader 不兼容时，训练器会降低或关闭 worker。正式实验优先使用离线
cache。`SONG_POINTSEG_CACHE_STRICT=0` 只用于诊断，不应用于正式训练。

---

## 12. VLM 离线权重

建议分开保存官方架构 / processor 与 SmolVLA policy 权重：

```bash
huggingface-cli download \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir /path/to/hf_models/SmolVLM2-500M-Video-Instruct

huggingface-cli download \
  lerobot/smolvla_base \
  --local-dir /path/to/hf_models/smolvla_base
```

对应参数：

```bash
--policy.vlm_model_name=/path/to/hf_models/SmolVLM2-500M-Video-Instruct
--policy.vlm_weights_path=/path/to/hf_models/smolvla_base
--policy.load_vlm_weights=true
```

二者职责：

- `vlm_model_name`：提供完整 SmolVLM 架构、config、processor、tokenizer 和原始模型定义；
- `vlm_weights_path`：指定权重覆盖源，可以是 SmolVLA policy checkpoint、单个 safetensors，
  或完整 raw SmolVLM 目录。

当 `vlm_weights_path` 是 `lerobot/smolvla_base` 时，加载器从 policy state dict 中抽取
`vlm_with_expert.vlm` 参数，并严格检查当前保留的 VLM 层是否缺失。不要把只下载了
`smolvla_base` 的目录改名成 `SmolVLM2-500M-Video-Instruct` 后同时承担两种职责；其中可能没有
完整 raw Hugging Face processor / tokenizer 布局。

---

## 13. 训练

### 13.1 推荐 adapter 训练命令

```bash
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --multi_gpu --num_processes 4 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=/path/to/lerobot_dataset \
  --pointseg_sample_cache_dir=/path/to/song_pointseg_cache_v7 \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name=/path/to/hf_models/SmolVLM2-500M-Video-Instruct \
  --policy.vlm_weights_path=/path/to/hf_models/smolvla_base \
  --policy.load_vlm_weights=true \
  --batch_size=48 \
  --steps=80000 \
  --log_freq=1 \
  --output_dir=/path/to/outputs/wep_vla_v04_adapter \
  --job_name=wep_vla_v04_adapter \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=4000 \
  --eval_freq=4000 \
  --num_workers=8 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.001 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=1500 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.point_action_fusion_enable=true \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

`batch_size` 是每个进程的 batch。上例有效 batch 为 `48 × 4 = 192`。24 GB 显卡应从更小
batch 开始，避免某个 rank 被系统 `SIGKILL`。

### 13.2 warm start 与精确 resume

从已有模型权重开始新实验：

```bash
--policy.path=/path/to/checkpoints/last/pretrained_model
```

此时不要同时传 `--policy.type=smolvla`。这是 warm start：加载 policy 权重，但新建优化器和
学习率计划。

精确恢复训练状态应使用 checkpoint 保存的训练配置：

```bash
python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --resume=true \
  --config_path=/path/to/checkpoint/train_config.json
```

恢复时优先相信 checkpoint 自带 policy config。再次传 adapter / VLM / PointSeg 参数只用于
显式验证，不应让命令覆盖成与 checkpoint 不同的结构。

### 13.3 两个训练入口的关系

- `train_song_benchmark.py` 是本项目正式入口；
- `src/lerobot/scripts/train_song.py` 使用相同 policy 核心，但包含不同 debug defaults、在线
  prior 点数和诊断代码；
- 在 dataset、cache、seed、batch、processor、policy config 和采样顺序完全相同时，两者前向
  应接近；正式可复现训练不要混用入口。

### 13.4 训练日志

重点监控：

| 指标                           | 含义                                            |
| ------------------------------ | ----------------------------------------------- |
| `loss_action`                  | flow velocity MSE，不是 rollout 终点 action MSE |
| `loss_pointseg_aux`            | soft BCE 与平滑项的加权辅助损失                 |
| `pred_foreground_ratio`        | 模型前景预测比例                                |
| `pseudo_foreground_ratio`      | 有效 soft prior 的前景比例                      |
| `pseudo_valid_ratio`           | 当前 batch 中 prior 有效监督覆盖率              |
| `pointseg_operation_prob_mean` | 平均前景概率                                    |

训练时 `shuffle=True`。dataset 通常按 task / episode / frame 连续存储，不 shuffle 的前 200 个
batch 可能只覆盖单一阶段或容易样本，因此不能把该 loss 与训练日志直接比较。

---

## 14. 只读离线评测

`src/lerobot/scripts/eval_song.py`：

- 使用 `model.eval()` 与 `torch.inference_mode()`；
- 不创建 optimizer，不执行 backward；
- 评测前后校验参数未改变；
- 输出逐 batch 指标与 `eval_metrics.json`。

示例：

```bash
python src/lerobot/scripts/eval_song.py \
  --policy.path=/path/to/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false \
  --dataset.repo_id=/path/to/lerobot_dataset \
  --pointseg_sample_cache_dir=/path/to/song_pointseg_cache_v7 \
  --batch_size=32 \
  --steps=200 \
  --seed=0 \
  --output_dir=/path/to/eval_loss \
  --wandb.enable=false
```

该脚本测量 flow 训练目标，不是闭环成功率。比较 checkpoint 时固定：

- `seed`；
- batch size；
- `steps`；
- dataset 顺序 / shuffle；
- cache；
- policy noise；
- 评测入口。

同一 checkpoint 在 batch=6 和有效 batch=192 下得到不同瞬时 loss 是正常的；如果用完全相同的
batch index、noise 和 \(t\) 仍明显不同，才应检查 processor、checkpoint config 或数值路径。

---

## 15. LIBERO rollout 评测

### 15.1 当前执行链

1. 创建与 task 对应的 LIBERO 环境；
2. 等待自由物体稳定，期间保持机械臂初始 pose；
3. 确保初始夹爪完全张开；
4. 读取 `agentview` RGB / depth，构造点云；
5. 在 overview 相机系加入虚拟夹爪并转换到当前 EEF；
6. adapter checkpoint 按 image feature 自动映射到 `agentview`；
7. 模型生成 UMI EEF-relative action chunk；
8. 还原为 absolute OSC target；
9. 按配置执行若干 chunk step；
10. 记录 success、轨迹、动作和实时 JSONL。

当前 absolute pose 执行器使用模型目标本身，不添加启发式目标超调量。标准 LIBERO baseline 通常
直接输出环境定义的 delta action；本项目使用 absolute target 接口，因此应报告该动作接口与执行
频率，但不应通过人为 overshoot 修复数据 / 控制语义错误。

### 15.2 公平串行评测

除了明确研究控制频率或最大时限的消融外，公平比较应：

- 固定同一 checkpoint、初始 state、env seed 与 policy noise seed；
- 不在失败后强制恢复初态重试；
- 每个初始状态只进行一次连续 rollout；
- 使用 suite 标准最大时限；
- 不启用 dataset-domain 环境或 oracle action；
- 固定 `action-index`、`exec-action-steps` 与 gripper control；
- 报告是否 batch 推理。

推荐基准命令：

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/test_checkpoints/checkpoints/pretrained_model \
  --suite libero_spatial \
  --all-tasks \
  --episodes 100 \
  --suite-gpu-ids 0 \
  --isolated-policy-workers 1 \
  --task-workers 10 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --policy-noise-seed 0 \
  --env-seed 0 \
  --action-index 0 \
  --exec-action-steps 12 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --control-freq 20 \
  --use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /path/to/eval_serial
```

如果论文基线采用其他 `action-index`、执行步数或 5 Hz，必须作为独立 protocol 明确记录，不能与
20 Hz 结果混写。

### 15.3 四卡多 suite

四张 GPU、四个 suite：

```bash
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /path/to/checkpoints/last/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --suite-gpu-ids 0,1,2,3 \
  --all-tasks \
  --episodes 10 \
  --task-workers 10 \
  --task-worker-backend process \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --inference-batch-wait-ms 5 \
  --action-index 0 \
  --exec-action-steps 12 \
  --control-freq 20 \
  --use-suite-max-steps \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /path/to/eval_parallel
```

`suite-gpu-ids` 的数量应与 suite 数量一致。它会为每个 suite 启动独立 GPU 进程，而不是在一个
Python 进程中做模型并行。

为了最大复现性，保持 `inference-batch-size=1`。共享 batch 推理虽然更省显存，但会改变稀疏点云
排序、CUDA kernel 和随机 flow 的数值路径，接触任务可能被微小轨迹差异放大。若追求速度并接受
轻微非确定性，可以逐步提高到 task worker 数。

`task-workers=10, episode-workers-per-task=1` 表示每个 suite 同时维护 10 个 task 环境。
把 `episode-workers-per-task` 提高到 4 会变成约 40 个 MuJoCo 环境 / suite；四 suite 可达到
160 个环境，42 核 / 120 GB 机器很容易 CPU 过载或被系统以 exit code `-9` 杀死。不要只根据
GPU 显存估算并行上限。

### 15.4 实时结果

评测会实时追加：

```text
evaluation_events.jsonl
```

并在任务、suite 和全局结束时生成 JSON report。进程异常退出时，JSONL 可用于恢复已经完成的
episode 记录。

串行交互模式：

- `n`：当前 episode 记为失败并进入下一个；
- `v`：保存当前动作轨迹 / 点云可视化；
- `r`：触发回滚诊断功能。

并行环境 worker 大于 1 时键盘控制自动关闭。

### 15.5 dataset-domain 与 oracle 诊断

```bash
--dataset-domain-env
--dataset-domain-demo-root /path/to/libero_demos
```

会恢复训练 demonstration 对应的 model XML 和初始 state，用于判断模型在数据域内是否工作。
它不是标准 benchmark。

进一步添加：

```bash
--dataset-domain-oracle-actions
```

会绕过模型，直接重放官方原始 action。它回答的是“当前环境重建和执行器能否重放 GT”，不是模型
成功率。若 oracle 都失败，应先修复数据对齐、controller 目标或 gripper 语义，不应继续通过降低
模型 loss 解决。

---

## 16. 真实机器人与本地推理

### 16.1 标准 observation

真实部署传入：

```python
cur_model_observation = {
    "joint_1": None,
    "joint_2": None,
    "joint_3": None,
    "joint_4": None,
    "joint_5": None,
    "joint_6": None,
    "joint_7": None,
    "gripper_width": gripper_width,
    "overhead": img_overhead_rgb,
    "hand": img_hand_rgb,
    "point_cloud": overhead_cloud_rgb,
    "pose_eular": policy_pose_eular,
}
```

要求：

- `point_cloud` 是 fixed overhead 相机系 XYZRGB；
- `pose_eular` 是同一相机系下的 EEF `[x,y,z,euler...]`；
- 图像为 RGB，不是 OpenCV BGR；
- adapter checkpoint 必须提供 checkpoint 声明的 image feature，真实数据通常是 `overhead`；
- 包装器在相机系加入夹爪，再把点云转换到当前 EEF，最后调用 policy。

如果相机移动，但 `pose_eular` 仍来自旧固定相机标定，点云与 EEF 不再共系，模型必然失效。

### 16.2 pickle OOD / 回放推理

部署侧可直接保存完整 observation：

```python
with open(f"/home/liusong/temp/ood_test_new{sno}.pkl", "wb") as file:
    pickle.dump(cur_model_observation, file)
```

推理：

```bash
python benchmarks/song_real_libero/scripts/smolvla_model_inference.py \
  --policy.path /path/to/checkpoints/last/pretrained_model \
  --obs.path /home/liusong/temp/ood_test_new0.pkl \
  --device cuda
```

仅有 PLY 文件不足以完整测试 adapter checkpoint，因为 PLY 没有与采集同步的 RGB 和语言上下文。
PLY 模式只适合 point-only checkpoint，或作为完整 pickle 观测中的点云替换消融。

### 16.3 模态敏感性分析

推理脚本支持在固定初始 flow noise 下分别削弱 RGB、点云、语言或动作上下文：

```bash
python benchmarks/song_real_libero/scripts/smolvla_model_inference.py \
  --policy.path /path/to/checkpoint \
  --obs.path /path/to/observation.pkl \
  --analyze-modalities \
  --analysis-seed 0
```

该结果衡量模型对某模态的敏感性，不自动等于该模态“提升了动作质量”。没有 GT 或 rollout 时，
只能用于定位 OOD 和依赖关系。

冻结 VLM 仍存在 RGB OOD 风险：新机械臂、光照和背景会改变 image token。点云前景分支可以降低
但不能完全消除此风险，因为 Action Expert 同时读取 VLM prefix。应通过真实 RGB 多样性、图像
增强、模态消融和闭环测试判断，而不是仅观察训练 loss。

---

## 17. 可视化与诊断顺序

出现“loss 很低但 rollout 很差”时，按以下顺序检查，避免先改模型结构。

### 17.1 数据正确性

1. dataset 视频是否与官方 demo 或真实采集一致；
2. LIBERO 是否逐 demo 恢复 `model_file`；
3. `state_observation_offset` 是否为 1；
4. action 是否为 controller target，而不是 achieved state；
5. 点云、EEF pose、虚拟夹爪是否在同一坐标系；
6. RGB camera 与 checkpoint image feature 是否匹配。

### 17.2 cache 正确性

1. cache 是否由当前 dataset 生成；
2. episode/frame mapping 是否一致；
3. p25/p50/p75/terminal 的杯子、杯把、杯架软概率是否连续；
4. pseudo PLY 是否包含原点云并正确着色；
5. PointSeg pred 与 pseudo 的差异是欠拟合还是标签本身缺失。

### 17.3 模型前向

1. checkpoint config 是否启用预期 adapter / PointSeg / fusion；
2. 同一 batch、同一 noise、同一 \(t\) 下两个入口 loss 是否一致；
3. action pad mask 是否正确；
4. 3,072 点与全景点是否按预期选点；
5. modality ablation 是否显示异常 RGB 或点云依赖；
6. rotation-6D 是否先解码再比较。

### 17.4 执行器

1. oracle raw action 能否完成 dataset-domain demo；
2. absolute target 是否按正确 controller frame 执行；
3. gripper width、qpos 和 delta-width 的符号 / 尺度是否一致；
4. control frequency、action index、chunk 执行步数是否与基线一致；
5. 是否存在未报告的 hold、release、adaptive steps 或失败重试。

### 17.5 最后才评估模型能力

只有数据、cache 和 oracle 执行都正确后，才根据以下现象调整模型：

- PointSeg 在终止阶段丢失目标；
- RGB OOD；
- XYZ feature 对视角 / 原点过敏；
- 动作只记住固定轨迹，未随物体姿态旋转；
- flow sampling 方差过大；
- chunk 开环执行过长。

---

## 18. 兼容性与版本迁移

| 来源                                        | 是否可直接用于当前版本 | 处理                                                        |
| ------------------------------------------- | ---------------------- | ----------------------------------------------------------- |
| 修复前 LIBERO dataset                       | 否                     | 用逐 demo model + offset 1 的脚本重建                       |
| 修复前 dataset 对应 cache                   | 否                     | dataset 重建后重新生成                                      |
| adapter 版真实 dataset，点云和 frame 未变化 | 条件兼容               | 核对 image key、点顺序、episode mapping；不确定时重建 cache |
| cache v7 + 完全相同 dataset                 | 是                     | 使用 strict 检查                                            |
| v0.2 / v0.3 checkpoint                      | 条件兼容               | 以 checkpoint config 构造模型，不能强行开启新 fusion        |
| 当前 checkpoint + `worldflow_enable=true`   | 不建议                 | evidence channel 语义未对齐                                 |
| point-only checkpoint + PLY                 | 是                     | 不需要 RGB adapter                                          |
| adapter checkpoint + 只有 PLY               | 否                     | 还需同步 RGB、task、EEF pose                                |

版本迁移的原则不是“shape 能对上就兼容”，而是数据与 token 的物理语义必须一致。

---

## 19. 常见问题

### 为什么 PointSeg 预测 3,072 个前景点仍可能漏掉杯架？

数量正确不代表组成正确。模型可能把高概率都分配给杯子、夹爪或近相机机械臂。应比较终止阶段
pseudo 与 pred 的空间组成，而不是只看前景点总数。

### 为什么机械臂靠近相机时更容易前景错误？

近距离机械臂占据更多点和更大 RGB 区域，XYZRGB 特征分布也发生变化。若训练中此类视角不足，
PointSeg 会把高显著性机械臂点当作操作前景。虚拟夹爪和真实机械臂外观不一致也会放大问题。

### 为什么训练 loss 约 0.0004，shuffle 离线评测却到 0.002？

训练日志是不同随机 batch、noise 与 \(t\) 上的瞬时 flow velocity MSE；未 shuffle 的连续切片可能
只覆盖很容易的 episode 阶段。先固定样本索引、noise、\(t\)、processor 和 checkpoint config，
再做逐样本比较。不能把 rollout action endpoint MSE 与训练 flow loss混为一谈。

### 为什么模型第一条 rotation-6D 不再是单位旋转？

当前原点是实际 `observation.state`，第一条 action 是相对于实际 EEF 的控制器目标。它本来可以
包含位移和旋转。旧版以 `action[0]` 为原点才会人为得到单位首动作。

### 为什么 gripper 设置更大的 `gripper_len` 仍解决不了开柜失败？

`gripper_len` 只改变虚拟点云几何，不修复 controller target、state/action 对齐、夹爪控制符号和
接触执行误差。若 oracle 重放都打不开，优先检查数据与执行器。

### 标准 LIBERO 评测是否需要人为超调 target？

不需要。标准评测执行策略动作。对于本项目 absolute target 接口，应正确重建并执行目标，而不是
添加经验偏置来掩盖转换错误。

### TensorFlow / TF-TRT warning 是否影响转换？

通常不影响。它们来自间接依赖初始化，只要实际 PyTorch / MuJoCo 流程正常即可。

### `exitcode=-9` 是什么？

通常是操作系统 OOM killer 或调度器强制杀进程，不是 Python/CUDA 可捕获异常。降低
`num_workers`、环境进程、episode 并行或 batch size，并查看系统日志与主机 RAM。

### EGL 下 Open3D 报 GLX `BadAccess` 怎么办？

MuJoCo EGL 与 Open3D GUI/GLX 可能冲突。服务器上保存 PLY/NPZ 后离线查看，不要在同一进程调用
`draw_geometries`。无 EGL 时可用 OSMesa CPU 离屏渲染，但会更慢。

### output directory 只有 wandb 仍报“已存在”怎么办？

正式训练使用新的 output path，或确认入口允许仅含 `wandb/` 的目录。不要并行启动多个 rank 前
由每个进程竞争创建 / 检查同一目录；应由 accelerate 主进程统一初始化。

---

## 20. 环境与最小校验

推荐 editable install：

```bash
pip install -e ".[smolvla,libero]"
```

若未安装：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

LIBERO 离屏：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

提交训练前至少执行：

```bash
python -m py_compile \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/smolvla/smolvlm_with_expert.py \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py
```

并保存以下 provenance：

```bash
git rev-parse HEAD
git status --short
sha256sum /path/to/checkpoint/model.safetensors
```

每个实验报告至少记录：

- branch / commit；
- dataset summary 和 cache manifest；
- checkpoint hash；
- 全部 policy flags；
- GPU 数与有效 batch；
- eval seed、频率、时限、action index、chunk 执行步数；
- 串行 / batch / 独立模型推理方式；
- 是否启用任何 dataset-domain 或 oracle 诊断。

---

## 21. 代码地图

### 模型

- [`configuration_smolvla.py`](../../src/lerobot/policies/smolvla/configuration_smolvla.py)：
  policy 配置、稳定 / 实验开关与兼容参数。
- [`modeling_smolvla.py`](../../src/lerobot/policies/smolvla/modeling_smolvla.py)：
  PointSeg conditioner、LitePT、Point–Action fusion、WorldFlow、flow matching。
- [`smolvlm_with_expert.py`](../../src/lerobot/policies/smolvla/smolvlm_with_expert.py)：
  SmolVLM / Action Expert、VLM 权重解析、RoPE 和联合注意力。
- [`song_pointseg.py`](../../src/lerobot/policies/smolvla/song_pointseg.py)：
  双向轨迹软先验与 PointSeg 网络。
- [`umi_processor.py`](../../src/lerobot/processor/umi_processor.py)：
  当前 EEF 坐标系下的 state/action 变换。

### 数据和 cache

- [`libero_hdf5_to_dataset.py`](scripts/libero_setting/libero_hdf5_to_dataset.py)：
  LIBERO model restore、帧对齐、controller target 重建、点云 / RGB 保存。
- [`real_hdf5_to_dataset.py`](scripts/real_setting/real_hdf5_to_dataset.py)：
  真实 XYZRGB HDF5 转换。
- [`song_cache_pointseg_samples.py`](scripts/song_cache_pointseg_samples.py)：
  多 GPU index-only cache v7。
- [`libero_pointcloud_utils.py`](scripts/libero_setting/libero_pointcloud_utils.py)：
  深度反投影、相机 / EEF 变换和虚拟夹爪。

### 训练和评测

- [`train_song_benchmark.py`](scripts/train_song_benchmark.py)：项目标准训练。
- [`eval_song.py`](../../src/lerobot/scripts/eval_song.py)：无梯度离线评测。
- [`libero_pointcloud_eval.py`](scripts/libero_setting/libero_pointcloud_eval.py)：
  串行 / 多 GPU rollout、数据域和 oracle 诊断。
- [`smolvla_model_inference.py`](scripts/smolvla_model_inference.py)：
  dataset、pickle、PLY 与模态分析。

---

## 22. 参考

- SmolVLA: [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844)
- LIBERO: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
- Flow Matching: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Rotation 6D: [On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035)
- PointACT: [PointACT](https://arxiv.org/abs/2605.21414)

---

## 23. 后续开发检查表

每次恢复本项目时，先回答以下问题：

1. 当前 branch / commit 是否仍与文档基线一致？
2. dataset 是否由逐 demo `model_file`、offset 1、controller target 版本生成？
3. cache 是否由这份 dataset 重新生成？
4. checkpoint 内 adapter、PointSeg、Point–Action fusion 开关是什么？
5. WorldFlow 是否仍关闭；若开启，evidence channel 是否已正式迁移？
6. 训练与推理的 RGB key、点云坐标系和 EEF pose 是否一致？
7. 比较 loss 时是否固定了样本、noise、\(t\) 和 processor？
8. rollout 失败前，oracle action 能否通过同一执行器？
9. 评测协议是否明确频率、时限、chunk、gripper 和并行方式？
10. 新实验是否保存了 dataset/cache/checkpoint 的可追溯信息？

只要上述契约保持一致，模型架构、数据处理和评测结果才具有可比性。
