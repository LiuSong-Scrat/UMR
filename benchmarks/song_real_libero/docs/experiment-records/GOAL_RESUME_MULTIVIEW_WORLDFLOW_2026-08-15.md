# 新容器 `/goal` 指令与 Codex 权限配置

## 1. `~/.codex/config.toml`

把权限键放在顶层；不要把 `approval_policy` 和 `sandbox_mode` 写到 `[projects."/home/liusong"]` 表内。`sandbox_workspace_write.network_access` 只在 workspace-write 模式下生效；这里保留该项作为降级到 workspace-write 时的网络配置，而当前有效模式是 `danger-full-access`。

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "high"
approval_policy = "never"
sandbox_mode = "danger-full-access"

[projects."/home/liusong"]
trust_level = "trusted"

[sandbox_workspace_write]
network_access = true
```

官方配置参考：<https://developers.openai.com/codex/config-reference/>

新容器启动后先用 `/permissions` 确认：

- project `/home/liusong` 为 trusted；
- approval policy 为 never；
- sandbox 为 danger-full-access；
- network 可用。

## 2. 可直接粘贴的 `/goal`

````text
/goal ```text
继续完成 WEP-VLA 的 multiview + WorldFlow 联合优化。不要把阶段性准备、机制审计或小样本结果误写成最终完成；只有完整 LIBERO-10 Full500 与因果消融满足下述门禁，目标才算完成。

一、恢复并严格读取

1. 从 origin 恢复仓库：git@github.com:LiuSong-Scrat/Lerobot.git。
2. 先读取并严格沿用：
   - /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/docs/experiment-records/PROJECT_HANDOFF_2026-08-11.md
   - /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/docs/experiment-records/CONTAINER_RECOVERY_2026-08-15.md
3. V51 Git-tracked workflow 位于远端分支：
   wep_vla_v0.4.3_v51_workflow_archive
   commit 707ba6012ef834169e67f0d5b16305cede78bcb7
4. V51 可执行代码必须使用独立 worktree：
   branch wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation
   commit 9db8504a2c575ad386f1d90efa98784c0ea8d701
5. 读取 workflow archive 中：
   benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/README.md
   benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/CURRENT_CONVERSATION_ARCHIVE_2026-08-15.md
6. 用 install_runtime_bundle.sh 恢复运行脚本；若 /opt/data/private 仍挂载，复用原有 dataset/checkpoint/artifact，不复制或删除大型数据。
7. 当前权威实验根目录固定为：
   /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811
   所有脚本、日志、cache、checkpoint、评测输出和 artifact 继续统一放在该目录，不要分散。

二、最终目标与权威基线

1. 首先确保输入层双视角相对同 checkpoint 主视角不下降，最好提升。
2. 其次确保开启 WorldFlow 相对同 checkpoint 的 World→Ego-disabled 臂产生严格正增益。
3. 最终模型同时包含输入级 multiview 与 WorldFlow。
4. 权威 fixed-action matched Ego-only baseline 为 472/500。
5. 已通过的单视角 V32 step100 为 475/500；同 checkpoint World→Ego-disabled 为 469/500，WorldFlow 因果贡献 +6。
6. 最终 Full500 双视角+WorldFlow 必须：
   - 严格高于 baseline 472/500；
   - 不低于 V32 的 475/500；
   - 严格高于同 checkpoint 主视角+WorldFlow；
   - 严格高于同 checkpoint 双视角+World→Ego-disabled。
7. 必须完成同 checkpoint、同 episode、同 fixed-action cache 的 2×2 消融：
   - 双视角 + WorldFlow；
   - 主视角 + WorldFlow；
   - 双视角 + World→Ego disabled；
   - 主视角 + World→Ego disabled。

三、不可违反的模型与方法约束

1. 双视角只能在输入点云阶段融合，不在模型层面增加多视角专用模块；模型输入始终严格为 10,000 点，单/多视角天然兼容。
2. cache 以 1 cm 为基础尺度：主视角覆盖、PointSeg grid、在线/离线采样一致性都以 0.01 m 为准。V51 的 4 cm 仅用于副视角是否足够新颖的粗 cell 判断，不是把 cache 降采样到 4 cm。
3. 不使用固定 99:1 或其他视角 quota，不使用局部最近点一一交换、几何遮挡、人工语义、人工目标点、PCA 几何捷径或 task-specific 技巧。
4. 不引入任何 learned gate；worldflow_ego_residual_gate_init 必须为 None。
5. 不冻结 Ego；Ego 与 World 必须共同训练。
6. 保持双 point-action adapter + 单共享 point-action expert 的大框架。
7. World 与 Ego 必须描述同一物理轨迹并保持坐标共轭关系；World 信息必须实际影响最终 Ego action。
8. 不修改 LitePT 等外部函数库。
9. 不以新增辅助损失、门控、LR/dropout 网格或梯度缝补作为主要创新。若 V51 失败，应从通用输入表示、Ego/World 对称点路径适配和双流负交互本身找实质原因，不做 task-specific 修补。
10. 关闭 WorldFlow 时必须兼容原 checkpoint 和旧网络行为。

四、V51 当前方案与严格执行顺序

1. V51 从权威 V32 step100 checkpoint 启动；source model SHA256：
   c258303b70d4cab64f89d93c825905813f6f49fbb08cf282b5d4321e1fdf1fb4
2. V51 输入：主视角 1 cm 细覆盖 + 副视角 4 cm 粗新颖性过滤；无固定 quota；严格 9500 scene + 500 gripper。
3. V51 训练：Ego/World 两条直接 point path 对称适配；两条 point path 为 5e-8，其他 action/language/shared/residual 路径保持非零 5e-9；这只是当前有机制证据的输入域适配验证，不得预先宣称有效。
4. 恢复后先核对真实文件、分支、SHA、GPU、tmux 和进程状态；以当前文件系统与 origin 为准，不只依赖聊天记忆。
5. 重跑 V51 三组回归：SE3/WorldFlow、PointSeg/cache、training/cache/gradient entry；必须全部通过。
6. 先在 tmux 中用 4 GPU 生成 cache：4-sample smoke 后生成 137590 samples、36 shards、version 12 正式 cache。
7. cache 后必须完成 36-shard online/cache exact-index 审计，验证：1 cm 基准、scale4 副视角新颖性、严格 10k、unique indices、gripper exact。任何一项失败都不得启动训练。
8. cache/audit 通过后，在 tmux 启动完整 paired 训练：
   - 10 tasks × 50 episodes；
   - paired primary+dual 共 275180 samples；
   - 本地 4 GPU；batch 44/GPU；global batch 176；worker 12；
   - 1564 optimizer steps，覆盖完整 paired epoch + 84 samples；
   - W&B 必须启用；
   - 保存 260/520/780/1040/1300/final1564，不保存少于 100 步 checkpoint。
9. 长任务必须使用 tmux；网络断开不能结束 cache、训练或评测。
10. 训练时监控 W&B loss、OOM/NaN、四个 optimizer group、Ego/World point path 梯度和参数漂移。不要仅凭 action_loss 低就推断 rollout 一定好，也不要在未显示明显收敛时把性能差简单归因于步数不足。
11. 至少完成一个完整 paired epoch 后，按预注册协议筛查所有 checkpoint；不要用单 task 或前 10 episode 替代完整训练结论。

五、评测协议

1. 使用 strict official LIBERO init、action_index=0、fixed action/noise、canonical exact-action cache、fixed_barrier batching。
2. 本地权威配置：4 GPU、总环境并行 30、inference batch 30、policy-noise seed 0、env seed 7。
3. Broad 使用每 task IDs 0,5,10,15,20,25,30,35,40,45，共 100 states。
4. Broad 候选必须同时满足：双视角+World >95，且严格高于主视角+World和双视角+World-disabled。只有候选才进入 Full500。
5. Full500 使用每 task episodes 0..49；复用同一 canonical cache 做同状态比较；报告完整 2×2 简单效应和交互项。
6. A800 评测只使用物理 GPU 3,4,5，总环境并行最高 40，inference batch 固定 40；必须显式 fixed_barrier。不要再使用历史 6 卡/80 workers/batch80 配置。
7. batch 并行数本身不作为成功率变化解释；动态到达队列结果不能覆盖 fixed-barrier 权威结果。

六、权限、自动执行与安全边界

1. 使用 ~/.codex/config.toml：model=gpt-5.6-sol，model_reasoning_effort=high，approval_policy=never，sandbox_mode=danger-full-access；/home/liusong trust_level=trusted。
2. 不要为只读命令、常规文件编辑、测试、tmux、cache、训练、评测或已授权的 origin push 反复询问许可；在目标范围内直接执行并持续给出简短进展。
3. 绝对禁止删除 /opt/data/private 下任何文件。不得执行 git reset --hard、覆盖用户改动或其他破坏性清理。
4. 不把 checkpoint、cache、dataset 或视频推到 Git；只推代码、配置、脚本、轻量 artifact、handoff 和恢复文档。
5. 不修改或删除与本目标无关的脏工作区内容。
6. 遇到普通技术失败应在既定范围内自行诊断并修复；只有确实需要新的权限、外部凭据、破坏性操作或实质扩展目标时才停止。

七、归档与远端

1. 每个实质阶段及时更新 `PROJECT_HANDOFF_2026-08-11.md` 和实验 artifact，明确区分“已验证”“已否决”“未运行”。
2. 所有代码与 workflow 变更形成小粒度 commit 并推送到对应 origin 分支，以便容器随时重建。
3. 不删除失败 run、partial rollout、cache 或 checkpoint；保留证据并通过新 tag/目录写后续结果。

从恢复审计开始直接执行。当前已知状态是：V51 cache、训练和评测均尚未启动；上一次 pytest 因切换到归档请求而中断，没有新结果。不要把历史 preflight 当作性能证据。

八、2026-08-15 同日 continuation（优先级高于上一行的 launch-era 状态）

1. 上一行仅保留最初恢复点的历史记录。V51 后续已完成正式 cache、审计、完整 paired 训练和全部 six-checkpoint Broad；所有 checkpoint 均未严格超过 `95/100`，正式归档为 `screened_out`，没有 Full500。详见 `PROJECT_HANDOFF_2026-08-11.md` 的 6.10。
2. 当前继续方案为 V52 纯输入级 `1 cm voxel consensus + 4 cm persistent novelty`，fusion 名为 `consensus_multiscale_novelty_union`。可执行 branch/commit：`wep_vla_v0.4.3_v52_consensus_multiscale@9447a43a0e2ed3130a578618ac92225f71eb8a31`。workflow branch/commit：`wep_vla_v0.4.3_v52_workflow_archive@8fcf983`。详见 handoff 6.11、6.14。
3. V52 回归、4-sample smoke、正式 `137590` samples/36-shard v12 cache、36-shard online/cache exact-index audit、真实 V32 optimizer/gradient-role preflight、完整1564-step paired epoch和最终参数漂移审计均已通过。six-checkpoint Broad 已完成：step260=`96/93/96`，step520=`98/95/96`，step780/1040/1564 因理论最高95数学早停，step1300=`96/96/null`；唯一联合候选为 step520。step520 已由原 waiter 启动 Full500 第一臂 dual+World，必须继续同 checkpoint/episode/cache 的逐步2×2，不能重复启动已有目录或会话，也不能在最终 artifact 前宣称完成。最新证据详见 handoff 6.14。
4. 按用户要求，V52 runtime 已通过 merge commit `664f14e` 合入 `wep_vla_v0.4.3_multiview_doubleflow`，合并后回归 `144 passed` 且可直接解析 step520 checkpoint。续跑时以实时文件系统、origin 和 handoff 6.10--6.13 为准；不得把 V51/V52 launch-era 状态覆盖到更晚证据上。最终 Full500 门禁、安全边界和所有方法约束保持完全不变。
5. V52 step520 后续已在本地权威 Full500 dual+World 第一臂 `461/487`、26 failures、理论最高 `474/500` 时数学早停；aggregate 状态为 `failed_all_ranked_candidates`，其余3个Full500消融臂按门禁未启动，所有partial/cache保留。用户上传的单卡A800 batch80 Full500为 `462/500=92.4%`，只作为跨硬件交叉否定证据，不替代本地total30/batch30协议。详见handoff 6.15和 `v52_step000520_a800_gpu5_b80_full500_cross_hardware_audit.json`。V52已否决，下一轮优化必须继续满足原始方法边界和最终2×2门禁。
```
````
