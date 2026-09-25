# WEP-VLA / Song Real-LIBERO 项目交接文档

更新时间：2026-08-15

容器迁移恢复入口：`benchmarks/song_real_libero/docs/experiment-records/CONTAINER_RECOVERY_2026-08-15.md`。V51 的完整 Git-tracked 运行包位于远端分支 `wep_vla_v0.4.3_v51_workflow_archive@707ba6012ef834169e67f0d5b16305cede78bcb7`，当前可执行代码为 `wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation@9db8504a2c575ad386f1d90efa98784c0ea8d701`。

## 当前强制覆盖项（2026-08-15 V51 新容器恢复）

- 三组回归已重新通过；schema compatibility 修复后完整合并结果为 `130/130`。
- V51 v12 cache 已完成：`137590` samples、`36` shards、1 cm 基准、scale4 副视角新颖性；36-shard online/cache exact-index audit 的 strict10k、unique indices 与 gripper exact 全部通过。
- 第一次训练启动在 step 0 因 paired semantic-contract 错把支持的 dual v12 / primary v11 storage schema 差异当作不兼容而停止；没有 optimizer update 或 checkpoint，失败目录与日志保留。
- 通用修复 commit `9db8504a2c575ad386f1d90efa98784c0ea8d701` 只从 paired semantic equality 移除 schema version；每个 cache 仍由 reader 严格限制为 supported versions `(11,12)`。
- 独立重启 `v51r1_schemafix` 已于 2026-08-15 15:05 进入正式训练，tmux 为 `wep_v043_v51r1_schemafix_conservative_symmetricpoint_paired_train`；W&B online，4 GPU、batch44/GPU、worker12、目标1564 steps。step19 时无 OOM/NaN，约2.1秒/步；这只是运行健康证据，不是性能结论。
- 权威 output 为 `joint_multiview_worldflow/libero10_500ep/training/v51r1_from_v32step100_multiscale_novelty1cm4cm_paired_symmetricpoint5e8_policy5e9_devicebound_schemafix_4gpu_b44_w12_1564steps`。原无 schemafix 的 output 不得覆盖或删除。

## 当前强制覆盖项（2026-08-14）

- 此后所有 A800 测评只使用物理 GPU `3,4,5`；不得再按历史 6 卡配置启动新测评。
- A800 测评总环境并行上限为 `40`，推理 batch 固定为 `40`。
- LIBERO-10 的当前通用划分为：GPU 3 运行 tasks `0,3,6,9`，GPU 4 运行 tasks `1,4,7`，GPU 5 运行 tasks `2,5,8`；每个 task 使用 4 个 episode worker，总并行为 `16+12+12=40`。
- 当前入口脚本为实验根目录中的 `scripts/eval_libero10_v16_one_checkpoint_a800_gpu345_total40_b40_fixedbarrier.sh`；多 checkpoint 的 tmux 入口为 `scripts/launch_eval_libero10_v16_steps100_240_480_720_a800_gpu345_total40_b40_tmux.sh`。
- 同时筛选四个 checkpoint 时，使用实验根目录中的 `scripts/launch_eval_libero10_v17_steps100_240_480_720_a800_gpu345_concurrent4_total40_b40_tmux.sh`。该模式让 `100/240/480/720` 各使用 10 个环境并发，总并发仍为 40，推理 batch 参数仍为 40；不得与单 checkpoint 独占 40 环境的串行入口同时运行。
- v17 已完成一整个 `10 tasks x 50 episodes` 训练 epoch。本地 first-10 screen 的 `100/240/480/720` 为 `90/92/94/96`，但 step 720 的正式 500-state 流在 `267/286` 时出现第 19 个失败，最高只可能追平 `481/500`，已数学早停并保留全部输出。覆盖完整 episode 区间的固定分层集合 `0,5,...,45` 上，正常 WorldFlow 为 `92/100`，同 checkpoint 关闭 World-to-Ego 为 `96/100`，因果增益是 `-4`；此前 first-10 的 `+3` 是切片偏差，v17 已正式否决。
- v19 已从 exact-zero-residual、baseline-compatible 的 v14 起点启动，仅增加训练期 World-to-Ego stochastic depth `p=0.5`。它不增加参数、门控、辅助损失或推理分支；纯 Ego 和完整 World/Ego 样本共用原 action loss，且所有 Ego/World/残差参数保持非零学习率。训练使用 4 GPU、batch 48/GPU、worker 12、720 steps、W&B enabled，tmux 为 `wep_v043_libero10_v19_w2e_stochastic_depth_p50_720`。固定分层评测 waiter 为 `wep_v043_v19_stratified_worldflow_causal_after_train`，训练完成后自动评估 steps `100/240/480/720`，仅对超过同状态 baseline `95/100` 的 checkpoint 运行同 checkpoint 因果消融。
- v19 已完成并筛除。固定分层 episode IDs `0,5,...,45` 上，steps `100/240/480/720` 分别为 `92/93/95/92`，同 IDs baseline 为 `95/100`；没有 checkpoint 严格超过 baseline，因此 full-500 自动取消。补做的最佳 step480 同-checkpoint World-to-Ego 消融仍为 `95/100`，World 净因果增益为 `0`：World 正常臂独有成功 `(5,35),(6,25),(6,30)`，Ego-only 独有成功 `(1,45),(3,25),(9,30)`，共同失败 `(7,20),(9,25)`。v19 将 v17 的 broad World 因果效应从 `-4` 改善到 `0`，但尚未产生正增益。
- v20 residual boosting gradient isolation 已完成 720 步、138,240 个样本，即完整 `10 tasks x 50 episodes` 数据的一整个 epoch。它不增加参数、门控、辅助损失或推理分支；代码 commit 为 `fe99a71`，完整测试 `67/67` 通过，W&B 的 720 个 keep-ratio 均值为 `0.500752`。固定 broad steps `100/240/480/720` 为 `94/96/91/92`；step240 同-checkpoint 关闭 World-to-Ego 为 `93`，故 World 因果增益为 `+3`，是唯一候选。但正式 full-500 正常臂在 `259/279` 时已有 20 个失败，即便剩余 221 个全成功也最多 `480/500`，无法超过 `481/500` baseline，已数学早停。所有输出和 cache 均保留，matched baseline 与 full causal arm 未启动。v20 改善了 World 因果性但未达到最终目标，不能恢复多视角联合阶段。
- v21 已完成一整个 epoch。最终 runtime audit：720 条 W&B history 完整，World-correction keep ratio 均值 `0.245978`，三个 optimizer group 均保持非零 LR，78 个 World BN buffer 与 v14 逐位一致，4,326 个 residual 参数全部非零。固定分层 Broad 的 steps `100/240/480/720` 为 `95/89/95/97`；step720 同 checkpoint 关闭 World-to-Ego 为 `95`，所以 Broad World 因果增益为 `+2`。但正式 Full-500 在停止时为 `411/435`、24 个失败，即使余下65个全部成功也最多 `476/500`，无法超过 baseline `481/500`；v21 已正式否决，所有 partial 输出保留。
- v22 将 v21 step720 的547个 World-only参数保留、把1461个 baseline-common 参数逐位恢复，固定分层仅 `94/100`。这否定了“硬重置 shared/Ego 后直接拼接 World”的方案：Ego/shared 与 World 已产生共适应，不能硬拆。v23 对 v14→v21 的整套联合权重 delta 做统一 `alpha=0.50/0.75` 缩放，结果分别为 `95/100` 和 `94/100`，同样筛除；不再继续做 checkpoint 插值网格。
- v24 已完成 exact resume 到 step1440，累计处理 `276,480` 个样本（约 `2.0094` epochs）。W&B 有完整 1440 条 history，action loss 从 `2.4497e-4` 降到 `1.4639e-4`；78 个 World BN buffer 与 v14 逐位一致，4,326 个 residual 参数全部非零。固定分层 Broad 的 steps `960/1200/1440` 为 `92/95/93`，全部未严格超过同状态 baseline `95/100`，所以 v24 正式筛除，Full-500 未启动。继续同一 p=0.75 目标、同一 floor LR 增加第二个 epoch 已被否决。审计：`singleview_worldflow/libero10_500ep/artifacts/v24_step001440_exact_resume_floorlr_runtime_audit.json`；筛选：`taskbalanced_v24_floorlr_continuation_stratified_checkpoint_causal_screen.json`。
- v25 已完成并筛除。它处理 `138,240` 个样本（`1.004724` epochs），W&B 720 条 history 完整，action loss 从 `2.4864e-4` 降至 `1.6744e-4`；三个 optimizer group 均保持非零 LR，78 个 World BN buffer 与 v14 逐位一致，4,326 个 residual 参数全部非零。固定分层 Broad 的 steps `100/240/480/720` 为 `94/94/92/88`，没有 checkpoint 超过同状态 baseline `95/100`，Full-500 未启动。step240 参数审计显示 Ego action-in 漂移仅为 v20 的 `7.07%`，而 World body/residual 漂移分别保留为 `98.89%/105.12%`，所以“降低 Ego 漂移但保持 World 学习”在机械上已经实现，却仍随 World 训练成熟而退化；不再继续 dropout/LR 网格。证据：`v25_step000720_p50_anchor_common0005_runtime_audit.json`、`v25_vs_v20_step000240_low_drift_parameter_audit.json`、`taskbalanced_v25_p50_anchor_common0005_stratified_checkpoint_causal_screen.json`。
- v26 已完成并正式否决。代码 commit `90afd26` 新增默认关闭的训练期 World 坐标系 SE(3) 重参数化：对 World 点云、carrier `C` 和整条 World 轨迹一致施加随机 `A`，使 `G'=(AC)B(AC)^-1=A G A^-1`，映回 Ego 后仍严格得到同一物理动作 `B`。它只使用原 action loss，不增加第二次 forward、辅助损失、参数、门控或推理操作；完整 GPU 测试 `69/69` 通过。训练完成 720 steps、138,240 samples（`1.004724` epochs），W&B 720 行完整；坐标增强每步均启用，noise conjugacy error 为 0、path conjugacy error 约 `1.64e-7`，四项 World auxiliary loss 均为 0，三个 optimizer group 最终 LR 均非零，78 个 World BN buffer 与 v14 逐位一致，4,326 个 residual 参数全部非零。
- v26 固定分层 Broad 的 steps `100/240/480/720` 为 `96/94/96/93`。step100 的 same-checkpoint World-to-Ego-disabled 为 `89/100`，Broad World 因果增益 `+7`，因此进入正式 Full-500；step480 的消融反而为 `98/100`，World 因果增益 `-2`，其余 checkpoint 未进入 Full。权威 step100 Full-500 正常臂在 `391` 个已完成 episode 时为 `372/391`，累计第 19 个失败后，即使余下109个全部成功也最多 `481/500`，不能严格超过 established baseline `481/500` 和所需 `482/500`，故数学早停。matched baseline 与 Full causal arm 未启动，所有 partial 输出/cache 均保留。Broad artifact：`taskbalanced_v26_coordframe_aug_p75_stratified_checkpoint_causal_screen.json`；正式 gate：`taskbalanced_v26_coordframe_aug_p75_local4_fixedbarrier_full500_matched_worldflow_gate.json`。这说明随机坐标重参数化能产生很强的早期 Broad 因果信号，但不足以在 Full-500 上达到最终目标；不再搜索 dropout/LR/增强幅度网格。
- v27 的 asymmetric PCGrad 已在机制层面否决。它保持同一 forward/action loss 和同一模型，但 76 个 DDP update 中 Ego/World common-gradient cosine 仅在 `[-5.56e-6,7.18e-6]`，投影系数均值仅 `-4.84e-6`，数值上几乎等同既有 p=0.75 训练；因此在首个 checkpoint 前停止，没有生成 rollout，也不能作为性能证据。artifact：`v27_ego_priority_pcgrad_mechanism_rejection_step000076.json`。
- v28 已完成并正式否决。它完成1080步、138,240样本，W&B 1080行完整；Broad steps `100/240/480/720/1080` 为 `97/87/96/94/93`，其中 step100 的同 checkpoint World-to-Ego ablation 为 `95`，得到局部因果 `+2`。但权威 Full-500 正常臂在 `317/338` 时累计21个失败，即使余下162个全成功也最多 `479/500`，不能超过 established `481/500`，故数学早停。gate：`taskbalanced_v28_ego_tangent_projection_p75_local4_fixedbarrier_full500_matched_worldflow_gate.json`；全部 partial output/cache 保留。
- v28 的单次-forward residual time profile 定位到一个独立于学习率/梯度路由的连续时间错误：legacy head 直接输出终点 twist，再除以 `(1-t)` 还原速度，但网络在 `t=0.9` 没有把 residual 缩至 `t=0` 的十分之一。step100 平移 residual 均值从 `t=0` 的 `2.118 mm` 到 `t=0.9` 仍为 `1.851 mm`，对应 legacy implied velocity 从 `2.118 mm` 放大到 `18.511 mm`；step240/480/720 同样存在约一个数量级末段放大。证据：`v28_residual_time_profile32_summary.json` 及逐 checkpoint profile。此前通过两次独立 forward 做 action 差分的探针被标为无效，因为 repeat max-abs `0.021--0.033` 与 intervention effect 同量级。
- v29 代码 commit/HEAD `f15f72e` 增加默认关闭的 `worldflow_endpoint_residual_rate_parameterization`：head 预测有界 residual rate，物理 SE(3) 终点 twist 固定乘 `(1-t)`，随后除以 remaining 还原的速度不再解析放大，且 correction 在 `t→1` 自动为0。它不是门控，不增加参数、辅助 loss、第二次 forward 或 task-specific 规则；旧 checkpoint 默认推理逐值不变，v14 零 residual 起点仍 exact-Ego。完整 GPU 测试 `83/83`。
- v29 已完成并否决。固定分层 Broad steps `100/240/480/720/1080 = 97/91/94/94/92`，step100 同 checkpoint 关闭 World-to-Ego 为 `94`，局部因果 `+3`；但 Full-500 正常臂在 `362/382` 时已有20个失败，最高只能 `480/500`。rate 参数化修复了末段解析放大，但没有达到 Full 目标。
- v30 组合 v29 rate boundary、v26 训练期 World 坐标重参数化和 v28 shared Ego-tangent 保护。Broad 为 `93/90/96/91/95`；step480 同 checkpoint 关闭 World-to-Ego 为 `90`，World 局部因果 `+6`。Full-500 在 `322/341` 时累计19个失败，最高只能 `481/500`，正式否决。单次-forward profile 证明残差数值已稳定：step480 在 `t=0.9` 的有效终点平移仅约 `0.240 mm`，不存在 v28 的末段爆炸；剩余问题是残差选择性/表示，而不是幅值失控。
- v31 将同一6D残差解释为 Ego carrier-frame 左 twist；commit `eb6393e`，完整测试 `73/73`。一整个训练 epoch 正常完成，Broad steps `100/240/480/720/1080 = 93/94/92/95/93`，没有 checkpoint 严格超过同 IDs baseline `95/100`，因此 screened out，Full-500 未启动。该结果否定了 carrier-frame 左残差作为当前答案；全部 checkpoint、日志、cache 和 artifact 保留。
- v32 的历史“正式否决”结论已于 2026-08-15 纠正：当时错误使用了旧协议 `481/500` 作为门槛，并在 step100 `386/405` 时过早停止。使用当前同协议固定动作 baseline `472/500` 和完全相同的 fixed-barrier-v18 确定性评测代码续完后，step100 正常分支为 `475/500 = 95.0%`，同 checkpoint 关闭 World-to-Ego 的完整因果消融为 `469/500 = 93.8%`。因此 v32 同时取得 baseline `+3` 和 World 因果贡献 `+6`，联合门禁正式通过。历史 Broad 仍为正常/消融 `96/93`；此前 step480 的旧 Full partial 不改变 step100 的新权威结论。详细证据见 6.9。
- v33 已完成最终筛查并正式否决。commit `1b9905d` 将同一6D World head 从“预测终点上的有限 body twist”改为“当前 Flow 状态上的右平凡化 body-twist velocity”：`dB_t/dt = B_t hat(xi_body)`。该物理切向量被提升回预训练 pose9/rotation-6D 的原始 scale/shear gauge，直接加到 Ego Euclidean vector field；它不增加参数、门控、辅助损失、第二次 forward 或 task-specific 规则，零残差 exact-baseline，完整测试 `80/80`。正式训练使用单视角10,000点、全 `10 tasks x 50 episodes`、4 GPU、batch32/GPU、worker12、W&B、p=0.75 residual-anchor、shared Ego-tangent 和训练期 World 坐标重参数化。按用户要求训练在完整 step720 checkpoint 后停止；约731步之后的未保存更新不纳入结果。最终 Broad 的 steps `100/240/480/720 = 96/90/93/95`；只有 step100 超过同状态 baseline `95/100`，其同-checkpoint World-to-Ego-disabled 为 `90/100`，局部 World 因果增益 `+6`。但权威 Full-500 正常臂在 `378/398` 时已有20个失败，即使剩余102个全成功也最多 `480/500`，不能超过 established `481/500`，故数学早停；matched baseline 与 Full causal arm未启动。v33 比此前最接近目标的 v32 step100（`386/405`、19失败、最高481）更差。v33结束时没有遗留训练或评测进程，也没有继续该参数网格；多视角继续封存，直到 WorldFlow 真正通过 Full-500 与正因果 gate。
- v34 是新的结构性验证，不是参数网格。审计发现历史 `worldflow_frame_origin=current_ee` 同时承担了动作共轭和场景点云变换：它虽然消除了全局动作共轭的旋转杠杆臂，却也把 World 场景的绝对平移信息删除，导致所谓 World 分支实际只看到“以当前末端为原点、世界轴对齐”的局部点云。commit `29fb401` 新增默认保持历史行为的 `worldflow_scene_frame_origin=carrier`，并允许 `global` 仅对场景恢复固定世界坐标，而动作 Flow 继续使用零平移的 local-world carrier。它不增加参数、门控、辅助损失、第二次 forward、task rule 或 LitePT 修改；完整 WorldFlow/SE(3) 测试 `82/82`。v34 从 exact-zero v14 开始，使用单视角10,000点、全10-task×50-episode数据、4 GPU、batch48/GPU、worker12、720步和W&B；Ego/World/残差全部由原 action loss 联合训练，不使用 stochastic-depth、stop-gradient、梯度投影、World辅助损失或坐标增强。训练 tmux 为 `wep_v043_libero10_v34_global_scene_local_action_fulljoint_720`；Broad与Full waiter分别为 `wep_v043_v34_global_scene_local_action_broad_after_train` 和 `wep_v043_v34_all_candidates_full500_after_broad`。初始更新有限且约2.1秒/步，显存约21.6–23.7GiB/GPU；W&B run为 `worldflow-v34-global-scene-local-action-bodyendpoint-rate-fulljoint-20260814`。
- v34 已完整结束 `720/720`，checkpoint `100/240/480/720` 均落盘且训练日志无 OOM/NaN。新的固定 action baseline Full-500 已完成：`472/500 = 94.4%`，本地4卡、fixed-barrier30、推理batch30、seed0、canonical exact-action cache `read_write`。同一 canonical 结果在分层 IDs `0,5,...,45` 上仍为 `95/100`，因此 Broad 的严格 `>95` 门槛协议匹配。历史 `481/500` 仅保留为 established older-protocol reference，不能冒充 matched baseline。权威 artifact：`taskbalanced_baseline_v042_fixed_action_step030000_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_canonical_fixed_action_v34protocol.json`；补充审计：`baseline_v042_fixed_action_v34protocol_full500_audit.json`。v34 Broad 已自动启动 step100。
- 用户已明确放宽 WorldFlow 设计边界：原 v0.4.3 WorldFlow 只是概念原型；只要求 `worldflow_enable=false` 时旧网络/checkpoint兼容。World分支可以从新初始化或从Ego bootstrap，不必保留旧World函数。必须保持双 point-action adapter + 单共享 point-action expert，大目标是让两个坐标流真正互相学习并超过 Ego-only，而不是继续做残差/LR/梯度补丁；仍禁止门控、task-specific技巧和LitePT外部修改。
- 对称双坐标 v35 已在独立 worktree `/home/liusong/ProgramFiles/Huggingface/lerobot_worldflow_symmetric_dualcoord`、分支 `wep_vla_v0.4.3_worldflow_symmetric_dualcoord` 实现，commit `4562fd17c8de5c7ee15ae930b1afdebb7881fd93`。默认 `legacy_asymmetric` 完全保留历史路径；新 `symmetric_dual_coordinate` 将 Ego/World scene 放在一个上下文 block、两套 action token 放在同一共享 expert block，使两流在每层 expert 内双向注意。两个 head 分别接受坐标正确的直接 Flow 监督，但只执行 World-informed Ego head；没有固定 endpoint 投票或物理 residual。完整 CPU 测试 `84/84`，关键 CUDA 互注意/梯度/因果mask测试 `5/5`。
- v35 只在 v34 完整 matched Full500+causal gate 被拒绝后启动。条件训练 tmux：`wep_v043_v35_symmetric_dualcoord_after_v34_gate`；Broad和Full tmux：`wep_v043_v35_symmetric_dualcoord_broad_after_train`、`wep_v043_v35_all_candidates_full500_after_broad`。若启动，它直接从 immutable Ego-only baseline 做一次 Ego-to-World bootstrap，使用global World scene、local action carrier、projected Ego path、Ego/World等权直接Flow目标、4GPU、batch48/GPU、worker12、720步、W&B和checkpoint100/240/480/720。完整 World-view 因果消融会原位mask World scene+action tokens并跳过post-expert W2E，不改变布局、seed或episode。
- v34 Broad 最终筛查已确定：step100为`92/100`；step240正常/关闭World-to-Ego为`96/95`，局部因果`+1`；step480为`96/93`，局部因果`+3`。step720在`83/94`、11个失败时已数学不可能超过95，按用户要求立即停止，最大可能仅`89/100`，未启动其因果臂且保留全部partial/cache。step240与step480均进入Full-500。首个控制器错误沿用了历史`481/500`门槛并在step240 `374/393`、19失败时停止；该partial保留但判据已废止，因为权威同协议baseline是`472/500`。corrected v2要求至少`473/500`、到第28个失败才数学早停，并复用原canonical cache在新output tag下重跑；tmux为`wep_v043_v34_all_candidates_full500_matchedfixedaction_v2`。阈值纠正审计：`v34_full500_gate_matched_fixed_action_threshold_correction_v2.json`。
- corrected v34 Full门槛最终正式否决两个候选：step240在`470/498`达到第28个失败，最高只能472；step480在`439/467`达到第28个失败，最高也只能472。aggregate为`taskbalanced_v34_global_scene_local_action_bodyendpoint_rate_fulljoint_local4_fixedbarrier_all_candidates_full500_matched_worldflow_gate.json`。v35随后自动启动；首次因独立src-layout worktree未置于`PYTHONPATH`而在任何update前退出，已在启动脚本中固定v35 `src`并增加import-path/新字段preflight后重启。正确运行已进入4卡batch48/GPU、worker12、W&B训练，step20有限稳定、约1.8--2.0秒/步、显存约22.7--24.0GiB，bootstrap明确`ego_frozen=False`。v35 Broad/Full evaluator现已显式断言cross-attention、symmetric layout、bootstrap和World direct loss，并使用matched baseline472/required473，不再继承v34 endpoint-residual假设或历史481门槛。
- v35 step100 参数级运行审计已通过：零初始化的 Ego→World 与 World→Ego cross-attention 输出投影均已全量非零（L2分别`0.04474/0.05318`）；从Ego复制出的两套 point-action adapter 在20个对应tensor上已独立分化；Ego action输入/输出与Ego adapter也均相对immutable baseline发生非零更新。三个optimizer group均为正LR，配置中无门控、physical residual执行融合、冻结Ego、task rule或geo/bridge/equiv辅助目标。这证明双坐标与双向交互实际参与训练，但性能仍只以rollout为准。artifact：`singleview_worldflow/libero10_500ep/artifacts/v35_step000100_symmetric_dualcoord_runtime_audit.json`。
- v35 已完成720步/138,240样本和checkpoint100/240/480/720，但四个checkpoint均在Broad中被数学早停，正式`screened_out`：step100=`54/71`且17失败、最高83；step240=`26/33`且7失败、最高93；step480=`28/33`且5失败、最高95；step720=`19/27`且8失败、最高92。无checkpoint可能严格超过同协议Broad baseline `95/100`，所以没有启动因果臂或Full500。完整screen为`taskbalanced_v35_symmetric_dualcoord_global_scene_coprediction_stratified_checkpoint_causal_screen.json`，Full gate为`screened_out`。所有partial/cache保留。参数活动证明两流确实学习，因此失败定位为拓扑破坏：对称布局把scene放到Ego action之前，改变了预训练Ego action token的位置和shared expert函数。
- v36 已启动function-preserving Ego-first结构控制，不做loss/LR补丁：Ego action保持预训练首suffix block及位置，追加的World scene/action通过同一shared Action Expert读取Ego，零输出初始化的post-expert World→Ego cross-attention再把World写回Ego；两套point-action adapter、一个shared expert、global World scene、direct Ego/World Flow共同训练均保留，无门控、residual投票、冻结、task rule或LitePT修改。训练从immutable Ego-only baseline bootstrap，4GPU、batch48/GPU、worker12、720步、W&B和checkpoint100/240/480/720；tmux为`wep_v043_v36_egofirst_train`，Broad/Full waiters为`wep_v043_v36_egofirst_broad_after_train`与`wep_v043_v36_egofirst_full500_after_broad`。W&B：`worldflow-v36-egofirst-global-world-coprediction-20260814`。
- v36 step100运行审计已通过：Ego→World/World→Ego输出投影已从bootstrap零输出增长到L2 `0.04348/0.04649`，两套adapter的144,496个对应参数全部分化，Ego action输入/输出与Ego adapter均相对immutable baseline更新且`ego_frozen=false`。legacy布局只更新World action-type行（720/1440非零），预训练Ego action首块没有额外type embedding扰动。artifact：`singleview_worldflow/libero10_500ep/artifacts/v36_step000100_egofirst_runtime_audit.json`。
- v32 通过权威 WorldFlow gate 后已经按约定解档输入层双视角。所有候选都保持模型架构不变、场景+gripper总输入严格为10,000点，并且没有门控、task-specific规则、几何遮挡或LitePT改动。旧方案的固定分层筛查已确认：v41 novelty-union-1cm为`95/100`；v43 union-FPS最高`93/100`；v44 novelty-3cm最高`93/100`；v45 voxel-cover-FPS-1cm最高只能追平V32的`96/100`；其完整partial/cache均保留。v41 Full500在`328/355`、27失败时已无法达到同权重主视角V32的`475/500`，并数学早停；完成状态上相对主视角为`-10`。
- v46 是当前新的通用输入 coreset，独立worktree为`/home/liusong/ProgramFiles/Huggingface/lerobot_v40_v32_novelty_union`，分支`wep_vla_v0.4.3_v46_multiscale_novelty_union`。输入融合实现commit为`9b63311bd23392d21cf164745ba5437aabc89ad6`；当前HEAD为`d752f14df8c75cfcd9acd9c6b34d8e2e7d5b7296`，后者仅在内部训练入口把DDP policy加载显式绑定到各rank本地`cuda:N`，保存配置时仍恢复可移植的`cuda`，没有修改LitePT或模型函数。方法在1cm细网格保护全部主视角覆盖，在3cm粗网格只接纳仍然缺失的副视角cell且每cell一个代表；插入数量完全由几何覆盖决定，不分配视角quota。真实数据30样本审计为严格10,000点，副视角插入均值`89.3`、范围`1--616`，并验证主视角细cell、主视角gripper尾部和副视角粗cell约束。完整测试`109 passed`；DDP设备绑定定向测试另为`3 passed`。zero-shot固定分层为`93/98`后第5个失败，最高95；在已完成状态上稳定恢复V32的4个失败但新增5个失败，因此zero-shot被拒绝，不过`-1`且稳定恢复失败状态，保留一次充分paired训练的潜力。
- v46全量cache已于2026-08-15完成，输出为`joint_multiview_worldflow/libero10_500ep/pointseg_cache_multiscale_novelty_union_1cm3cm`。manifest为version 11、`num_samples=137590`、4个rank、36个完整shard且无临时shard，模式为`indices`、`multiscale_novelty_union`、fine voxel `0.01m`、严格10,000点（9,500 scene+500 gripper）。随后36-shard审计逐shard取中点重新在线融合，全部满足cache与在线`point_indices`逐元素一致、严格10k、索引唯一和gripper尾部完全一致；副视角插入范围`3--1103`、均值`131.278`，确认不是固定quota。权威artifact为`joint_multiview_worldflow/libero10_500ep/artifacts/v46_cache_online_exact_index_audit_36shards.json`。所有cache均保留，不得删除。
- exact-index审计通过后，V47首次从权威v32 step100启动，双/单视角paired coverage为`dataset.num_frames=275180`。原batch48/GPU运行在step34因rank0加载期额外CUDA上下文导致OOM；增加内部DDP本地设备绑定后，重试确认每卡严格一个rank并越过step34，但在step143遇到真实最坏稀疏batch显存峰值：rank2只剩`765.75 MiB`而该步需`804 MiB`。两个失败run、日志与W&B均原样保留，没有checkpoint，也没有删除任何`/opt/data/private`文件。
- V47s 已完成全部`1564`步：4 GPU、batch44/GPU、worker12、W&B，`1564*176=275264`覆盖完整275180 paired样本并只超出84样本。最终短尾200步 action/World loss 分别下降`9.71%/9.49%`，但长窗口782步仅下降`1.49%/1.60%`，属于epoch尺度平台。四个optimizer组和Ego/World两套point-action adapter、shared expert、Ego action I/O、双向cross-attention均发生非零漂移。六个固定分层双视角+World checkpoint `260/520/780/1040/1300/1564`分别为`91/87/87/89/84/92`个成功（均在第5个失败时数学早停，括号外为已完成时成功数，最大可能均为95），无候选进入Full500。最终step1564同checkpoint主视角+World为`93/100`，相比V32主视角+World的`96/100`也下降，说明主要问题是共同策略漂移/遗忘，不只是双视角输入。aggregate为`joint_multiview_worldflow/libero10_500ep/artifacts/v47_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json`，primary诊断artifact为`singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v47s_v32_multiscale_novelty_paired_uniformlr5e8_b44_step001564_4gpu_total30_b30_alltasks10ep_codefbfacd7_fixedbarrierv18_primaryonly_stratified_step5_primary_world_diagnostic.json`。V47正式否决，所有输出保留。
- V48 是针对V47共同策略漂移的通用输入域适配控制，不修改V46输入融合、V32模型、action loss或paired数据。point-input adaptation组初始LR为`5e-8`；pretrained Ego/shared nonpoint、new World bidirectional和physical residual head均以非零`5e-9`联合训练，因此不是冻结、门控、辅助loss或task-specific技巧。正式配置仍为4 GPU、batch44/GPU、worker12、1564步、W&B和一个完整paired epoch；输出为`joint_multiview_worldflow/libero10_500ep/training/v48_from_v32step100_multiscale_novelty1cm3cm_paired_point5e8_policy5e9_devicebound_4gpu_b44_w12_1564steps`，训练tmux为`wep_v043_v48_point_dominant_paired_train`。step260只读审计已精确确认四组初始LR为`[5e-9,5e-8,5e-9,5e-9]`、当前LR均为正、四组参数及Ego/World两套point-action adapter、shared expert、Ego action I/O和双向cross-attention全部发生非零漂移；无OOM/Traceback。同期action/World长窗口分别轻微上扬`0.87%/1.00%`，仅作为负迁移预警，不替代完整epoch rollout。收敛/参数审计、六checkpoint固定分层三臂筛查和候选Full500 2x2门禁已分别在`wep_v043_v48_convergence_monitor`、`wep_v043_v48_all_checkpoints_stratified_3arm`、`wep_v043_v48_all_candidates_full500_2x2_gate`中等待；不会在完整paired epoch前用早期结果替代最终结论。
- V48 step780仍稳定、无OOM/Traceback；三个260步窗口的action/World均值依次为`2.3733e-4/2.4038e-4`、`2.3576e-4/2.3788e-4`、`2.2793e-4/2.2952e-4`，长780步前后半仅下降约`1.02%/0.64%`，状态仍为`no_material_tail_change_detected`，继续完整epoch。进一步审计发现V48高LR point-input组只包含Ego点输入路径，World `scene_encoder/scene_context_proj/point_action_adapter`仍在低LR World组；step260的World point-action adapter漂移仅为V47的`5.52%`，而Ego adapter为`106.83%`。因此已准备但未启动V49语义修正：独立worktree`/home/liusong/ProgramFiles/Huggingface/lerobot_v49_symmetric_point_adaptation`、branch`wep_vla_v0.4.3_v49_symmetric_point_adaptation`、当前有效HEAD`0b8e7b48dbddcbab847737ef6d676709cc8dfbef`。它仅在训练期把Ego/World两条直接点输入路径对称放入`5e-8`组，所有action/language/shared/residual路径仍以非零`5e-9`联合训练；forward、模型结构、loss与旧配置默认行为不变，并显式拒绝在legacy/不支持的fusion下静默启用。无训练机制审计发现旧HEAD `ba5429d`仍按optimizer group名称实施V32 Ego-tangent保护，会把移入point组的World点路径误归为Ego/shared并清除其World样本梯度，因此旧HEAD正式无效。commit `5de3eae`将物理梯度角色与LR分组解耦，commit `0b8e7b4`进一步加入训练入口的直接梯度合并回归测试：Ego/World参数故意同处高LR point组，合并后两者梯度均精确保留且不执行权重更新。真实V32 source上1063个高LR point tensor中，234个World-only tensor明确保留World梯度，829个Ego point tensor仍受保护；全1243个trainable tensor无重叠/遗漏，四组仍为`153/1063/25/2`和`5e-9/5e-8/5e-9/5e-9`。artifact为`joint_multiview_worldflow/libero10_500ep/artifacts/v49_real_checkpoint_optimizer_and_gradient_role_preflight_v2.json`；显式worktree `PYTHONPATH`测试为WorldFlow/SE3 `78/78`、PointSeg `34/34`、训练入口WorldFlow dataset/gradient merge `14/14`。训练入口继续硬断言导入路径和精确HEAD，否则在任何update前退出。V49当前输出目录和训练tmux均不存在，确认未训练且不会自动启动。
- 用户于V48实际step1108明确要求停止训练、直接测试。V48训练及原final-checkpoint monitor/waiter均已停止，四卡已确认释放；未保存的1108更新不进入判断，完整checkpoint只有`260/520/780/1040`。V49条件启动tmux也已取消，V49不会自动训练。existing-only fixed-barrier30/inference batch30分层100状态筛查已全部完成：step260为`87/92`且5失败、最高95；step520完整`95/100`；step780为`79/84`且5失败、最高95；step1040为`94/99`且5失败、最高95。四点均不能严格超过Broad baseline `95/100`，所以全部筛除，没有运行primary/World causal arms或Full500。四个checkpoint的失败集合交集为空，只有`(task8,episode15)`重复2次、`(task9,episode25)`重复3次；step520/780均修复了V32的4个Broad失败，却各自引入5个新失败。这证明副视角信息仍能恢复具体状态，但V48只是在移动失败边界而没有形成净稳健性，继续同一训练轨迹缺乏单调改善证据。aggregate为`joint_multiview_worldflow/libero10_500ep/artifacts/v48_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json`，final gate为`joint_multiview_worldflow/libero10_500ep/artifacts/v48_all_candidates_full500_2x2_multiview_worldflow_causal_gate.json`且状态`screened_out`。全部评测进程已退出、四卡空闲、所有partial/cache保留。
- V49 的评测判据已在训练前预注册，artifact为`joint_multiview_worldflow/libero10_500ep/artifacts/v49_preregistered_stratified_and_full500_2x2_causal_gate_protocol.json`，并锁定评测脚本/evaluator SHA256。Broad使用固定分层100状态，要求`双视角+World > 95`、严格优于同checkpoint的`主视角+World`和`双视角+关闭World→Ego`。Full500运行完整2×2四臂，最终要求`双视角+World > baseline 472`、不低于V32的`475`，同时严格优于同checkpoint主视角臂和World消融臂；另报告另外两个简单效应与交互项。该artifact已通过JSON、source artifact与脚本哈希复核，状态仍为`preregistered_unrun`；没有启动V49训练或评测。
- 按“停止训练、直接测试现有权重”补完V48最佳step520的机制诊断：同一固定分层100状态下，`双视角+World=95`、`主视角+World=97`、`双视角+关闭World→Ego=89`，因此多视角净因果贡献为`-2`，WorldFlow净因果贡献为`+6`。失败集合进一步显示，多视角伤害的4个状态`(0,45),(1,0),(2,15),(6,30)`与WorldFlow伤害的4个状态完全相同；与此同时WorldFlow恢复10个其他状态，多视角恢复2个状态。这把V48失败定位为“双视角输入与未充分适配的World点路径之间的负交互”，而不是WorldFlow优势消失；它为V49的Ego/World点路径对称适配提供了直接机制依据，但不是V49性能证据。汇总为`joint_multiview_worldflow/libero10_500ep/artifacts/v48_step000520_postscreen_multiview_worldflow_causal_diagnostic.json`，集合审计为`v48_step000520_postscreen_failure_set_interaction_analysis.json`。V48仍保持`screened_out`，未启动Full500，诊断tmux已退出、四卡空闲。
- 按“停止训练、直接测试”又完成了V32现有权重的保守multiscale novelty零训练边界筛查。独立worktree为`/home/liusong/ProgramFiles/Huggingface/lerobot_v50_conservative_multiscale_novelty`、branch`wep_vla_v0.4.3_v50_conservative_multiscale_novelty`、HEAD`3daeddd7de6afb7db086dffb3a0ebcf590d5db9c`；仅把输入预处理的副视角粗新颖性尺度变成可配置字段，默认仍为3.0，旧checkpoint逐值兼容，不改模型/权重/loss/LitePT且无门控。真实30帧几何审计中scale3/4/5/6的副视角插入均值分别为`89.3/53.1/34.733/23.8`，全部满足严格10k、唯一索引、主视角1cm细cell保护、副视角粗cell新颖和gripper精确保留；只实测最有依据的scale4与唯一scale5边界，不继续scale6。scale4在`93/97`、4失败时最高只能`96/100`，同97状态V32为`94`，恢复3个V32失败但新增4个；scale5在`92/97`、5失败时最高只能`95/100`，同97状态V32为`93`，同样净`-1`。两者均数学筛除，未启动Full500，零训练coarse-scale族关闭。证据为`joint_multiview_worldflow/libero10_500ep/artifacts/v50_v32step100_zeroshot_conservative_multiscale_stratified_rejection.json`、`v50s5_v32step100_zeroshot_conservative_multiscale_stratified_rejection.json`和`v50_conservative_multiscale_scale_selection_geometry_audit_30samples.json`；所有partial/cache保留，两个tmux已退出、四卡空闲。V49仍保持已验证但未训练状态，后续不得把V50零训练失败误写成V49性能证据。
- 基于V48“World点路径适配不足”的机制诊断与V50“scale4是唯一可追平V32 Broad上界的保守输入”证据，已离线准备但未启动V51。独立worktree为`/home/liusong/ProgramFiles/Huggingface/lerobot_v51_conservative_symmetric_point_adaptation`、branch`wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation`、HEAD`93e2d8a3177a8addc40229da00962fcc2e7b7100`。V51合并V49的Ego/World直接point路径对称`5e-8`适配与V50的fine1cm/coarse4cm输入；所有非point action/language/shared/residual路径仍以非零`5e-9`联合训练，不改forward模型、action loss、架构或LitePT，也无门控。coarse scale已贯通无cache训练、在线pseudo collator、离线cache current/future采样、manifest和cache注入；cache schema升至12但兼容不可变version11 primary cache，历史缺字段的multiscale cache只解释为3×，V51配置4×会明确拒绝而不静默混用。测试为SE3/WorldFlow`78/78`、PointSeg/cache`35/35`、训练/cache/梯度入口`16/16`；真实V32参数预检仍为1243个trainable tensor无遗漏/重叠，四组`153/1063/25/2`，234个World-only point tensor保留World梯度、829个Ego point tensor保持Ego-tangent保护，artifact为`v51_real_checkpoint_optimizer_and_gradient_role_preflight_v2.json`。已准备4卡cache+36-shard exact-index审计脚本和4卡batch44/卡、worker12、1564步、W&B、完整paired epoch训练脚本；训练硬依赖version12 scale4全量cache与在线索引审计，不存在则在update前退出。预注册协议为`joint_multiview_worldflow/libero10_500ep/artifacts/v51_preregistered_training_and_2x2_causal_gate_protocol.json`，Broad要求双视角+World严格超过95并严格优于同checkpoint主视角与World消融；Full500要求超过baseline472、不低于V32的475且两个因果贡献均严格为正。当前4×cache目录、V51训练输出和所有V51 tmux均不存在，确认cache/训练/评测均未启动；不得把机制预检写成性能证据。
- 单GPU A800命令若只给`--inference-batch-size 50`而未给`--inference-batching-mode fixed_barrier`，实际仍是dynamic batching。一次baseline `99/100`和v32三次`91/95/91`均属于动态到达队列结果，不能覆盖fixed-barrier权威结论。既有Full500 canonical结果抽取相同episode `0--9`时，baseline为`92/100`、v32为`93/100`；Full500仍分别为`472/500`和`475/500`。
- 文档后文出现的 6-A800、80 workers 或 batch 80 均是历史实验记录，仅用于追溯结果，不再代表当前执行配置。

本文档用于在新 Codex 会话、换人维护或长时间中断后快速恢复项目上下文。内容区分为：

- 已由代码或实验确认的结论；
- 当前工作区状态；
- 尚未完成、不能当作结论的事项；
- 下一位执行者应立即开展的工作。

## 1. 项目背景与目标

项目基于 LeRobot/SmolVLA，主要研究以下内容：

1. 使用 RGB、语言、机器人状态和 3D 点云生成机器人动作 chunk；
2. 使用基于轨迹运动先验的软前景标签训练 PointSeg，从场景点云中提取操作相关物体；
3. 保持官方 SmolVLA/SmolVLM 的 VLM 结构，支持载入官方预训练权重；
4. 同时支持 LIBERO 仿真数据与真实 RGB-D/人手示教数据；
5. 探索 World/Ego、SE(3) 等变动作生成等辅助结构，但这些结构默认关闭；
6. 最终希望提高视角变化、物体位置变化、操作位姿变化和异构末端条件下的泛化能力。

整体数据流如下：

```text
LIBERO HDF5
  -> 恢复每条 demo 的 model_file 并重放 state
  -> RGB / overhead 点云 / 绝对末端动作 LeRobotDataset
  -> PointSeg 轨迹先验 cache
  -> WEP-VLA 训练
  -> 官方 LIBERO episode 评测

真实 RGB-D + 相机轨迹
  -> WiLoR/MANO 手部重建
  -> RGB-D 全局度量对齐
  -> handpose_wilor.jsonl
  -> 按独立 recording segment 切分 HDF5 episode
  -> LeRobotDataset / cache / 训练 / 真机推理
```

## 2. 当前代码快照

### 2.1 LeRobot 主仓库

- 路径：`/home/liusong/ProgramFiles/Huggingface/lerobot`
- 分支：`wep_vla_v0.4.3_multiview_doubleflow`
- 主 worktree 的 v34 核心实现 commit 为 `29fb401`；当前 HEAD 后续仅包含实验记录。对称双坐标 v35 位于上述独立 worktree/branch，核心 commit 为 `4562fd1`。v33 current-state body-velocity residual 为 `1b9905d`；v32 endpoint body-frame right-invariant residual 为 `d04aca7`；v31 carrier-frame left residual 为 `eb6393e`；v29 residual-rate terminal boundary 为 `f15f72e`；v28 Ego-tangent optimizer routing 为 `5d469a0`。
- 当前实验脚本、checkpoint、日志和 artifact 统一位于交接文档顶部所述 experiment root；不得删除 `/opt/data/private` 下任何文件。
- 临时实验统一使用 `/tmp/temp`，不要把临时诊断文件写入正式数据目录。

### 2.2 HandPoseExtraction 外部仓库

- 路径：`/home/liusong/ProgramFiles/HandPoseExtraction`
- 分支：`main`
- HEAD：`6f2fdc52427380d0e9bd51efc4b86f979c157663`
- 该仓库当前存在未提交改动，包括 RGB-D/MANO v5 对齐、无时序滤波 pipeline、gripper 构造和测试。
- `handpose_extraction/smoothing.py` 当前处于删除状态。
- 不允许整体 reset、checkout 或覆盖这些改动；必须先逐文件审查。
- LeRobot 工作区权限不一定允许直接写外部仓库。下一会话若要修复 HandPoseExtraction，需要显式取得该目录写权限。

## 3. 已确认结论

### 3.1 模型与 VLM adapter

- 官方 SmolVLA/SmolVLM 的视觉语言骨干结构必须保持不变，才能正确载入官方大规模图像/语言预训练权重。
- `vla_adapter_enable=true` 时，训练和推理 batch 必须包含 RGB 图像。
- `vlm_model_name` 用于确定模型架构和 processor；`vlm_weights_path` 可指向离线 SmolVLA/VLM 权重来源。
- adapter 模式通常使用：
  - `vla_adapter_enable=true`
  - `vla_adapter_freeze_vlm=true`
  - `load_vlm_weights=true`
- 冻结 VLM 不等于冻结整个策略。PointSeg、点云编码器、point-action fusion 和 Action Expert 仍可训练。
- 当前配置中 `point_action_fusion_enable=true`，动作 token 会在 Action Expert 之前与点 token 交互。
- `worldflow_enable`、`worldflow_se3_head_enable`、`se3_enable` 和 `se3_final_correction_enable` 默认均为 `false`。
- 当前常规基线在关闭 WorldFlow/SE3 时运行；不能把实验性辅助分支默认当作稳定增益。

关键文件：

- `src/lerobot/policies/smolvla/configuration_smolvla.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`
- `src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- `benchmarks/song_real_libero/scripts/train_song_benchmark.py`
- `benchmarks/song_real_libero/scripts/smolvla_model_inference.py`

### 3.2 LIBERO 数据转换

已经确认旧数据转换的核心问题不是简单的“同一个场景位置被复用”，而是每条官方 demo 都携带独立 `model_file`。如果转换时忽略它，即使 flattened simulator state 正确，柜体、把手和可移动/可配置 fixture 仍可能与原 demo 存在毫米到厘米级偏差。

当前正确约定：

- `state_observation_offset=0`；
- 输出 observation 和 source action 使用同一个 source index；
- `restore_demo_model=true`；
- `require_source_fps_match=true`；
- overhead/agentview RGB 和点云必须来自恢复后的正确 demo 环境；
- action 最终转换为训练使用的绝对末端表示；
- `action_index=0` 是正式配置，历史上的 `action_index=1` 只用于定位旧时序/执行器问题。

配置位置：

- `benchmarks/song_real_libero/configs/libero.json`
- `benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py`

当前 LIBERO 数据转换与训练/评测命令记录在：

- `benchmarks/song_real_libero/docs/experiment-records/WEPVAL_V042_GeneralDataset_ToolSeg.md`
- `benchmarks/song_real_libero/docs/experiment-records/WEP_VLA_041_DATASET_FIXED--WEPVAL_V042.md`

### 3.3 LIBERO 评测

- 默认评测必须使用官方 LIBERO 测试 episode，而不是训练数据集中的 demo episode。
- `strict_official_init=true` 是标准 benchmark 路径。
- `dataset_domain_env` 与 `dataset_domain_oracle_actions` 仅用于诊断，结果不能作为标准 benchmark 成绩。
- 当前正式 `action_index=0`。
- 夹爪推荐配置为 `delta_width_initial_sync`，只在 episode 初始同步一次内部 gripper target，之后按 chunk 内相对宽度变化执行。
- `gripper_delta_alignment=current_minus_previous` 符合轨迹时间方向；`next_minus_current` 是旧的一行提前行为。
- 标准测评不能失败后强制恢复初始状态重试。
- 并行评测时每个隔离 policy worker 应拥有独立模型和噪声状态；共享 batch 推理可能改变 flow-matching 随机轨迹及调度时序。

关键文件：

- `benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py`
- `benchmarks/song_real_libero/configs/libero.json`

### 3.4 PointSeg 与 cache

- PointSeg 目标是前景/背景软二分类，不应依赖人工语义分割、人工指定目标点或真实点流。
- 轨迹先验应覆盖：
  - 末端附近与末端共同运动的工具/被抓物体；
  - 末端逐渐靠近的交互目标；
  - 接触末期虽然运动较小、但仍位于 gripper 附近的相关物体。
- 离线 cache 与无 cache 的在线 prior 计算必须使用同一套算法。
- cache 必须同时覆盖增广点云与原始非增广全场景点云，不能用固定点数阈值暗中判断数据类型。
- `current-points`/`future-points` 小于目标数时保留全部有效点，后续由 mask 处理变长输入。
- 当前 ToolSeg/PointSeg 的生产命令以 `WEPVAL_V042_GeneralDataset_ToolSeg.md` 为准；修改参数前先核对 checkpoint 的 `train_config.json`。

关键文件：

- `benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py`
- `benchmarks/song_real_libero/scripts/train_song_benchmark.py`
- `src/lerobot/policies/smolvla/song_pointseg.py`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`

### 3.5 真实 RGB-D 与 MANO mesh 对齐

当前真实数据目录：

```text
/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/
data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking2
```

已确认：

- RealSense depth 已对齐到 color，使用的是对齐后 color camera intrinsics；
- Brown-Conrady 投影/反投影数值与 `pyrealsense2` 一致，误差不是主要来源；
- JSONL 的 `record_index` 与 RGB-D frame index 一致；
- mesh 旧版明显错位的主要原因是用投影顶点的 signed depth median 平移完整 MANO，容易被手中方块、指缝背景和局部近层误导；
- v5 使用检测框内 RGB-D 点云与完整 MANO 表面的粗到细 Chamfer 分位数搜索；
- v5 只允许一个全局平移同时作用于所有 joints/vertices，不进行逐点吸附、逐关节替换或时序轨迹滤波；
- 每个 recording segment 是独立 episode 的原始录制，任何时序状态都必须在 segment 边界 reset；
- 深度有效下界为 0.30 m。

当前正式 JSONL：

```text
.../Dynamic_Ego_CubeStacking2/handpose_wilor.jsonl
```

其状态：

- 256/256 帧版本为 `wilor_mano_mesh_rgbd_chamfer_v5`；
- 每帧都有一个有效 hand；
- 每帧都使用 `mano_mesh_rgbd_chamfer_translation`，没有降级；
- 旧 v4 备份为 `handpose_wilor.v4_global_median_backup.jsonl`；
- v5 mesh 比 v4 更符合真实深度，但在遮挡严重处仍存在 WiLoR 单目 MANO 形状/姿态误差；
- 已完成 115 帧 HDF5 smoke test，位置连续性阈值 0.05 m 下无断点；
- smoke test 输出在 `/tmp/temp/hdf5_v5_smoke_20260809`，不是生产训练数据。

关键文件：

- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/adapters/wilor.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/depth_fusion.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/gripper.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/handpose_extraction/pipeline.py`
- `/home/liusong/ProgramFiles/HandPoseExtraction/scripts/run_rgbd_sequence_wilor.py`
- `benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py`
- `benchmarks/song_real_libero/scripts/real_setting/README_CAMERA_MOTION.md`

## 4. 当前最高优先级未完成事项：episode 内旋转突变

### 4.1 已完成的初步定位

对当前 v5 `handpose_wilor.jsonl` 按独立 recording segment 使用旋转矩阵 geodesic angle 检查，结果为：

| Segment | 帧数 | 相邻旋转中位数 | P95 | 最大值 |
|---|---:|---:|---:|---:|
| 0 | 115 | 2.19° | 11.51° | 32.87° |
| 1 | 141 | 1.99° | 28.36° | 32.73° |

典型异常：

- frame `26 -> 27`：旋转约 31.09°，但 21 个 MANO joints 的中位移动仅约 1.59 mm；
- frame `117 -> 118`：旋转约 32.73°，joints 中位移动仅约 2.63 mm；
- frame `146 -> 147`：旋转约 29.81°，joints 中位移动仅约 2.79 mm。

因此可以确认：

- 旋转跳变真实存在于 HDF5 生成之前的 JSONL rotation matrix，不只是 Euler 的 `±π` 显示跳变；
- 跳变与 mesh 全局平移、相机外参或 ORB-SLAM 轨迹不是同一个问题；
- `pose_frame=camera` 时 HDF5 不应用相机外参，所以该路径不会制造上述原始旋转突变；
- `build_humanhand_hdf5_dataset.py` 的逐帧 `as_euler("zyx")` 可能另外引入 Euler 分支跳变，但不是当前 30° geodesic 跳变的根因。

### 4.2 当前最可能根因

`HandPosePipeline` 当前明确是 stateless：

```python
previous_rotation_camera_gripper=None
```

而 `pinch_gripper_from_hand` 使用多个 thumb/index finger segment 构造加权 forward/lateral axis。手指接近共线、抓住方块或 WiLoR 某个关节点轻微抖动时，候选轴的投影和权重关系会突然变化；由于没有 segment 内的方向对应关系，相邻帧可能选择不同坐标轴方向，造成 20–33° 的姿态跳变。

这不是要通过重度 EMA 掩盖的问题。正确修复应优先解决坐标系构造的唯一性和退化条件，然后才考虑是否需要非常轻量的连续性约束。

### 4.3 尚未完成的工作

1. 尚未把上述异常帧做 RGB、深度、mesh、joints、gripper axes 的并排可视化报告；
2. 尚未逐项分解 `x_up/y_right/z_forward` 哪个轴贡献了跳变；
3. 尚未实现 segment-scoped orientation correspondence；
4. 尚未添加 noisy-joint/degenerate-pinch 的回归测试；
5. 尚未重新生成修复后的完整 JSONL；
6. 尚未重新生成最终生产 HDF5、LeRobotDataset 和 cache。

## 5. 下一步行动

### P0：修复旋转突变

按以下顺序执行：

1. 在 `/tmp/temp` 写只读诊断脚本，列出每个 segment 的旋转 geodesic jump；
2. 对 top-k 异常 transition 保存：
   - 前后两帧 RGB；
   - MANO mesh/joints；
   - `x_up/y_right/z_forward`；
   - 每个候选 axis 及其权重；
   - handedness、bbox、depth correction source；
3. 判断跳变来自：
   - forward candidate 切换；
   - lateral axis 退化；
   - palm normal 符号变化；
   - WiLoR articulation 本身突变；
4. 优先采用通用、非滤波式修复：
   - 使用稳定 wrist/MCP palm frame 作为主姿态；
   - thumb/index 只决定 opening 和局部夹取方向，不让退化指尖主导完整姿态；
   - 对等价坐标轴表示做 segment 内最小 geodesic 对应；
   - 只有在当前帧几何退化时才使用上一帧方向消歧，不做位置/旋转 EMA；
   - segment 开始必须 reset；
5. 先用当前 256 帧离线重算并比较修复前后；
6. 目标验收建议：
   - 没有视觉真实大转动时，相邻旋转不应出现 20–33° 单帧跳变；
   - 修复不能改变 mesh/joints 的 3D 几何或全局位置；
   - segment 边界不纳入连续性约束；
7. 最后处理 Euler 存储：保留 quaternion/matrix 为真值；如果下游必须使用 Euler，则按 episode 做连续 branch unwrap，但不能用 unwrap 掩盖真实 matrix jump。

### P1：重新生成真实训练数据

P0 完成后：

1. 不使用 `--reuse-jsonl` 重新生成 `handpose_wilor.jsonl`；
2. 验证 pipeline version、frame count、无缺失 hand；
3. 按 `segments.txt` 生成最终 HDF5；
4. 检查 position、rotation geodesic、opening width、Euler branch 和相机轨迹；
5. 再生成 LeRobotDataset 与 PointSeg cache；
6. 旧 dataset/cache 不应与新 HDF5 混用。

### P2：剩余工程事项

- 解决当前环境 SciPy 对 NumPy `1.23.4` 的版本警告，建议升级到满足 SciPy 要求的 `>=1.23.5`，但变更环境前应锁定依赖并重跑测试；
- 将 HandPoseExtraction 当前未提交改动整理为小粒度 commit；
- 补充完整真实数据质量报告，不只检查位置 jump；
- 对 WorldFlow/SE3 分支继续保持默认关闭，除非单独消融实验确认收益。

## 6. 常用命令

### 6.1 查看当前 v5 JSONL

```bash
python /home/liusong/ProgramFiles/HandPoseExtraction/scripts/run_rgbd_sequence_wilor.py \
  --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
  --input /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking2 \
  --jsonl /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/rgbd_records/Dynamic_Ego_CubeStacking2/handpose_wilor.jsonl \
  --reuse-jsonl \
  --show3d
```

注意：`--reuse-jsonl` 只查看已有结果，不运行任何新算法。

### 6.2 重新推理手部姿态

```bash
python /home/liusong/ProgramFiles/HandPoseExtraction/scripts/run_rgbd_sequence_wilor.py \
  --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
  --input "$DYNAMIC_OUTPUT_DIR" \
  --jsonl "$DYNAMIC_OUTPUT_DIR/handpose_wilor.jsonl" \
  --fast \
  --force-handedness right \
  --fusion-mode model-depth \
  --min-depth-m 0.30 \
  --max-depth-m 3.0 \
  --depth-window 5 \
  --min-valid-depth-points 6 \
  --include-mesh
```

不要在旋转修复完成前覆盖唯一结果；先输出到临时 JSONL 比较。

### 6.3 HDF5 连续性检查

```bash
python benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py \
  --hdf5_dir /path/to/hdf5_dir \
  --threshold 0.05
```

该脚本当前主要检查位置；旋转 geodesic 质量检查仍需补充。

### 6.4 回归测试

HandPoseExtraction：

```bash
cd /home/liusong/ProgramFiles/HandPoseExtraction
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/liusong/anaconda3/envs/reap/bin/python -m pytest \
  tests/test_geometry.py tests/test_offline_sequence.py -q
```

LeRobot：

```bash
cd /home/liusong/ProgramFiles/Huggingface/lerobot
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/liusong/anaconda3/envs/reap/bin/python -m pytest \
  tests/scripts/test_real_camera_motion.py -q
```

最近通过结果：HandPose 26 项，LeRobot real-camera-motion 22 项。

## 6.5 2026-08-14：LIBERO 单视角 WorldFlow 当前运行状态

- 当前同协议固定动作 baseline 是 `472/500 = 94.4%`；严格通过线是至少 `473/500`，并且同 checkpoint 的完整 World 因果消融必须更差。历史 `481/500` 仅作参考，不能替代当前同协议 baseline。
- v36 已完成并被 Broad 筛除：step `100/240/480/720` 的保留结果分别为 `85/91`（6 失败）、`56/67`（11 失败）、`75/80`（5 失败）、`82/87`（5 失败）；各自的理论最高值均不超过 `95/100`，没有启动 causal 或 Full-500。
- 当前独立代码工作树是 `/home/liusong/ProgramFiles/Huggingface/lerobot_worldflow_parallel_dualcoord`，分支 `wep_vla_v0.4.3_worldflow_parallel_dualcoord`，提交 `a04fc510cf35e61369c4ae3fc656a507c9ed9e00`。v37 使用两个原长度 action suffix、两个 point-action adapter、一个共享 point-action expert，并在进入 expert 前进行双向 Ego/World cross-attention；不含门控、task-specific 技巧、物理 residual vote 或 LitePT 修改。完整 WorldFlow/SE(3) 测试通过 `86/86`。
- v36 的 50-step World-flow 滑动均值从 steps 1--50 的约 `0.0451` 持续下降到 steps 671--720 的约 `0.0113`，720 步仍不能视为明确收敛。因此 v37 训练长度设为 `2160` optimizer steps（约 3 epochs），而不是在 720 步强制结束。checkpoint 为 `100/240/480/720/1080/1440/1800/2160`，需同时参考 W&B 损失趋势和 rollout，不能仅凭早期 checkpoint 排除后期收敛。
- 物理 batch48/GPU 在新的并行双坐标拓扑中会在 optimizer state 建立后 OOM。当前有效配置改为 microbatch24/GPU、gradient accumulation2、4 GPUs，保持有效单卡 batch48、全局 batch192；worker12、W&B enabled。激活检查点只用于训练共享 expert，推理函数不变。
- 当前正式运行 variant：`v37r3_baseline_bootstrap_parallel_dualcoord_global_world_coprediction_equaljoint_4gpu_mb24_ga2_effb48_w12_2160steps`。tmux：`wep_v043_v37r3_parallel_dualcoord_train`、`wep_v043_v37r3_parallel_dualcoord_broad_after_train`、`wep_v043_v37r3_parallel_dualcoord_full500_after_broad`。W&B：`https://wandb.ai/liusong-scrat/wep_vla_v043_libero10/runs/worldflow-v37r3-parallel-dualcoord-global-world-coprediction-3epoch-20260814`。
- v37r3 已越过此前 step2 OOM 点并稳定进入训练循环；初期 loss 有正常 batch 波动，未出现 OOM、Traceback 或非有限值。此前 r0/r1/r2 的失败输出与日志均保留作审计。没有删除 `/opt/data/private` 下的任何文件。

## 6.6 2026-08-15：v37 最终结论与 v38r1 canonical 双流（覆盖 6.5 的“当前”状态）

- 权威协议不变：Ego-only matched fixed-action baseline 为 `472/500 = 94.4%`，WorldFlow 必须至少 `473/500`，并且同一 checkpoint 的完整 World→Ego 因果消融必须更差。固定分层 Broad baseline 为 `95/100`，候选必须严格高于95才进入因果和 Full500。
- v37r3 已完成2160步、约三个 task-balanced epochs 并正式否决。最终 last720 的 World/Ego 变化仅 `-1.74%/+0.32%`，共享 Expert 从1080到2160基本不动，属于 epoch-scale plateau，不是训练不足。八个 Broad checkpoint `100/240/480/720/1080/1440/1800/2160` 均在第五个失败或更早失去严格超过95的可能；没有启动 causal 或 Full500。完整 screen 为 experiment root 下 `singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v37r3_parallel_dualcoord_global_world_coprediction_stratified_checkpoint_causal_screen.json`。
- 当前代码改为 v38r1 canonical World-conditioned dual flow。独立 worktree `/home/liusong/ProgramFiles/Huggingface/lerobot_worldflow_canonical_dualflow`，branch `wep_vla_v0.4.3_worldflow_canonical_dualflow`，commit `62e8670537ba72339c762ba954c528646757eb5f`；configuration/model/trainer SHA256 分别为 `e671316cdbe84b19d2ea2f5a3d7552398d2c42f4271455a981ff2585804ef0a2`、`4df699b348891bbf1ac2e73d814258ed728795ccc6479fced42ae293c7b30431`、`abeb34735bbc3e873733d8ea369905a1015a08f60ed206fb979e5c88dfe16c61`。
- v38r1 保持两个独立 point-action adapter、两个 action head 和一个共享 point-action Action Expert。World 场景使用绝对 global 点云；World 前端将场景、语言和当前 carrier 编码成完整 token-space residual，叠加到同一 canonical action state 的 Ego token。World/Ego 在进入共享 Expert 前双向交换。canonical residual、carrier projection 和两条 cross-attention output projection 均为完整矩阵、exact-zero bootstrap、随后共同训练；不存在 scalar/learned gate、冻结 Ego、task rule、World geo/bridge/equiv auxiliary loss、SE3 或 LitePT 修改。
- 两步4-GPU smoke 已证明 baseline-preserving 起点：step1 Ego/World flow loss 均为 `0.0002622`，step2 才轻微分化为 `0.00008427/0.00008520`；所有双流模块及 Ego/shared 路径均获得更新。SmolVLA WorldFlow/SE3 tests `91/91`、dataset tests `12/12` 通过。
- 正式训练 variant 为 `v38r1_canonical_world_token_residual_dualflow_4gpu_mb24_ga2_effb48_w12_2160steps`，tmux `wep_v043_v38r1_canonical_dualflow_train`。配置：4 GPUs、microbatch24/GPU、accumulation2、effective batch48/GPU、global batch192、worker12、W&B enabled、完整单视角10-task×50-episode数据、2160 updates，保存 `100/240/480/720/1080/1440/1800/2160`。
- 首个完整 step720 epoch 的独立窗口 `481--720` Ego/World loss 为 `0.00018501/0.00018376`；整个720-step长窗口前后半仅变化 `+0.47%/+0.85%`，状态 `no_material_tail_change_detected`。step480 的短尾回升已恢复，没有出现 v37 式 World下降而Ego持续恶化。World双向交互/residual/carrier继续增长，共享 Expert layer0 drift 仅 `0.004234`。这只是训练充分性证据，不是 rollout 通过证据；训练仍完成三 epochs。
- 收敛 monitor 为 `wep_v043_v38r1_convergence_monitor`。Broad waiter `wep_v043_v38r1_broad_after_convergence` 会先审计最终 short200 与 epoch720 窗口；若 epoch-scale World loss 仍显著下降，则写入 `v38r1_extension_required_before_broad.json`、不启动 rollout，并由 `wep_v043_v38r1_conditional_resume_to2880` 精确恢复 model/optimizer/scheduler/RNG 至第四个 epoch。若已平台化则直接筛查全部 checkpoints。
- Full waiter 为 `wep_v043_v38r1_full500_after_broad`。评测固定 action/noise/cache、local4 total30、inference batch30；先要求 Broad `>95/100`，再要求同 checkpoint 完整 World→Ego causal delta 为正，最后要求 Full500 `>=473/500`。在 WorldFlow 通过之前，多视角继续封存。
- 所有运行文件统一位于 `/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811`；没有删除 `/opt/data/private` 下任何文件。

## 6.7 2026-08-15：v38r1 按用户要求停止训练并直接筛查现有权重（覆盖 6.6 的运行状态）

- 用户明确要求停止继续训练，在现有权重上测试；若效果没有改善，则该方案直接结束，不再以增加训练步数为理由继续。v38r1 训练、收敛 monitor、条件续训和旧的 step2160 waiter 均已停止；不会启动 step1080 之后的恢复训练。
- 完整且可评测的 checkpoint 只有 `100/240/480/720/1080`。训练进程实际在约 step1102 被终止，但不存在完整 step1102 checkpoint，因此它不进入任何性能判断；所有已生成权重、optimizer state、日志和 cache 均保留。
- 截止 step1080 的截断收敛审计完整解析1080个 update。最近200步 Ego/World loss 前后半分别变化 `+2.30%/+2.13%`，最近720步分别变化 `+1.84%/+1.09%`；World loss 在短、长尺度均没有继续实质下降，状态为 `no_material_tail_change_detected`。因此现有权重若 rollout 不改善，不能再归因于日志显示的明显未收敛。artifact：`singleview_worldflow/libero10_500ep/artifacts/v38r1_training_convergence_stopped_step001080_audit.json`。
- 当前 Broad tmux 为 `wep_v043_v38r1_stopped1080_broad`，按 `100/240/480/720/1080` 顺序测试；Full 条件 waiter 为 `wep_v043_v38r1_stopped1080_full500`。Broad 仍使用固定 IDs `0,5,...,45`、fixed-action/fixed-noise/canonical exact cache、local4 total30、inference batch30；累计5个失败即数学早停。只有正常臂严格超过 `95/100` 且同 checkpoint 完整 World-to-Ego causal delta 为正，才进入权威 Full500；Full500 baseline 为 `472/500`，通过线为至少 `473/500`。
- 新入口为实验根目录 `scripts/wait_then_eval_v38r1_stopped_step1080_local4.sh`。Broad日志为 `singleview_worldflow/libero10_500ep/logs/eval_v38r1_stopped_step1080_broad.log`；Full waiter日志为 `singleview_worldflow/libero10_500ep/logs/eval_v38r1_stopped_step1080_full500_waiter.log`。
- v38r1 最终筛查已完成并正式否决。steps `100/240/480/720/1080` 分别在 `74/79`、`89/94`、`90/95`、`93/98`、`90/95` 时累计第5个失败；每个 checkpoint 的最大可能最终成绩都只有 `95/100`，不能严格超过 Broad baseline。没有候选，因此没有启动 World-to-Ego causal arm 或 Full500，条件 Full waiter 写出 `screened_out` 后正常退出。聚合 screen：`singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v38r1_canonical_world_token_residual_dualflow_stratified_checkpoint_causal_screen.json`；正式 gate：`singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v38r1_canonical_world_token_residual_dualflow_local4_fixedbarrier_all_candidates_full500_matched_worldflow_gate.json`。
- 结论：现有 v38r1 权重未显示性能改善，而且 step1080 前的长窗口 loss 已无实质下降；按用户决定结束该方案，不继续训练。当前无 v38r1 训练或评测 tmux/进程，4张本地GPU已释放；所有输出均保留。

## 6.8 2026-08-15：v39 shared-state dual-adapter 仅测现有权重后停止

- v39 位于独立 worktree `/home/liusong/ProgramFiles/Huggingface/lerobot_worldflow_shared_state_dual_adapter`，branch `wep_vla_v0.4.3_worldflow_shared_state_dual_adapter`，commit `507ffc575067e706e85f7b276662c8dd96a3c0d2`。它保留两个 point-action adapter 和一个共享 point-action expert，但只使用一个 canonical Ego action state、一次共享 expert forward 和原 Ego action head；World adapter 以 Ego-conditioned action token 为 query，从 global World 点云和当前 carrier 生成完整 token residual，再写回同一 Ego token。不存在第二 action target/head、门控、World auxiliary loss、task rule、SE3 或 LitePT 修改；关闭 WorldFlow 的旧路径不变。
- 完整 SmolVLA WorldFlow/SE3 与 dataset 回归测试通过 `108/108`。4-GPU batch48/GPU、worker12 的两步 smoke 完成且无 OOM；step1/2 action loss 为 `0.0002378/0.0001936`。参数审计证明 World residual、World upstream、Ego adapter 和共享 expert 均发生更新，而未参与计算的旧 World head/cross-attention 参数保持完全不变。
- 用户随后要求停止继续训练，只测试现有权重。此时不存在正式训练 checkpoint，只有 smoke `step2`：累计仅见过384个样本，即约 `0.28%` epoch。正式720-step任务均衡训练没有启动，也不会续跑。
- 对现有 step2 使用固定分层 IDs `0,5,...,45`、fixed-barrier total30/inference batch30、policy-noise seed0 进行 Broad 筛查。终止时为 `91/98`、7个失败；即使剩余2个全部成功，最高也只有 `93/100`，低于同协议 baseline `95/100`，因此数学提前淘汰。没有启动 causal ablation 或 Full500。artifact：`singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v39_shared_state_dual_adapter_existing_step000002_4gpu_total30_b30_alltasks10ep_codefbfacd7_fixedbarrierv18_stratified_step5_v39_existingonly.json`。
- 按用户的 existing-weight-only 决策，v39 运行路线结束，不再训练。科学解释上，这不是“充分训练后结构无效”的收敛结论，因为被测权重仅训练2步；它只证明现有权重没有改善。当前没有 v39 训练或评测进程，4张本地GPU均空闲；所有 checkpoint、cache、progress、日志和 artifact 均保留。

## 6.9 2026-08-15：v32 按当前 baseline 纠错复核并通过 WorldFlow 联合门禁

- 历史 v32 step100 Full-500 在 `386/405` 时因沿用旧协议 `481/500` 门槛被提前停止；当前权威同协议 baseline 已明确为 `472/500 = 94.4%`，所以该历史否决无效。复核使用独立只读历史代码工作树 `/home/liusong/ProgramFiles/Huggingface/lerobot_worldflow_v32_eval`，branch `wep_vla_v0.4.3_worldflow_v32_eval_recheck`，commit `dba65f7e0466364dcb43eaa016a7d7fb3c05c90d`。评测器 SHA256 为 `7229c3047aa488c1df8b236542f31df0726ea702d102958eff173ddeefc3bd28`，与当前 fixed-barrier-v18 确定性评测器逐字节一致。
- 确定性协议为本地4卡、30个固定 worker slot、inference batch30、policy-noise seed0、每 suite/task/episode/model-call 显式噪声种子，以及包含 observation/instruction/seed 的 canonical exact-action cache。相同 canonical cache 的重放可保证完全相同的 action；新建 CUDA canonical rollout 受固定 seed/slot 约束，但严格逐值复现仍以复用 exact cache 为准。
- v32 step100 正常 Full-500 最终为 `475/500 = 95.0%`，超过同协议 Ego-only baseline `472/500` 三个成功；同 checkpoint、同 episode、同 seed、同 fixed-barrier 布局下关闭 World-to-Ego 执行作用的完整因果消融为 `469/500 = 93.8%`。World 分支净因果贡献为 `+6`，所以 `normal > baseline` 且 `normal > causal` 的联合门禁正式通过。
- 正常 artifact：`singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v32_residual_rate_coordframe_bodyframe_ego_tangent_p75_formal_step000100_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_normal_baseline472_recheck.json`；因果 artifact：`singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v32_residual_rate_coordframe_bodyframe_ego_tangent_p75_formal_causal_step000100_4gpu_total30_b30_alltasks50ep_codefbfacd7_fixedbarrierv18_worldtoegoablated_causal_baseline472_recheck.json`；联合 gate：`singleview_worldflow/libero10_500ep/artifacts/taskbalanced_v32_residual_rate_coordframe_bodyframe_ego_tangent_p75_local4_fixedbarrier_full500_matched_worldflow_gate_step000100_baseline472_recheck.json`。
- v32 保留两个 point-action adapter 与一个共享 point-action expert。Ego action block 保持预训练位置；World scene/action block 在其后读取 Ego，expert 后由 Ego-to-World cross-attention 条件化 World，并用 body-frame endpoint residual 返回 Ego。每个去噪时刻都由当前 Ego flow state 通过 SE(3) 共轭得到 World state（`projected_ego_path`），因此二者描述同一物理轨迹，不是两个无约束漂移的动作。它没有 learned gate，但历史训练包含 `p=0.75` World-to-Ego stochastic depth、residual anchor、shared-gradient Ego-tangent projection 和训练期坐标系增强；因此它是“Ego 主流 + 物理共轭 World residual 流”的非对称双流，不是两个完全平权、独立积分的对称双流。
- WorldFlow 单视角阶段现已达到目标，可以按既定消融顺序解除多视角封存。下一阶段只在输入层恢复通用多视角融合，模型输入仍严格为10,000点；先证明 multiview 对该通过门禁的 v32 不降或提升，再验证最终 WorldFlow+multiview 中两项贡献均为正。所有输出与 cache 均保留，未删除 `/opt/data/private` 下任何文件。

## 6.10 2026-08-15：V51 完整执行后 Broad 否决

- V51 可执行 worktree 为 `/home/liusong/ProgramFiles/Huggingface/lerobot_v51`，branch `wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation`，最终可执行 commit `508af939a204c8f3a42eccf0fa7f187ea9a0f6a4`。workflow archive 为 `/home/liusong/ProgramFiles/Huggingface/lerobot_v51_workflow_archive`，branch `wep_vla_v0.4.3_v51_workflow_archive`。三组回归、4-sample smoke、`137590` samples/36 shards v12 cache、36-shard online/cache exact-index 审计、完整 paired epoch 1564-step 训练、optimizer/gradient-role 和 parameter-drift 审计均已完成；这些仅证明执行与机制正确，不是性能通过证据。
- child-env CUDA/EGL 修复后的权威 Broad namespace 是 `v51r1childenvfix`。六个 checkpoints `260/520/780/1040/1300/1564` 的双视角+World 成绩依次为：`76/81`（5 failures，最大95）、`90/95`（5，最大95）、`95/100`、`94/99`（5，最大95）、`91/97`（6，最大94）、`90/95`（5，最大95）。没有任何 checkpoint 严格超过 Broad baseline `95/100`，因此没有候选，也没有启动 Full500。
- 聚合 Broad artifact：`joint_multiview_worldflow/libero10_500ep/artifacts/v51r1childenvfix_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json`。正式 Full waiter artifact：`joint_multiview_worldflow/libero10_500ep/artifacts/v51r1childenvfix_all_candidates_full500_2x2_multiview_worldflow_causal_gate.json`，状态为 `screened_out`。V51 已正式归档为负结果，不能再把它描述为“尚未运行”。
- V51 的副视角进入量平均只有约53点，且重叠区域仍始终保留主视角首个数组点，输入表示明显偏向主视角。失败 episode 会随 checkpoint 漂移，task5/episode0 与 task3/episode25 有重复，但没有证据支持只对特定 task/episode 做修补。

## 6.11 2026-08-15：V52 1 cm voxel consensus + 4 cm persistent novelty（当前运行状态）

- V52 是纯输入级几何表示修改：每个主视角占据的1 cm cell，在主/副视角联合 cell 内选择离联合 centroid 最近的真实输入 medoid；主视角未覆盖的区域，仍按4 cm coarse cell 接纳一个副视角真实 medoid。输出严格为原始索引、10,000点、唯一索引、主视角1 cm cell 全覆盖、gripper 500点 exact；无合成点、固定视角 quota、局部最近点一一交换、遮挡/语义/task trick、learned gate 或 LitePT 修改。fusion 名称为 `consensus_multiscale_novelty_union`。
- 可执行 worktree：`/home/liusong/ProgramFiles/Huggingface/lerobot_v52_consensus_multiscale`；branch `wep_vla_v0.4.3_v52_consensus_multiscale`；固定 commit `9447a43a0e2ed3130a578618ac92225f71eb8a31`，已推送且工作树 clean。完整回归为 `137 passed, 7 warnings`。30个真实样本几何 preflight 中，旧 scale4 副视角点平均约 `82.77`，V52 总副视角 medoid 平均约 `528.07`，其中 overlap consensus medoid 平均约 `445.30`；所有样本均满足10k/unique/主视角细 cell 覆盖和副视角几何 contract。
- workflow archive：`/home/liusong/ProgramFiles/Huggingface/lerobot_v52_workflow_archive`；branch `wep_vla_v0.4.3_v52_workflow_archive`；当前 commit `179fd1a8a52345ea3d270e37a55b978a2de73d5e`。4-sample smoke 及其4个 smoke shards exact-index audit 均通过：10k、unique、primary fine coverage、secondary geometry、gripper exact、online/cache exact match 全为真。真实 V32 checkpoint optimizer preflight 也通过：1243个 trainable tensors，四组 tensor 数 `153/1063/25/2`，LR 为 `5e-9/5e-8/5e-9/5e-9`，Ego/World point roles 均进入相应训练组。最终 Full500 gate 还会逐臂核验 exact-action cache v2 manifest、共同 policy identity、固定 seed/batching 和预期 camera/causal runtime context。
- 正式 cache 正在 tmux `wep_v043_v52_consensus_cache_audit` 生成到 `joint_multiview_worldflow/libero10_500ep/pointseg_cache_consensus_multiscale_novelty_union_1cm4cm`。`wep_v043_v52_audit_then_paired_train` 会等待 cache manifest 和所有 cache worker 退出，随后执行36-shard exact-index 审计；仅审计通过才启动完整 paired 1564-step 训练。正式 audit artifact 目标为 `joint_multiview_worldflow/libero10_500ep/artifacts/v52_cache_online_exact_index_audit_36shards.json`；训练目录为 `joint_multiview_worldflow/libero10_500ep/training/v52_from_v32step100_consensus_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_4gpu_b44_w12_1564steps`。
- 后续 tmux 已预注册但当前仅等待：`wep_v043_v52_convergence_monitor`、`wep_v043_v52childenvfix_all_checkpoints_stratified_3arm`、`wep_v043_v52childenvfix_all_candidates_full500_2x2_gate`。它们依次要求完整训练收敛/参数漂移证据、所有 six-checkpoint Broad 三臂筛查，以及仅对候选执行同 checkpoint/episode/seed/fixed-barrier canonical exact-action-cache 的 Full500 2×2 因果门禁。Broad aggregate 目标为 `v52childenvfix_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json`；Full aggregate 目标为 `v52childenvfix_all_candidates_full500_2x2_multiview_worldflow_causal_gate.json`。
- 当前 V52 尚无正式 cache manifest、训练 checkpoint、Broad 或 Full500 结果，因此不得宣称性能完成。最终门禁仍是双视角+World `>472/500` 且 `>=475/500`，并严格高于同 checkpoint 主视角+World和双视角+World→Ego-disabled；只有完整2×2通过后才能完成 `/goal`。

## 6.12 2026-08-15：V52 正式 cache/audit 通过并启动 paired 训练（覆盖 6.11 的运行状态）

- 正式 cache 已由四个 ranks 全部完成并发布 manifest：version `12`、`137590` samples、`36` shards、fusion `consensus_multiscale_novelty_union`、fine voxel `0.01 m`、coarse novelty scale `4.0`、current/future 各 `10000` 点、gripper `500` 点。cache tmux 在所有 rank done markers 收齐后正常退出，没有手工合并或绕过分布式完成协议。
- 权威36-shard exact-index audit 已通过并写入 `joint_multiview_worldflow/libero10_500ep/artifacts/v52_cache_online_exact_index_audit_36shards.json`。它从每个 shard 取一个 midpoint，共审计36个真实样本；online/cache exact index、10k、unique、primary fine-cell coverage、secondary geometry contract、gripper exact 全为真。副视角插入数量 min/max/mean 为 `208/1273/544.31`；其中 overlap consensus medoids 为 `128/952/462.47`，coarse-novel medoids 为 `7/914/81.83`。
- 真实 V32 checkpoint optimizer/gradient-role artifact 已正式写入 `joint_multiview_worldflow/libero10_500ep/artifacts/v52_real_checkpoint_optimizer_and_gradient_role_preflight.json`，状态 `passes_v52_real_checkpoint_optimizer_preflight=true`。1243个 trainable tensors 无重叠或遗漏；四组分别为153 tensors@`5e-9`、1063@`5e-8`、25@`5e-9`、2@`5e-9`；Ego point roles 829 tensors，World-only point roles 234 tensors，对称 point-path adaptation 和 Ego-tangent 保护边界均通过。
- 完整 paired 训练已由原 tmux `wep_v043_v52_audit_then_paired_train` 自动启动，未重复创建 output。训练进程固定 executable commit `9447a43a0e2ed3130a578618ac92225f71eb8a31`，4 GPUs、batch44/GPU、global176、1564 steps、完整 paired epoch、checkpoint `260/520/780/1040/1300/1564`；进程环境为 `WANDB_MODE=online`。启动后已通过 step1--11，无 OOM、NaN 或 traceback，四卡显存约21--22 GB。训练日志：`joint_multiview_worldflow/libero10_500ep/logs/train_v52_from_v32step100_consensus_multiscale1cm4cm_paired_symmetricpoint5e8_policy5e9_4gpu_b44_w12_1564steps.log`。
- 收敛/漂移 monitor、six-checkpoint Broad 和候选 Full500 2×2 waiters 继续存活。当前仍没有 Broad 或 Full500 性能证据，不能标记 `/goal` 完成。

## 6.13 2026-08-15：V52 完整训练、首个 Broad 候选与主分支运行时合入（覆盖 6.12 的运行状态）

- V52 paired 训练已完整完成 `1564/1564` steps 和一个 task-balanced paired epoch，六个 checkpoint `260/520/780/1040/1300/1564` 均已原子发布。最终 `v52_training_convergence_complete.json` 为 `training_complete=true`、`maximum_parsed_step=1564`，状态 `short_tail_decline_epoch_scale_plateau`；最终参数漂移审计证明 exact architecture key match、四个 optimizer groups、全部 architecture roles、Ego/World 对称 point path 和 optimizer state 全部通过。训练与 monitor tmux 均正常退出，无 OOM、NaN 或 traceback。
- child-env-fixed Broad 使用固定 IDs `0,5,...,45`、fixed-barrier-v18、policy-noise seed0、每臂100 episodes。step260 三臂为 dual+World `96`、primary+World `93`、dual+World-to-Ego-disabled `96`，因 WorldFlow delta `0` 被筛除。step520 三臂为 `98/95/96`，多视角 delta `+3`、WorldFlow delta `+2`，`passes_joint_screen=true`，是首个正式 Full500 候选。step780 在 `72/77` 时累计5失败、理论最高95；step1040 在 `75/80` 时累计5失败、理论最高95；二者均数学早停且未运行消融。step1300/1564 继续由原 Broad waiter 顺序筛查；Full500 waiter 必须等待六点聚合后按预注册排名执行，当前不得宣称最终门禁通过。
- 按用户要求，V52 executable 历史已通过 merge commit `664f14e` 合入 `wep_vla_v0.4.3_multiview_doubleflow`，同时保留主分支已有的默认关闭 WorldFlow 能力和全部较新 handoff。合并后的主分支可从自身 `src/lerobot` 直接解析 step520 的 `camera_view_coarse_novelty_scale=4.0`、`multiview_input_symmetric_point_path_adaptation=true` 并导入 `consensus_multiscale_novelty_union_sample_fused_point_cloud`；配置/导入 preflight 通过，四个正式测试文件回归为 `144 passed, 8 warnings`。这只证明运行时兼容与回归正确，不替代剩余 Broad 或 Full500 性能门禁。

## 6.14 2026-08-15：V52 six-checkpoint Broad 完成，step520 启动 Full500（覆盖 6.13 的运行状态）

- 按用户明确要求，Broad 编排改为两阶段：先完成全部 checkpoint 的 dual+World 100-episode 初筛，再仅对严格 `>95/100` 的 checkpoint 逐步运行 primary+World 和 dual+World-to-Ego-disabled。workflow archive 已推送 `4216d49e556a1109bb2305e2fd46a011534b4528`；阶段二为保留被中断的 step1300 primary 部分输出使用新 namespace。初版长标签触发 Linux `NAME_MAX`，四个 worker 在创建日志前退出、没有产生评测结果；随后以短标签 `s5p2` 修复并推送 `8fcf983`。旧 step1300 primary 部分目录保留在 `65/100`，没有覆盖或删除。
- six-checkpoint dual 初筛及候选消融已全部完成。step260 为 `96/93/96`，WorldFlow delta `0`，筛除；step520 为 `98/95/96`，multiview delta `+3`、WorldFlow delta `+2`，唯一联合候选；step780 在 `72/77`、step1040 在 `75/80`、step1564 在 `93/98` 时达到第5个失败，理论最高均为95，数学早停且未运行消融；step1300 dual/primary 为 `96/96`，multiview delta `0`，筛除且未运行 disabled arm。
- 权威聚合 Broad artifact 为 `joint_multiview_worldflow/libero10_500ep/artifacts/v52childenvfix_all_checkpoints_stratified_3arm_multiview_worldflow_screen.json`，`candidate_steps_ranked=[520]`。所有淘汰 checkpoint、被中断的 eager primary、数学早停 progress、cache 和日志均保留。
- step520 已通过 Full500 2×2 preflight，并由 tmux `wep_v043_v52childenvfix_all_candidates_full500_2x2_gate` 启动第一臂 dual+World 500 episodes。协议仍为同 checkpoint、episodes `0..49`、policy-noise seed0、fixed-barrier-v18、local4 total30/batch30 和 canonical exact-action-cache v2；只有 dual+World 达到至少 `475/500` 才会逐步运行 primary、dual-disabled 和 primary-disabled。当前尚无 Full500 结果，绝不能标记最终目标完成。

## 6.15 2026-08-16：V52 step520 Full500 数学早停，A800 Full500 交叉复核（覆盖 6.14 的运行状态）

- 本地权威 step520 dual+World 第一臂在完成 `487/500` 时得到 `461` successes、`26` failures；剩余13个即使全成功也最多 `474/500`，低于预注册的 `475/500` 非退化线，因此 workflow 以 `mathematical_early_stop_dual_below_v32_nondegradation` 正常停止。只启动了 dual+World；primary+World、dual+World-to-Ego-disabled 和 primary-disabled 均未启动，不能把缺失消融解释为通过。step gate 为 `joint_multiview_worldflow/libero10_500ep/artifacts/v52childenvfix_step000520_full500_2x2_multiview_worldflow_causal_gate.json`，aggregate 为 `v52childenvfix_all_candidates_full500_2x2_multiview_worldflow_causal_gate.json`，状态 `failed_all_ranked_candidates`；全部 partial、cache 和日志保留。
- 对同一487个已完成 episode 与权威 V32 做逐状态对齐：V52 与 V32共同成功441、共同失败5，V52恢复20个V32失败但新增21个失败，净变化 `-1`；V32在尚未运行的13个task8 episode上全部成功，所以V52的数学上限恰好是474。该结果说明1 cm overlap consensus能够移动失败边界，但没有产生净稳健性。
- 用户上传的单卡 A800 Full500 附件已核验：step520、dual pointcloud、WorldFlow enabled、strict official init、seed0/7、1000 steps、fixed_barrier，结果为 `462/500 = 92.4%`；task8=`38/50`、task9=`42/50` 是主要失败来源。它实际使用10×8=`80`个环境和batch80，cache目录名却含`total40_b40`，所以不是预注册的A800 total40/batch40，也不能替代本地local4 total30/batch30门禁。代码哈希与V52 executable一致；A800相对V32为恢复15个、引入28个，净 `-13`。A800与本地在共同487状态上仍有20个local-only failures和30个A800-only failures，证明fixed_barrier只稳定给定布局，不保证跨硬件/批布局动作相同。轻量交叉审计为 `joint_multiview_worldflow/libero10_500ep/artifacts/v52_step000520_a800_gpu5_b80_full500_cross_hardware_audit.json`；9 MB原始summary不进Git。
- V52 已正式否决，`/goal` 仍未完成。下一轮必须从上述逐状态净交换、task6/8/9退化和Broad抽样假阳性中寻找通用输入表示或双流负交互原因；不得继续对V52运行未通过前置门禁的Full500消融，也不得把A800 batch80结果包装成权威协议结果。

## 7. 禁止事项与设计边界

- 不引入人工分割、人工指定目标点或人工光流监督；
- 不使用 PCA 等依赖场景分布、容易退化的几何捷径；
- 不逐点把 MANO mesh 吸附到 RGB-D 表面；
- 不用重度滤波掩盖错误坐标变换或错误姿态定义；
- 不跨 recording segment 继承任何时序状态；
- 不修改第三方 WiLoR/LitePT 源码来绕过本项目适配问题；
- 不整体覆盖或 reset 当前脏的 HandPoseExtraction 工作树；
- 不将 dataset-domain diagnostic 成绩当作官方 LIBERO benchmark 成绩。

## 8. 新会话启动提示词

在新对话中直接发送以下内容：

```text
读取并严格沿用：
/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/docs/experiment-records/PROJECT_HANDOFF_2026-08-11.md

不要询问我，直接执行其中 P0“修复 episode 内旋转突变”。先在 /tmp/temp 生成只读诊断，定位当前 Dynamic_Ego_CubeStacking2 中 20–33° 单帧旋转跳变究竟来自 x_up、y_right 还是 z_forward。修复必须通用、segment-scoped，不使用 PCA、人工监督、逐点 mesh 吸附或重度 EMA。修复后重新生成临时 JSONL，与现有 v5 做定量及可视化对比；通过测试后再更新正式文件。不要覆盖或 reset HandPoseExtraction 中其他未提交改动。
```
