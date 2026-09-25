# WEP-VLA 容器迁移恢复入口（2026-08-15）

当前代码、历史结构分支和 V51 可复现运行包已推送到 `git@github.com:LiuSong-Scrat/Lerobot.git`。

新容器可直接粘贴的完整 `/goal` 与 `~/.codex/config.toml` 权限配置见：`benchmarks/song_real_libero/docs/experiment-records/GOAL_RESUME_MULTIVIEW_WORLDFLOW_2026-08-15.md`。

## 2026-08-16 权威状态覆盖

- 下方最短流程和V51状态保留为迁移历史；继续工作时以本段、Goal文档continuation和handoff 6.15为准。
- 当前主分支为 `wep_vla_v0.4.3_multiview_doubleflow`；V52 executable 固定为 `wep_vla_v0.4.3_v52_consensus_multiscale@9447a43a0e2ed3130a578618ac92225f71eb8a31`，workflow archive 固定为 `wep_vla_v0.4.3_v52_workflow_archive@8fcf983`。
- V51与V52的cache、审计、完整paired训练和Broad均已完成。V52唯一候选step520在本地权威Full500 dual+World `461/487`、26 failures、理论最高474时数学早停；其余Full500消融臂按门禁未启动。A800 batch80交叉复核为`462/500=92.4%`。V52已否决，不能重复启动旧输出，也不能宣称Goal完成。
- 现有dataset、cache、checkpoint、partial rollout和失败artifact全部保留；恢复后应先读取实时Git/tmux/GPU状态和handoff 6.15，再进入下一轮通用输入表示/双流负交互优化。

## 最短恢复流程

```bash
git clone git@github.com:LiuSong-Scrat/Lerobot.git lerobot
cd lerobot
git fetch origin --prune
git switch wep_vla_v0.4.3_v51_workflow_archive

git worktree add ../lerobot_v51 \
  origin/wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation

export LEROBOT_REPO="$PWD/../lerobot_v51"
export EXPERIMENT_ROOT=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811

bash benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/install_runtime_bundle.sh
```

恢复包 README：

```text
benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/README.md
```

## 当前固定提交

| 用途 | 分支 | 提交 |
|---|---|---|
| 权威项目记录 | `wep_vla_v0.4.3_multiview_doubleflow` | `a1c0ac5f682c7858de88b5a3396365a4c47a9d26` |
| V32 确定性复核 | `wep_vla_v0.4.3_worldflow_v32_eval_recheck` | `dba65f7e0466364dcb43eaa016a7d7fb3c05c90d` |
| V46 1cm/3cm input | `wep_vla_v0.4.3_v46_multiscale_novelty_union` | `d752f14df8c75cfcd9acd9c6b34d8e2e7d5b7296` |
| V49 对称点路径 | `wep_vla_v0.4.3_v49_symmetric_point_adaptation` | `0b8e7b48dbddcbab847737ef6d676709cc8dfbef` |
| V50 保守输入 | `wep_vla_v0.4.3_v50_conservative_multiscale_novelty` | `3daeddd7de6afb7db086dffb3a0ebcf590d5db9c` |
| V51 可执行代码 | `wep_vla_v0.4.3_v51_conservative_symmetric_point_adaptation` | `9db8504a2c575ad386f1d90efa98784c0ea8d701` |
| V51 运行归档 | `wep_vla_v0.4.3_v51_workflow_archive` | `707ba6012ef834169e67f0d5b16305cede78bcb7` |
| 对称双坐标 | `wep_vla_v0.4.3_worldflow_symmetric_dualcoord` | `4562fd17c8de5c7ee15ae930b1afdebb7881fd93` |
| 并行双坐标 | `wep_vla_v0.4.3_worldflow_parallel_dualcoord` | `a04fc510cf35e61369c4ae3fc656a507c9ed9e00` |
| canonical dualflow | `wep_vla_v0.4.3_worldflow_canonical_dualflow` | `62e8670537ba72339c762ba954c528646757eb5f` |
| shared-state dual adapter | `wep_vla_v0.4.3_worldflow_shared_state_dual_adapter` | `507ffc575067e706e85f7b276662c8dd96a3c0d2` |

## 重要运行状态

- V51 v12 cache 已完成：`137590` samples、`36` shards；36-shard exact-index audit 全部通过。
- 原训练入口在 step 0 暴露 dual v12 / primary v11 compatible schema 误拒绝；无 optimizer update，失败证据保留。
- schema compatibility 修复后测试为 `130/130`；修复 commit 为 `9db8504a2c575ad386f1d90efa98784c0ea8d701`。
- `v51r1_schemafix` 正在 tmux `wep_v043_v51r1_schemafix_conservative_symmetricpoint_paired_train` 中训练；恢复时先查 tmux、进程、W&B 与最新 checkpoint，不得重复启动。
- cache 基准 voxel 为 `1 cm`；`4 cm` 只用于副视角新颖性判断。
- 大型 dataset、checkpoint、cache 和 rollout 不进 Git；若 `/opt/data/private` 在新容器继续挂载，仍按原绝对路径复用。
- 不得删除 `/opt/data/private` 下任何文件。

完整对话决策归档位于 V51 workflow archive 分支：

```text
benchmarks/song_real_libero/workflows/v51_conservative_symmetric_point_adaptation/CURRENT_CONVERSATION_ARCHIVE_2026-08-15.md
```

## SSH 断线与 Codex 会话恢复

长时间评测、cache和训练必须继续放在独立 `tmux` 中；不要依赖 VS Code integrated terminal 或 Codex IDE extension 进程保持任务存活。客户端（发起 SSH 的电脑）在 `~/.ssh/config` 中为该服务器设置 keepalive，例如：

```sshconfig
Host wep-server
    HostName <server-host>
    User liusong
    ServerAliveInterval 30
    ServerAliveCountMax 10
    TCPKeepAlive yes
    ControlMaster auto
    ControlPersist 10m
```

Codex 权限键必须在 `~/.codex/config.toml` 顶层，位于任何 `[projects.*]` 表之前：

```toml
approval_policy = "never"
sandbox_mode = "danger-full-access"
model = "gpt-5.6-sol"
model_reasoning_effort = "high"

[projects."/home/liusong"]
trust_level = "trusted"
```

每次重连后先验证实际配置，而不是只查看文件文本：

```bash
codex doctor --json | jq '.checks[] | select(.id == "config.load" or .id == "sandbox.helpers")'
```

期望看到 `approval policy: Never`、`filesystem sandbox: unrestricted`。如果当前 Codex turn 已由宿主以更低权限创建，仅修改配置不会改变这个已经运行的 turn；需要结束旧 app-server 后重新打开或恢复会话。

优先在原 VS Code 窗口点击 reconnect，不要同时为同一 remote workspace 打开第二个窗口。若新窗口提示会话“正在另一个窗口运行”，先关闭旧窗口，再从普通 SSH 终端检查：

```bash
pgrep -af 'openai.chatgpt.*codex.*app-server'
```

只有确认对应 PID 属于已经断开的旧窗口后，才对该 stale app-server 使用普通 `kill <PID>`；不要手工删除 `~/.codex/thread-writer-locks/*.lock`，也不要在仍有 owner 进程时强占同一 thread。若无法可靠区分旧窗口，可使用 VS Code 命令面板的 `Remote-SSH: Kill VS Code Server on Host...` 后重连；这会关闭该主机上的 VS Code remote terminals，因此必须先确认真正的长任务都在独立 tmux 中。

最抗断线的 Codex 交互方式是在服务器 tmux 内使用 CLI。当前 WEP 会话可按保存的 thread ID 恢复：

```bash
tmux new-session -A -s codex-wep \
  'exec codex -C /home/liusong/ProgramFiles/Huggingface/lerobot -s danger-full-access -a never resume 01a00405-aa05-7312-8b24-c189ea35f0da'
```

CLI 的 `codex resume --last` 也可以恢复当前目录最近的会话；显式 thread ID 更不容易选错。恢复前仍须确保 IDE 中没有另一个活跃 owner。评测/训练 tmux 与 Codex tmux 分开命名，这样重启 Codex 或 VS Code 不会影响实验任务。
