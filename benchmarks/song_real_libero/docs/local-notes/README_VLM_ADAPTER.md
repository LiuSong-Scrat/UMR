# WEP-VLA v0.3.1 Frozen-VLM Image Adapter

本版本在 `wep_vla_v0.3.1_cache_v7` 的点云与动作架构上重新加入 SmolVLA 静态 RGB 输入。目标是保留 PointSeg、LitePT、PointAction fusion 和 Action Expert，同时正确载入并冻结经过大规模图文预训练的 SmolVLM 权重。

## 1. 模型结构

每个观测只使用每个相机的最后一帧 RGB。prefix token 顺序为：

```text
RGB image tokens ─┐
object point token├─> frozen SmolVLM prefix ──KV/context──> Action Expert
background token ┤
language tokens ─┘

foreground LitePT tokens ──> PointActionSelfAttention ──> noisy action tokens
```

image、point 和 language token 位于同一个 prefix attention block，因此 point token 能通过冻结的 SmolVLM 层读取图像和语言语义。Action Expert 同时接收当前已有的 foreground/action fusion，以及包含 image、object/background point 和 language 的 VLM context。

没有引入旧 `wepvla_v03_adapter` 的 gate 或 gated action bridge，也没有改动 `litept/model.py`。

### 冻结与训练边界

启用 adapter 后：

- 冻结并保持 eval：SmolVLM vision encoder、connector、语言 embedding、VLM transformer 和 LM head；
- 训练：PointSeg、LitePT 点云编码器、point projection、PointAction fusion；
- 训练：Action Expert、action/time projection 和 action output projection；
- `encode_robot_state=false` 时 state projection 仍不参与训练；
- WorldFlow 和 SE(3) 仍由原配置控制，默认关闭。

冻结 VLM 不会阻断 point token 的梯度。梯度会穿过固定 VLM 运算回传到 point projection 和点云编码器，但不会更新 VLM 参数。

## 2. 离线 VLM 权重

推荐下载完整原始 SmolVLM：

```bash
huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir /opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct
```

目录应包含 `config.json`、processor/tokenizer 文件和模型 safetensors。只有 `lerobot/smolvla_base` policy checkpoint 不等于完整原始 SmolVLM 目录。

- `vlm_model_name`：VLM 架构和 processor/tokenizer 来源；离线训练时指向完整原始 SmolVLM 目录。
- `vlm_weights_path`：可选权重覆盖来源，可以是完整原始 SmolVLM，也可以是含 `model.safetensors` 的 SmolVLA policy checkpoint。

```bash
# 直接载入完整原始 SmolVLM
--policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct

# 从 SmolVLA policy checkpoint 只抽取 VLM 权重
--policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
--policy.vlm_weights_path=/path/to/smolvla_policy_checkpoint
```

`vla_adapter_enable=true` 且 `vla_adapter_freeze_vlm=true` 时，配置会自动启用 `load_vlm_weights`、`train_expert_only` 和 `freeze_vision_encoder`。默认 `num_vlm_layers=16` 保持当前 SmolVLA/Action Expert 结构，不在 VLM 内插入额外层。

## 3. 生成包含 RGB 的数据集

Adapter 训练要求 metadata 至少声明一个 `observation.images.*`。RGB 保存为 LeRobot 标准图片路径，而不是把 NumPy 数组直接写入 episode buffer，因此 `compute_episode_stats` 可以正常读取。

### 真机 HDF5

脚本默认查找 `observations/images/<camera>`：

```bash
python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
  --input-dir /path/to/real_hdf5 \
  --camera overhead \
  --output-root /path/to/real_adapter_lerobot_dataset \
  --repo-id real_adapter_lerobot_dataset \
  --point-cloud-storage zarr \
  --workers 8 \
  --overwrite
```

非默认 HDF5 图像键可指定：

```bash
--image-key observations/images/overhead
```

只有明确使用 `--image-key none` 才生成无 RGB 的旧格式数据集，该数据集不能用于 adapter 模式。

### LIBERO

LIBERO 默认保存 point-cloud reference camera 的 RGB：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /path/to/libero_demos \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --num-workers 14 \
  --point-cloud-storage zarr \
  --output-root /path/to/libero_adapter_lerobot_dataset \
  --repo-id libero_adapter_lerobot_dataset \
  --save-rgb-images \
  --vis-count 2
```

可用 `--image-camera agentview` 指定静态图像相机；`--no-save-rgb-images` 只用于旧纯点云模型。

## 4. PointSeg cache

prior/cache 仍只依赖点云和轨迹，格式与 cache-v7 一致，RGB 不进入 cache。离线 cache 已启用轻量索引模式，不会为每个样本解码图像：

```bash
torchrun --standalone --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id=/path/to/real_adapter_lerobot_dataset \
  --output-dir=/path/to/song_pointseg_sample_cache_v7 \
  --batch-size=24 \
  --num-workers=14 \
  --current-points=10000 \
  --future-points=10000 \
  --vis-count=4 \
  --overwrite
```

不提供 `pointseg_sample_cache_dir` 时，在线 prior 仍使用当前双向轨迹软标签；在线 wrapper 会保留完整样本和 RGB。

## 5. 训练

在 cache-v7 命令上增加 adapter 和本地 VLM 参数：

```bash
export SONG_POINTSEG_REQUIRE_POINTOPS=1
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=/path/to/real_adapter_lerobot_dataset \
  --pointseg_sample_cache_dir=/path/to/song_pointseg_sample_cache_v7 \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct \
  --batch_size=48 \
  --steps=80000 \
  --log_freq=1 \
  --output_dir=/path/to/outputs/wep_vla_v031_frozen_vlm \
  --job_name=wep_vla_v031_frozen_vlm \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  --save_freq=2000 \
  --eval_freq=2000 \
  --num_workers=12 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.002 \
  --policy.pointseg_foreground_ratio=0.08 \
  --policy.pointseg_background_ratio=0.08 \
  --policy.pointseg_min_foreground_points=4000 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.point_action_fusion_enable=true \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false
```

训练日志中 learnable parameter 数量应明显小于 total parameter 数量。VLM 参数 `requires_grad` 必须全部为 `false`，Action Expert 和 point 模块必须有可训练参数。

旧纯点云行为仍使用 `--policy.vla_adapter_enable=false`，且不要求 RGB。

## 6. 推理

源码和 benchmark 推理 wrapper 都会根据 checkpoint 的 `vla_adapter_enable` 自动决定是否要求 RGB。真机输入示例：

```python
cur_model_observation = {
    "pose_eular": policy_pose_eular,
    "gripper_width": gripper_width,
    "point_cloud": overhead_cloud_rgb,
    "overhead": img_overhead_rgb,
    "hand": img_hand_rgb,
}
action = inference.single_inference(cur_model_observation, task=task)
```

图像可为 `HWC`/`CHW`、batched 或带时间维；策略只使用最后一帧。实际相机由 checkpoint 的 `config.image_features` 决定。缺少 RGB 时会明确报告期望和可用 key，不会静默使用空图像。

LIBERO evaluator 会从 `raw_obs` 读取 checkpoint 指定相机，并使用与数据转换相同的 `agentview` 翻转规则。

## 7. 兼容性检查

- 旧纯点云 checkpoint：保持 `vla_adapter_enable=false`，输入行为不变。
- 新 adapter checkpoint：必须能访问完整 VLM 架构/processor；迁移机器后可通过 CLI 覆盖失效绝对路径。
- cache-v7 不改变 prior 格式，但新数据集的 frame index 必须对应重新生成的 cache。
- PointSeg、gripper cloud、UMI 坐标变换、WorldFlow/SE(3) 开关保持当前分支逻辑。
- 多卡训练时冻结 VLM 不参与反向更新，PointAction fusion 与 Action Expert 正常进行 DDP 梯度同步。
