
Mujoco==3.3.4 Robotsuite==1.4.0



## Libero Benchmark
## 1.准备WEP-VLA Lerobot格式Benchmark数据集 # --state-observation-offset 0 
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/libero_demos \
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
  --state-observation-offset 0 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --image-camera agentview \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --overwrite \
  --vis-dir /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data/libero_4suite_lerobot_dataset/visualizations \
  --output-root /opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data/libero_4suite_lerobot_dataset \
  --repo-id song_libero_4suite_pointcloud

# Cache

torchrun --standalone --nproc_per_node=4   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data/libero_4suite_lerobot_dataset   --output-dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data/libero_4suite_lerobot_toolseg_cache   --batch-size=24   --num-workers=4   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=4  --overwrite

# TRAIN
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4   benchmarks/song_real_libero/scripts/train_song_benchmark.py     --policy.path=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model     --policy.push_to_hub=false     --dataset.repo_id=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data/libero_4suite_lerobot_dataset      --pointseg_sample_cache_dir=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_data/libero_4suite_lerobot_toolseg_cache     --policy.vla_adapter_enable=true     --policy.vla_adapter_freeze_vlm=true     --policy.vlm_model_name=/opt/data/private/liusong/hf_models/SmolVLM2-500M-Video-Instruct     --policy.vlm_weights_path=/opt/data/private/liusong/hf_models/smolvla_base     --policy.load_vlm_weights=true     --batch_size=48     --steps=80000     --log_freq=1     --output_dir=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_after30k_lr5 --job_name=wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_after30k_lr5      --policy.device=cuda     --wandb.enable=true     --wandb.disable_artifact=true     --save_freq=2000     --eval_freq=2000     --num_workers=14     --policy.pointseg_enable=true     --policy.pointseg_backbone_type=litept     --policy.pointseg_grid_size=0.01     --policy.pointseg_feature_dim=64     --policy.pointseg_aux_loss_weight=0.0005     --policy.pointseg_foreground_ratio=0.025     --policy.pointseg_background_ratio=0.025     --policy.pointseg_min_foreground_points=2500     --policy.pointseg_min_background_points=0      --policy.pointseg_use_temporal_priors_as_input=false      --policy.pointseg_use_pseudo_selection=false     --policy.worldflow_enable=false     --policy.worldflow_se3_head_enable=false     --policy.se3_enable=false     --policy.se3_final_correction_enable=false

# FULL TEST
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model \
  --suite-gpu-ids 0,1,2,3 \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --strict-official-init \
  --gripper-control-mode delta_width_initial_sync \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --waypoint-max-hold-steps 1 \
  --isolated-policy-workers 1 \
  --task-workers 10 \
  --episode-workers-per-task 2 \
  --inference-batch-size 80 \
  --no-release-event-exec-enable \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 \
  --grasp-exec-steps 24 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_FULL


# Small TEST

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MALLOC_ARENA_MAX=2 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model \
  --suite libero_goal \
  --task-id 3 \
  --task-id 5 \
  --task-id 6 \
  --episodes 50 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --strict-official-init \
  --gripper-control-mode delta_width_initial_sync \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --waypoint-max-hold-steps 1 \
  --isolated-policy-workers 1 \
  --task-workers 3 \
  --episode-workers-per-task 25 \
  --inference-batch-size 75 \
  --no-release-event-exec-enable \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 \
  --grasp-exec-steps 24 \
  --max-steps 800 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_delta_width_initial_sync_800_mujoco334