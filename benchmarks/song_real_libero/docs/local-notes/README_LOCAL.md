# LIBERO SETTTING
  ## LeroboDatasetFromHDF5 ####################### (Download) Hdf5-->Add Gripper-->Zarr-->Dataset
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python \
      benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
      --config benchmarks/song_real_libero/configs/libero.json \
      --demo-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_demos \
      --suite libero_spatial \
      --suite libero_object \
      --all-tasks \
      --episodes 2 \
      --num-workers 10 \
      --point-cloud-storage zarr \
      --output-root /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_4suite_lerobot_dataset \
      --repo-id song_libero_4suite_pointcloud \
      --save-video \
      --vis-count 2 
  ## PriorCache ###########################   MultiGPU  --nproc_per_node=4
  torchrun --standalone --nproc_per_node=1   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_4suite_lerobot_dataset   --output-dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_4suite_lerobot_cache   --batch-size=24   --num-workers=14   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=4   --overwrite

  ## TrainCode ###########################   MultiGPU CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu --num_processes 4 python benchmarks/song_real_libero/scripts/train_song_benchmark.py
    export SONG_POINTSEG_REQUIRE_POINTOPS=1 
    python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
      --policy.type=smolvla \
      --policy.push_to_hub=false \
      --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/temp_dataset \
      --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/libero_setting/libero_4suite_lerobot_cache \
      --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
      --policy.load_vlm_weights=false \
      --batch_size=8 \
      --steps=500000 \
      --log_freq=1 \
      --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh \
      --job_name=song_libero_pointseg_fresh \
      --policy.device=cuda \
      --wandb.enable=true \
      --wandb.disable_artifact=true \
      --save_freq=5000 \
      --eval_freq=5000 \
      --num_workers=0 \
      --policy.pointseg_enable=true \
      --policy.pointseg_backbone_type=litept \
      --policy.pointseg_grid_size=0.01 \
      --policy.pointseg_feature_dim=64 \
      --policy.pointseg_aux_loss_weight=0.002 \
      --policy.pointseg_foreground_ratio=0.08 \
      --policy.pointseg_background_ratio=0.08 \
      --policy.pointseg_min_foreground_points=500 \
      --policy.pointseg_min_background_points=0 \
      --policy.worldflow_enable=false \
      --policy.worldflow_se3_head_enable=false \
      --policy.se3_enable=false \
      --policy.se3_final_correction_enable=false

  ## BenchmarkEval ###########################  
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python   benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py   --config benchmarks/song_real_libero/configs/libero.json   --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh_post/checkpoints/last/pretrained_model  --suite libero_spatial  --all-tasks   --episodes 10  --action-index 0   --exec-action-steps 12   --save-video --render-mode offscreen   --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/10w_eval_spatial_temp

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python   benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py   --config benchmarks/song_real_libero/configs/libero.json   --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh_post/checkpoints/last/pretrained_model   --suite libero_spatial --suite  libero_goal    --all-tasks   --episodes 10  --action-index 0   --exec-action-steps 12   --save-video --render-mode offscreen   --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/song_real_libero/outputs/eval_temp






# REAL SETTING
  ## RawDataFromCamera 
    python benchmarks/song_real_libero/scripts/real_setting/record_bestman_rgbd.py \
    --camera L515 \
    --output benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video \
    --num-frames 0 \
    --storage video \
    --video-fps 15 \
    --space-toggle-recording 

  ## HDF5FromRawData  Raw--->HDF5
    ###############################
    #inference可视化  --run-inference or --show-inference
    #切片可视化  --show 
    #真机数据保持在 overhead 相机坐标系，不使用 camera_to_world 外参。
    #如果要一键离线推理再交互切片：
    python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
      --input benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube \
      --output-dir benchmarks/song_real_libero/data/real_setting/humanhand_demo_video_ego_cube \
      --run-inference \ 
      --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
      --fast \
      --force-handedness right \
      --fusion-mode model-depth \
      --camera-names overhead,hand 
      

    #直接用推理后的结果交互切片
    python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
      --input benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube \
      --output-dir benchmarks/song_real_libero/data/real_setting/humanhand_demo_video_ego_cube   

    #离线推理后切分点切片
    python benchmarks/song_real_libero/scripts/real_setting/build_humanhand_hdf5_dataset.py \
      --input benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube \
      --jsonl benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube/handpose_wilor.jsonl \
      --output-dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_cube_demo \
      --no-interactive \
      --pose-frame camera \
      --segments "$(cat benchmarks/song_real_libero/data/real_setting/rgbd_records/humanhand_demo_video_ego_cube/segments.txt)" \
      --max-points 50000 \
      --segment-workers 16

  ## 预处理---Continuous---HalfReduce---AddGripper---MixedStageGen
    ## Continuous----Check HDF5 Quality
      python benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py  --hdf5_dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_demo
    ## Continuous----Check HDF5 Quality

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/hdf5_edit_reduce.py

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/add_gripper_cloud_to_hdf5.py

    python /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/scripts/check_discontinuous_hdf5.py

  ## LeRobotDatasetFromRealHDF5  HDF5-->(Add Gripper-->)Current EEF/UMI-->Zarr-->Dataset
    # 输入 HDF5 已经包含 observations/cloud_rgb/<camera>，不会下载 LIBERO 数据、
    # 重播轨迹、渲染深度图或从深度图反投影点云。
    # 如果 HDF5 已经由 add_gripper_cloud_to_hdf5.py 加过末端点云（StageGen Mixed），追加： --input-has-gripper-cloud 
    # 如果无虚拟末端点云，使用 --gripper-points 500 --gripper-max-width 0.08 
    # 当使用变长点云（适配StageGen Mixed）训练，使用 --num-points 0
    export HDF5_USE_FILE_LOCKING=FALSE
    python benchmarks/song_real_libero/scripts/real_setting/real_hdf5_to_dataset.py \
      --input-dir benchmarks/song_real_libero/data/real_setting/humanhand_offline_demo \
      --output-root benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset \
      --repo-id song_real_pointcloud \
      --fps 15 \
      --num-points 0 \
      --input-has-gripper-cloud \
      --point-cloud-storage zarr \
      --workers 6 \
      --vis-count 2 \
      --overwrite \
      --task  "Place the Red Cube on the Blue Cube"


  ## PriorCache ###########################   MultiGPU  --nproc_per_node=4
  torchrun --standalone --nproc_per_node=1   benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py   --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset   --output-dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_priorseg_cache   --current-points 50000 --future-points 16384  --batch-size=24   --num-workers=14   --shard-size=4096  --storage-dtype=float16   --nn-chunk-size=1024   --vis-count=0   --overwrite

  ## TrainCode ###########################   MultiGPU CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu --num_processes 4 python benchmarks/song_real_libero/scripts/train_song_benchmark.py
    export SONG_POINTSEG_REQUIRE_POINTOPS=1 
    python benchmarks/song_real_libero/scripts/train_song_benchmark.py \
      --policy.type=smolvla \
      --policy.push_to_hub=false \
      --dataset.repo_id=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_lerobot_dataset \
      --pointseg_sample_cache_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/data/real_setting/real_priorseg_cache \
      --policy.vlm_model_name=/home/liusong/SmolVLM2-500M-Video-Instruct \
      --policy.load_vlm_weights=false \
      --batch_size=4 \
      --steps=500000 \
      --log_freq=1 \
      --output_dir=/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/real_setting/ep-vla \
      --job_name=ep-vla  \
      --policy.device=cuda \
      --wandb.enable=true \
      --wandb.disable_artifact=true \
      --save_freq=5000 \
      --eval_freq=5000 \
      --num_workers=0 \
      --policy.pointseg_enable=true \
      --policy.pointseg_backbone_type=litept \
      --policy.pointseg_grid_size=0.01 \
      --policy.pointseg_feature_dim=64 \
      --policy.pointseg_aux_loss_weight=0.002 \
      --policy.pointseg_foreground_ratio=0.08 \
      --policy.pointseg_background_ratio=0.08 \
      --policy.pointseg_min_foreground_points=500 \
      --policy.pointseg_min_background_points=0 \
      --policy.worldflow_enable=true \
      --policy.worldflow_se3_head_enable=false \
      --policy.se3_enable=false \
      --policy.se3_final_correction_enable=false

  ## BenchmarkEval ###########################  
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  python   benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py   --config benchmarks/song_real_libero/configs/libero.json   --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh_post/checkpoints/last/pretrained_model  --suite libero_spatial  --all-tasks   --episodes 10  --action-index 0   --exec-action-steps 12   --save-video --render-mode offscreen   --output-dir /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/10w_eval_spatial_temp


