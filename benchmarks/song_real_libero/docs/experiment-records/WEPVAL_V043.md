# Experiment root (all new cache/checkpoints/eval/log artifacts stay here)

```bash
export SONG_V043_EXPERIMENT_ROOT=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview/runs/wep_vla_v043_dualview_baseline_guard_20260811
export SONG_V043_SOURCE_ROOT=/opt/data/private/liusong/benchmarks/song_real_libero/data/libero_setting/wep_vla_v042_general_dataset_multiview
export SONG_V043_DATASET_ROOT=$SONG_V043_EXPERIMENT_ROOT/dataset
export SONG_V043_CACHE_ROOT=$SONG_V043_EXPERIMENT_ROOT/pointseg_cache_fps_union
export SONG_V043_TRAIN_ROOT=$SONG_V043_EXPERIMENT_ROOT/training
export SONG_V043_STAGE1_TRAIN_ROOT=$SONG_V043_TRAIN_ROOT/stage1_dualview_fps_union_4gpu_b48
export SONG_V043_STAGE2_TRAIN_ROOT=$SONG_V043_TRAIN_ROOT/stage2c_worldego_joint_se3_chart_conjugate_4gpu_b24_accum2
export SONG_V043_EVAL_ROOT=$SONG_V043_EXPERIMENT_ROOT/eval
export WANDB_DIR=$SONG_V043_EXPERIMENT_ROOT/wandb
export SONG_V043_BASELINE_CKPT=/opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model
export SONG_V043_PYTHON=/home/liusong/anaconda3/envs/reap/bin/python3.10
export SONG_V043_TORCHRUN=/home/liusong/anaconda3/envs/reap/bin/torchrun
mkdir -p "$SONG_V043_CACHE_ROOT" "$SONG_V043_TRAIN_ROOT" "$SONG_V043_EVAL_ROOT" "$SONG_V043_EXPERIMENT_ROOT/logs" "$SONG_V043_EXPERIMENT_ROOT/artifacts" "$WANDB_DIR"
```

The immutable source dataset is exposed through the `dataset` symlink inside
the experiment root. The previous equal-fusion cache and training output are
also linked there for comparison; do not move or duplicate those large trees.

The active dual-view method is a camera-agnostic equal-union FPS contract. Both
stored views are already expressed in the current end-effector frame. The
pipeline removes the duplicated gripper tail, concatenates all 9,500 scene
points from each view, applies exact pointops CUDA farthest-point sampling to
the 19,000-point scene union, and selects 9,500 scene points. It then appends
one unchanged 500-point gripper tail, preserving the checkpoint's 10,000-point
model input.

The same helper runs during offline cache generation, cacheless online
pseudo-label generation, and model inference. Cache shards save the selected
indices relative to the 19,500-point raw union, so training reconstructs exactly
the points used to generate pseudo labels. The manifest records
`camera_view_fusion=fps`,
`camera_view_weights=null`, the input/output counts, and gripper count.

The old weighted-budget and action-drift sweeps are retained only as rejected
historical artifacts. They are not a Stage 1 gate and no active launcher action
uses them. The legacy budget mode remains the default only so old checkpoints
that do not contain `camera_view_fusion` load without a behavioral change.

## Union-FPS full-data rejection and next input-only candidate (2026-08-13)

Union-FPS is no longer an active Stage-1 candidate. The final controlled run
used all LIBERO-10 `10 tasks x 50 episodes`, exact paired primary/dual coverage,
four GPUs at batch 48/GPU, 12 workers, and 1,500 optimizer steps. This is
288,000 sample exposures over 275,180 paired samples, or 1.047 paired epochs.
The point-input path used a 10x learning rate relative to the jointly trainable
pretrained action path; WorldFlow and SE(3) remained disabled.

Steps 25, 75, 100, 375, 750, 1125, and 1500 all reached at least three failures
before completing the strict 100-episode guard, so none could reach 98/100.
The immutable result is
`artifacts/full500_union_fps_paired_disc_all_checkpoint_rejection.json`.
This rules out insufficient data coverage as the explanation and prevents this
candidate from entering the formal 500-episode ablation or unlocking WorldFlow.

The next bounded candidate is `camera_view_fusion=full_union`. It is still a
pure input-layer operation: transform every view to Ego, remove repeated
gripper tails, retain every scene point from every view, and append one primary
gripper tail. For two 10,000-point inputs this yields 19,500 points. No camera
quota, voxel/occlusion rule, learned gate, or model parameter is added. A paired
batch pads the exact 10,000-point primary copy to 19,500 and supplies a padding
mask; the dual copy contains all 19,500 valid points. PointSeg and LitePT already
support variable point counts and emit fixed-size downstream tokens. This
candidate must first pass unit tests, cache alignment, and a 50-episode resource
smoke before any full-data training.

### Full-union rejection (2026-08-13)

`full_union` passed its implementation and resource checks but failed the
closed-loop gate catastrophically. The Task-8 cache contains all 20,744 source
frames at 19,500 points/frame. A cache-contract repair explicitly permits the
intended 10,000-point primary replay to pair with a 19,500-point two-view cloud;
the collator pads only the primary copy and supplies the exact padding mask.
The four-GPU training smoke then completed forward, backward, and checkpoint
save at batch 48/GPU with 12 workers/process. Peak observed memory was below
24 GiB and the expected mixed-batch `pseudo_valid_ratio` was about 0.75.

The formal Task-8 pilot used exact paired coverage for 225 optimizer steps:
43,200 exposures over 41,488 paired samples, or 1.041 epochs. WorldFlow and SE3
were disabled. Checkpoints 25 and 75 were evaluated on the same official
states with two point-cloud views and one RGB view. Both failed the first 15/15
episodes. Their mathematical maximum became 35/50, below the archived Task-8
baseline of 45/50, so both rollouts and the queued steps 100/225 were stopped.
No output was deleted. The immutable early-rejection record is
`artifacts/task08_full_union_paired_oneepoch_early_rejection.json`.

This rules out variable-length full union as the next Stage-1 method and also
shows why offline finite loss is not acceptance evidence. A same-weight
primary-only diagnostic on states 0--14 was launched to separate damage from
the paired update from damage caused specifically by the 19,500-point dual
input. WorldFlow remains locked.

## Task-8 50-episode pilot status (2026-08-11)

All pilot files are under:

```text
$SONG_V043_EXPERIMENT_ROOT/pilots/libero10_task08_moka_50ep
```

- Distributed resource smoke: 96 samples, 4 GPUs, batch 24/GPU, exit 0. All
  eight cross-rank samples audited had 9,500 unique FPS scene indices and the
  exact primary-view gripper indices `19000:19500`.
- Full pilot cache: 20,744/20,744 samples from episodes `400:450`, 12 shards,
  exit 0. Ten rank/shard boundary samples passed the same index and label-shape
  audit.
- Four-GPU training smoke: batch 48/GPU (global batch 192), 2 optimizer steps,
  exit 0, observed peak memory at most 17.5 GiB/GPU, finite losses and gradients
  (`loss_action=0.001208/0.001259`, `grad_norm=0.066/0.049`), and a reloadable
  1.40 GB checkpoint whose config records dual-view FPS with WorldFlow and SE3
  disabled.
- The formal Stage-1 pilot completed 100 optimizer steps, or 19,200 samples
  (about 0.93 epoch of the 20,744-frame pilot), on 4 GPUs at batch 48/GPU.
  Immutable checkpoints were written every 25 steps below
  `training/stage1_dualview_fps_union_4gpu_b48`; all four reload and contain
  finite weights. The earlier single-GPU
  2,000-step partial run is retained as a historical artifact but was stopped
  when the four-GPU requirement was selected; it is not a Stage-1 candidate.
- This completed pilot was launched with `num_workers=2`. Per the updated
  runtime requirement, every subsequent training launch defaults to
  `num_workers=12`.
- Checkpoint screening on official Task-8 states 0--19 selected step 50. Step
  50 and step 100 both scored 18/20; the earlier checkpoint is the conservative
  tie-break because it uses less task-specific fine-tuning.
- The final Stage-1 gate used four independent GPU policy processes and 30
  total MuJoCo episode workers. Shard worker/batch sizes were `8,8,7,7`; actual
  maximum observed inference batches were `5,6,5,4`. The immutable step-50
  checkpoint scored **47/50 (94.0%)**, versus the archived Task-8 baseline
  **45/50 (90.0%)**, passing non-regression by 2 episodes / 4.0 percentage
  points. Failed official state indices were `2,12,26`. The aggregate audit is
  `artifacts/stage1_parallel30_v17_task8_gate.json`.

### Stage-2 failure evidence and current Stage-2c contract (2026-08-12)

- The first WorldFlow-only attempt was evaluated at steps 25/75/100 and scored
  43/50, 39/50, and 42/50. It is rejected; its immutable audit is
  `artifacts/stage2_worldflow_parallel30_v17_task8_gates.json`.
- The first direct SE(3) attempt was stopped as soon as its initial parallel
  rollout was catastrophically below the gate. The preserved partial counts
  were step 25: 0/24, step 50: 0/24, step 75: 1/21, and step 100: 0/21. No
  partial episode is counted. See
  `artifacts/stage2b_worldego_joint_se3_aborted_early.json`.
- Root cause: the old checkpoint was trained with Gaussian noise in the raw
  pose9 chart, including near-zero rot6D coordinates. Feeding normalized
  rotation-matrix columns to the Action Expert changed its input distribution;
  adding random World tokens to the same bidirectional self-attention block
  also perturbed the pretrained Ego path before World had learned anything.
- Stage-2c therefore keeps two synchronized states. The Action Expert receives
  the exact legacy chart interpolation, while a separate physical `SE(3)` state
  is projected, geodesically integrated, and used for the final pose. World is
  initialized and evolved by exact Ego/World conjugation.
- Ego/World interaction is bidirectional without a gate: Ego conditions World,
  then the updated World conditions Ego through cross-attention. The
  World-to-Ego output projection is a full learnable matrix initialized to zero,
  which preserves the Stage-1 Ego function at initialization without freezing
  either branch or adding a scalar gate. Both branches and the shared Action
  Expert train jointly.
- A 10-step, 50-episode-dataset probe followed by an official-state 0--9
  rollout scored **9/10**. This equals Stage-1 on the same range; Stage-2c
  succeeded on Stage-1's failed episode 2 and failed episode 1. This establishes
  initial non-regression, not yet a WorldFlow improvement. The evaluation is
  under
  `eval/stage2c_bidir_se3_chart_probe10_v17_step000010_ep00_09_w10_b10`.
- The formal Stage-2c pilot completed 100 steps on four GPUs, batch 24/GPU,
  gradient accumulation 2, and 12 data workers. The 10-state screen for steps
  25/50/75/100 was 8/10, 10/10, 7/10, and 8/10. Per the early-stop rule, only
  step 50 expanded to all official states; it scored **45/50 (90.0%)**, with
  failures `12,20,31,35,40`. This is two successes / 4 percentage points below
  Stage-1 and therefore rejected. Its immutable audit is
  `artifacts/stage2c_worldego_bidirectional_se3_chart_step50_parallel30_v17_task8_gate.json`.
- One epoch was not enough to fit the randomly initialized World path: at step
  100 its measured World error was still 0.720 m / 19.46 degrees. Simply
  extending the same optimizer is unsafe because the offline loss improved
  while 10-state rollout success was non-monotonic. Stage-2d therefore uses
  standard two-timescale fine-tuning: all parameters remain trainable, but the
  pretrained Ego/shared group uses `0.2 * base_lr` while new World and both
  cross-attention directions use `1.0 * base_lr`. This is neither a gate nor a
  freeze. Its new 100-step pilot writes below
  `training/stage2d_worldego_bidirectional_se3_chart_twotimescale_4gpu_b24_accum2`.
- Stage-2d completed and disproved a simple "more steps" explanation. Full
  official-state results were step 25: **42/50**, step 75: **47/50**, and step
  100: **44/50**. Step 75 only ties Stage-1; later is worse despite lower
  offline losses. The audit is
  `artifacts/stage2d_twotimescale_parallel30_v17_task8_gates.json`.
- The remaining geometric conditioning problem is the historical global-frame
  conjugation `G=C B C^-1`. Its translation includes the world-origin lever-arm
  term `(I-R_G) p_C`, so a small Ego rotation can become a decimetre-scale
  World translation. Stage-2e uses an exactly invertible carrier with the same
  world-aligned orientation but zero translation: the current EEF is the local
  World origin. This removes the lever arm while preserving bidirectional SE(3)
  conjugacy and does not add a gate. A 10-step probe is required before any
  formal rerun. That probe reduced World translation error from about 0.82 m
  to 0.21 m at step 10 and scored **9/10** on official states 0--9, equal to
  Stage-1 on the same range. It passed the feasibility gate, so the formal
  100-step Stage-2e pilot is rooted at
  `training/stage2e_worldego_bidirectional_se3_chart_localworld_twotimescale_4gpu_b24_accum2`.
- Stage-2e completed. Its 10-state screens for steps 25/50/75/100 were
  **10/10, 9/10, 9/10, and 10/10**. Full 50-state gates for the explicitly
  required steps 25/75/100 were **45/50, 43/50, and 46/50**. Step 100 was best
  but remained one success below Stage-1. The immutable reports are
  `artifacts/stage2e_localworld_short_gates_v17_task8.json` and
  `artifacts/stage2e_localworld_full_gates_v17_task8.json`.
- To test under-training directly, Stage-2f continued from Stage-2e step 100
  for another 100 updates with peak/decay LR `2.5e-6/2.5e-7`; the pretrained
  Ego/shared group retained its 0.2 multiplier and no parameter was frozen.
  Total-step 125/150/175/200 screens scored **10/10, 10/10, 8/10, and 9/10**.
  Expanding the two promising candidates gave total-step 125 **45/50**
  (failures `17,30,31,37,40`) and total-step 150 **47/50** (failures
  `23,42,44`). Thus additional training recovered Stage-1 but did not provide
  a strict WorldFlow gain, and later checkpoints remained non-monotonic.
- The next architecture iteration must verify that World-to-Ego information
  materially affects the final action prediction; simply increasing update
  count is now rejected. A candidate may advance to the final `libero_10`
  10-task x 50-episode comparison only after it strictly exceeds the Stage-1
  Task-8 gate. The final report must separately quantify baseline-to-multiview
  and multiview-to-doubleflow deltas.
- Stage-2h tested a checkpoint-compatible, zero-initialized conjugate-residual
  head with no gate. Its 25/50/75/100-step Task-8 results were **45/50,
  47/50, 43/50, and 45/50**; the best checkpoint only tied Stage-1. A paired
  step-50 causal diagnostic disabled both World-to-Ego cross-attention and the
  residual twist. The resulting mean action change (`2.19 mm`, `0.258 deg`)
  was below the same-input repeat control (`2.66 mm`, `0.354 deg`), while the
  residual-head Frobenius norm was only `4.68e-4`. Stage-2f's independent
  World prediction still had approximately `0.13 m / 17 deg` endpoint error.
  Consequently Stage-2h is rejected, and additional low-weight residual
  training is not considered evidence that WorldFlow is effective.
- Stage-2i therefore uses a two-phase, gate-free training protocol. Phase A
  trains Ego and World jointly while keeping the baseline-preserving Ego
  action path, but raises the normalized independent-World objectives enough
  to make the World score learnable. Only after the logged World endpoint
  errors materially converge may Phase B enable exact 1:1 conjugate score
  fusion. This is a capability curriculum rather than an arbitrary 99:1
  mixture, and neither Ego nor World parameters are frozen.
- Stage-2i's strong-auxiliary independent World branch did not converge fast
  enough to support Phase B. At total steps 25/50/75/100 its 10-state screens
  were **10/10, 9/10, 8/10, and 10/10**, while the independent World endpoint
  error remained about `10.6 cm / 12.75 deg`. It was not expanded to 50-state
  gates because the World estimate was still structurally too inaccurate.
- Stage-2j copied compatible trained Ego modules into the independent World
  branch without sharing or freezing parameters. The pose-chart copy did not
  transfer across frames: after 25 steps the World endpoint error was still
  about `9.3 cm / 25.06 deg`, so the branch was rejected. Stage-2k replaced the
  pose-chart World decoder with a dedicated six-dimensional twist head; its
  first batches were worse (`11--13 cm / 72--83 deg`) and the tmux job was
  stopped immediately under the early-stop rule.
- Stage-2l returned to the geometrically stable conjugate-residual formulation
  and combined it with Stage-2i's stronger World objectives. The residual-head
  weight norm grew monotonically at total steps 25/50/75/100:
  `5.72e-4, 1.25e-3, 1.60e-3, 1.70e-3`. All four checkpoints scored **10/10**
  on the identical short screen, but their full Task-8 gates were **47/50,
  46/50, 47/50, and 47/50**. Failure indices were respectively
  `12,17,19`; `6,15,18,36`; `9,36,38`; and `4,28,47`. This is direct evidence
  that additional steps make the World residual nonzero but do not make the
  closed-loop success metric monotonic or strictly better. Stage-2l therefore
  ties Stage-1 and is rejected for final 10-task expansion. The implementation
  regression suite passes `72/72` tests.
- The accumulated Stage-2f/h/l evidence rejects "too few updates" as the sole
  explanation. The next candidate must improve the alignment between World
  supervision and the executed Ego action, and must still strictly exceed
  **47/50** on the same Task-8 gate before any 10-task x 50-episode run.
- A Stage-2l audit then found that its SE(3)-augmented training path bypassed
  `conjugate_residual` and decoded the augmented sample with the rejected
  independent World pose-chart head. Stage-2m fixes that implementation error:
  ordinary training, augmented-frame equivariance training, and online
  denoising now all use the same
  `Ad_{carrier}(Ego velocity) + World residual` decoder. The augmented carrier
  is exactly `A C`, where `A` is the sampled coordinate transform and `C` is
  the Ego-to-World carrier. The augmented bidirectional expert output is also
  used instead of being discarded. No gate or frozen branch was introduced.
- The fix reduced measured World flow loss from roughly `0.03--0.05` to
  `0.0002--0.0076`, geo loss from `0.14--0.18` to `0.015--0.027`, and
  equivariance loss from `0.25--0.35` to approximately `4e-5--1e-3`.
  Nevertheless its total-step 25/50/75/100 short screens were **9/10, 9/10,
  9/10, and 10/10**, and the corresponding full Task-8 gates were **45/50,
  46/50, 43/50, and 46/50**. Failure indices were respectively
  `9,11,21,28,31`; `5,25,36,46`; `7,15,17,24,30,31,37`; and `1,16,36,48`.
  The SE(3) consistency fix is retained as a correctness repair, but Stage-2m
  is rejected because it remains below Stage-1 and is still non-monotonic in
  checkpoint step. The post-fix regression suite passes `73/73` tests.

- Stage-2n replaces the asymmetric residual construction with exact, gate-free
  World/Ego consensus. If `p` is the Ego twist, `r` is the learned World
  residual, and `C` is the current Ego-to-World carrier, the two executed
  representations are now
  `q_ego = p + 0.5 Ad_C^-1(r)` and
  `q_world = Ad_C(q_ego) = Ad_C(p) + 0.5 r`. Thus both directions describe
  exactly the same action while Ego, World, both cross-attention directions,
  and the shared expert all remain trainable. Its 25/50/75/100 short screens
  were **10/10, 10/10, 9/10, and 9/10**. Full Task-8 gates at total steps 25
  and 50 were **42/50** and **47/50**. Step 50 only tied Stage-1, so Stage-2n
  was not expanded. The implementation regression suite passes `66/66`
  focused SE(3)+PointSeg tests (`45/45` for SE(3) alone).
- The Stage-2n 25-to-100 continuation loaded `policy.path` with `resume=false`;
  consequently it reset optimizer state and ran a second learning-rate warmup.
  Stage-2o removes this experimental confound by training once for 100 steps
  from the same immutable source, under one optimizer and one 100-step cosine
  schedule, while saving steps 25/50/75/100. Its identical official-init
  10-episode screens are **9/10, 10/10, 9/10, and 8/10**. This clean trajectory
  proves that later checkpoints are not intrinsically better even when no
  scheduler restart occurs; closed-loop selection must remain metric-based.
  All four full 50-episode Task-8 results were nevertheless measured because a
  one-episode short-screen difference is not a reliable cutoff, especially for
  step 25. They scored **44/50, 43/50, 46/50, and 47/50**, with failure indices
  `6,8,21,30,31,42`; `1,9,12,24,25,35,43`; `0,9,15,24`; and `26,28,33`.
  Stage-2o therefore does not pass the strict `>47/50` expansion threshold.
  Stage-2p continues from step 100 by restoring its optimizer, scheduler, RNG,
  and update number, rather than loading only policy weights. Updates 101--150
  run at the completed cosine schedule's constant floor LR with no new warmup;
  total steps 125 and 150 are the final clean test of the insufficient-steps
  hypothesis before changing the method again. Their short screens were both
  **9/10**. The complete Task-8 results were **45/50** at total step 125
  (failures `25,37,41,42,48`) and **48/50** at total step 150 (failures
  `15,16`). Thus additional training can help, but only the exact-resume
  constant-tail trajectory at step 150 strictly exceeds the Stage-1 result of
  47/50. This checkpoint is the first WorldFlow candidate allowed to advance;
  a same-checkpoint World-to-Ego causal ablation was measured before scaling.
  Disabling both World-to-Ego cross-attention and residual correction reduced
  the result to **43/50**, with failures `2,6,9,15,18,22,38`. The diagnostic
  uses the same 50 unique official initial states without retry. Consequently
  the online World path has a measured causal contribution of **+5/50** rather
  than merely inheriting an improved jointly fine-tuned Ego checkpoint.

The evaluator was also audited end-to-end: `camera_view_fusion` is now a real
argument of `observation_to_model_point_cloud`, and checkpoint fusion metadata
is copied into the rollout config before camera alignment returns. This prevents
the online evaluator from silently reverting the raw 19,500-point FPS union.

# Historical MultiView DataCollection (already complete; do not rerun/overwrite)
ulimit -n 65535
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=0 \
python benchmarks/song_real_libero/scripts/libero_setting/libero_hdf5_to_dataset.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --demo-root benchmarks/song_real_libero/data/libero_setting/libero_demos \
  --suite libero_10 \
  --all-tasks \
  --episodes 50 \
  --num-workers 14 \
  --num-points 10000 \
  --camera agentview \
  --camera robot0_eye_in_hand \
  --image-camera agentview \
  --image-camera robot0_eye_in_hand \
  --point-cloud-storage zarr \
  --fps 20 \
  --replay-mode states \
  --state-observation-offset 0 \
  --restore-demo-model \
  --require-source-fps-match \
  --save-rgb-images \
  --no-download-demos \
  --save-video \
  --vis-count 2 \
  --vis-dir="$SONG_V043_SOURCE_ROOT/libero_4suite_lerobot_dataset/visualizations" \
  --output-root="$SONG_V043_SOURCE_ROOT/libero_4suite_lerobot_dataset" \
  --repo-id song_libero_4suite_pointcloud


# fused multiview cache
SONG_POINTCLOUD_GRIPPER_POINTS=500 \
"$SONG_V043_PYTHON" "$SONG_V043_TORCHRUN" --standalone --nproc_per_node=4 \
  benchmarks/song_real_libero/scripts/song_cache_pointseg_samples.py \
  --dataset.repo_id="$SONG_V043_DATASET_ROOT" \
  --camera-views=agentview,robot0_eye_in_hand \
  --camera-view-fusion=fps \
  --output-dir="$SONG_V043_CACHE_ROOT" \
  --batch-size=24 \
  --num-workers=4 \
  --shard-size=4096 \
  --storage-dtype=float16 \
  --nn-chunk-size=1024 \
  --vis-count=4


# Train
ulimit -n 65535
export SONG_POINTSEG_REQUIRE_POINTOPS=1
export SONG_POINTCLOUD_GRIPPER_POINTS=500
OMP_NUM_THREADS=1 "$SONG_V043_PYTHON" -m accelerate.commands.launch \
  --multi_gpu --num_processes=4 --num_machines=1 \
  --mixed_precision=no --dynamo_backend=no --main_process_port=0 \
  benchmarks/song_real_libero/scripts/train_song_benchmark.py \
  --policy.path="$SONG_V043_BASELINE_CKPT" \
  --policy.push_to_hub=false \
  --dataset.repo_id="$SONG_V043_DATASET_ROOT" \
  --pointseg_sample_cache_dir="$SONG_V043_CACHE_ROOT" \
  --policy.camera_views=agentview,robot0_eye_in_hand \
  --policy.camera_view_fusion=fps \
  --policy.rgb_camera_views=agentview \
  --policy.vla_adapter_enable=true \
  --policy.vla_adapter_freeze_vlm=true \
  --policy.vlm_model_name="$SONG_V043_VLM_MODEL" \
  --policy.vlm_weights_path="$SONG_V043_VLM_WEIGHTS" \
  --policy.load_vlm_weights=true \
  --batch_size=48 \
  --steps=500 \
  --log_freq=1 \
  --output_dir="$SONG_V043_STAGE1_TRAIN_ROOT" \
  --job_name=wep_vla_v043_libero10_dualview_fps_union_4gpu_b48_500steps \
  --policy.device=cuda \
  --wandb.enable=false \
  --wandb.disable_artifact=true \
  --save_freq=100 \
  --eval_freq=100 \
  --num_workers=12 \
  --policy.pointseg_enable=true \
  --policy.pointseg_backbone_type=litept \
  --policy.pointseg_grid_size=0.01 \
  --policy.pointseg_feature_dim=64 \
  --policy.pointseg_aux_loss_weight=0.0005 \
  --policy.pointseg_foreground_ratio=0.025 \
  --policy.pointseg_background_ratio=0.025 \
  --policy.pointseg_min_foreground_points=2500 \
  --policy.pointseg_min_background_points=0 \
  --policy.pointseg_use_temporal_priors_as_input=false \
  --policy.pointseg_use_pseudo_selection=false \
  --policy.worldflow_enable=false \
  --policy.worldflow_se3_head_enable=false \
  --policy.se3_enable=false \
  --policy.se3_final_correction_enable=false


# Stage-gate evaluation against the archived 97.05 protocol

Set an immutable checkpoint step; never use `checkpoints/last` in the report.
The active throughput profile uses at most 30 environment workers and inference
batch 30. For the single-task pilot, the 50 official episodes are split across
four GPUs with worker/batch sizes `8,8,7,7`; no retry is permitted and every
episode index appears exactly once in the aggregate report.

```bash
# Screen checkpoints 000100/000200/000300/000400/000500 first. Substitute an
# immutable selected checkpoint below; do not assume that 000500 is best.
export SONG_V043_EVAL_CKPT=/absolute/selected/stage1/checkpoint/pretrained_model
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$SONG_V043_PYTHON" \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path "$SONG_V043_EVAL_CKPT" \
  --suite-gpu-ids 0,1,2,3 \
  --suite libero_spatial --suite libero_object --suite libero_goal --suite libero_10 \
  --all-tasks --episodes 50 \
  --policy-noise-seed 0 --env-seed 7 --strict-official-init \
  --gripper-control-mode delta_width_initial_sync \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --waypoint-max-hold-steps 1 \
  --isolated-policy-workers 1 --task-workers 3 --episode-workers-per-task 10 \
  --task-worker-backend process \
  --inference-batch-size 30 \
  --control-freq 20 --action-index 0 \
  --exec-action-steps 24 --adaptive-exec-max-steps 24 --grasp-exec-steps 24 \
  --max-steps 1000 --no-use-suite-max-steps --recreate-env-per-episode \
  --render-mode offscreen --no-visualize-foreground --save-video \
  --output-dir "$SONG_V043_EVAL_ROOT/dualview_fps_union_selected_parallel30"
```


# Stage 2c: joint bidirectional World/Ego flow with checkpoint-compatible SE(3)
# Leave --policy.worldflow_enable=false above for the Stage-1 baseline. Stage 2c
# appends causal World tokens after the unchanged Ego suffix and adds explicit
# Ego->World and World->Ego cross-attention. The latter has a zero-initialized
# full output projection (not a gate), so the initial Ego function is preserved.
#
# The dataset must contain world_ee_poses/ and, preferably,
# action_target_ee_poses/ sidecars produced by the current converters.
# Replace the WorldFlow flags in the training command with:
#
#   --policy.point_action_fusion_enable=true \
#   --policy.worldflow_enable=true \
#   --policy.worldflow_feature_dim=64 \
#   --policy.worldflow_grid_size=0.01 \
#   --policy.worldflow_bootstrap_from_ego=false \
#   --policy.worldflow_loss_weight=0.02 \
#   --policy.worldflow_geo_loss_weight=0.002 \
#   --policy.worldflow_bridge_loss_weight=0.005 \
#   --policy.worldflow_equiv_loss_weight=0.001 \
#   --policy.worldflow_pretrained_lr_multiplier=0.2 \
#   --policy.worldflow_new_lr_multiplier=4.0 \
#   --policy.worldflow_trans_weight=1.0 \
#   --policy.worldflow_rot_weight=1.0 \
#   --policy.worldflow_max_points=0 \
#   --policy.worldflow_require_action_target_sidecar=true \
#   --policy.pose9_action_noise_enable=false \
#   --policy.worldflow_noise_coupling=conjugate_ego \
#   --policy.worldflow_frame_origin=current_ee \
#   --policy.worldflow_action_fusion=conjugate_residual_consensus \
#   --policy.worldflow_augmentation_trans_scale=0.05 \
#   --policy.worldflow_augmentation_rot_scale=0.2 \
#   --policy.worldflow_se3_head_enable=false \
#   --policy.se3_enable=true \
#   --policy.se3_twist_head_mode=pose9_chart_endpoint \
#   --policy.se3_noise_trans_scale=0.10 \
#   --policy.se3_noise_rot_scale=0.10 \
#   --policy.se3_noise_gripper_scale=0.10 \
#   --policy.flow_time_sampling=integration_grid \
#   --policy.flow_time_zero_probability=0.25 \
#   --policy.se3_final_correction_enable=false
#
# These weights and the two-timescale optimizer are the configuration that
# produced the 48/50 Task-8 checkpoint and its 43/50 same-checkpoint causal
# ablation. This is a main/auxiliary loss scale, not the rejected 99:1 camera
# allocation. Do not add a residual gate and do not freeze Ego: World, Ego,
# both cross-attention directions, and the shared Action Expert train together.


# Real RGB-D collection and moving-camera processing
# See scripts/real_setting/README_CAMERA_MOTION.md. Main entrypoint:
#
#   bash benchmarks/song_real_libero/scripts/real_setting/song_rgbd_pipeline.sh --help

# Managed entrypoint

The commands above are recorded explicitly for auditability. The canonical
launcher adds path/hash/config checks, refuses busy GPUs by default, and writes
all logs below the experiment root:

```bash
benchmarks/song_real_libero/scripts/run_v043_dualview_baseline_guard.sh preflight
benchmarks/song_real_libero/scripts/run_v043_dualview_baseline_guard.sh cache
benchmarks/song_real_libero/scripts/run_v043_dualview_baseline_guard.sh validate-cache
benchmarks/song_real_libero/scripts/run_v043_dualview_baseline_guard.sh train

# Save and screen the entire 100/200/300/400/500 trajectory. Select an
# immutable Stage 1 checkpoint only after its online gate passes.
SONG_V043_EVAL_CKPT=/absolute/selected/stage1/checkpoint/pretrained_model \
SONG_V043_EVAL_TAG=stage1_fps_union_selected_parallel30 \
  benchmarks/song_real_libero/scripts/run_v043_dualview_baseline_guard.sh eval

SONG_V043_STAGE1_CKPT=/absolute/passed/stage1/checkpoint/pretrained_model \
benchmarks/song_real_libero/scripts/run_v043_dualview_baseline_guard.sh train-worldflow
```
