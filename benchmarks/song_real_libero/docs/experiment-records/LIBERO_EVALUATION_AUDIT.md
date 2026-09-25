# LIBERO Evaluation Audit

> Active project target: local and server evaluation use one uninterrupted
> rollout per LIBERO initial state. Only control frequency and horizon differ
> from the strict benchmark: `0.75 Hz + 600 steps`. Reset-and-retry, best-of-N,
> multi-sample action selection, and task-specific recovery are forbidden.

This document records the evaluation protocol and the local evidence used to
compare WEP-VLA v0.4 checkpoints. It is intentionally separate from training
documentation: a low offline flow-matching loss and an online LIBERO success
rate measure different things.

## Main conclusions

1. No checkpoint averaging is used. The active candidate is the explicit
   `024000_after32k_after32k` checkpoint with model SHA256
   `b32f9dd2bfa0f0627bb2c83990ee04b30b1787fd7f510c442322439b26c3db35`.
   Its archived complete single-checkpoint reports are Spatial `98/100`, Object
   `98/100`, Goal `77/100`, and LIBERO-10 `79/100`, the strongest complete
   four-suite evidence currently available for one immutable checkpoint.
2. That archived run is not the active deterministic protocol. Its report records
   `control_freq=5 Hz`, `max_steps=1000`, `action_index=0`, and
   `exec_action_steps=12`. This allows up to about 200 seconds of simulated
   control time for a spatial task.
3. The standard protocol used here is `20 Hz` with suite horizons
   `220/280/300/520/400` for spatial/object/goal/long-horizon/90. Spatial thus
   has about 11 seconds. A 5 Hz / 1000-step score must not be compared directly
   with a 20 Hz / 220-step score.
4. With the same explicit 20k checkpoint and `0.002` gripper delta threshold,
   the standard serial spatial result was `86/100` for action index 0 / execute
   12, and `89/100` for action index 1 / execute 16.
5. Dynamic inference batching is not numerically equivalent to serial
   inference for this checkpoint. Even very small action differences can
   bifurcate contact-rich rollouts. Comparable scores must use actual inference
   batch size 1.
6. An attempted deterministic sparse-voxel rewrite was functionally
   incompatible with this checkpoint: it changed PointSeg's input set,
   Tokenizer voxel boundaries, and equal-score point selection. In a controlled
   Goal task-3 episode-2 A/B, the rewritten path never moved the drawer, while
   the training-compatible path opened it by 10.3 cm and proceeded to the bowl.
   The rewrite was removed. The active model keeps checkpoint-compatible
   PointSeg, voxelization, and `topk`; only LitePT serialization-order shuffling
   is disabled by `model.eval()`, because it is an augmentation rather than a
   learned operation. Fixed seeds are not described as cross-process bitwise
   determinism.
7. Independent model processes remain a throughput mode, but comparable scores
   keep inference batch size 1. The evaluator also advances each task's LIBERO
   hard-reset sequence to the requested official episode index, so targeted and
   episode-sharded runs see the same randomized static-body layout as serial runs.
8. Every initial state is evaluated exactly once. A failed episode remains a
   failure; the evaluator never restores the initial state to try another flow
   sample. The configured `max_steps` is also passed into robosuite's internal
   `horizon`, so values above 1000 are no longer silently truncated at step 1000.
9. Reset settling directly sets the Panda fingers fully open. Robosuite keeps a
   separate integrated gripper target, so it must be synchronized to the finger
   qpos after settling. Without this synchronization, a later zero command
   silently drove the fingers from 0.08 m toward half-open. Gripper width events
   are also aligned to the current trajectory row rather than fired one row early.
10. The active controller normally executes the first 12 predicted rows. It may
    continue through rows 13 and 14 only when the achieved end-effector differs
    from the latest commanded pose by more than 9 mm or 0.10 rad, but not when
    the error already exceeds the 30 mm / 0.15 rad stale-chunk safety bound.
    This is a task-independent robot tracking check: object state, contacts,
    language, predicates, and success are not used to decide continuation.
11. The active controller executes 16 rows only after measured finger width lies
    in `(3 mm, 70 mm)` and the end effector has subsequently risen by at least
    15 mm. Width alone also fires while holding a fixed drawer handle; requiring
    lift makes this a robot-only proxy for a transported object. On the final
    checkpoint-compatible path, `t7e3` succeeded at step 377 and `t0e1`
    succeeded at step 336. A global 16-row policy failed `t0e1`, so the
    extension must remain conditional and its settings must be recorded.

## Checkpoint identity

Do not use a mutable `checkpoints/last` path in a report intended for
comparison. Use an explicit checkpoint directory and retain the model hash.

| Checkpoint | Model SHA256 | Local archived suite report |
| --- | --- | --- |
| `016000_after32k_after32k` | `51dd421cea5c5cfe32e6eeb37494dcff81aea487702cabbd2b8d0b14082d4834` | no complete v0.4 report found |
| `020000_after32k_after32k` | `bdeeda177cf0860a6a14b09ca5188970d1f1f9717f564e211ff468339fc4ff43` | historical 20k report plus standard spatial audit |
| `024000_after32k_after32k` | `b32f9dd2bfa0f0627bb2c83990ee04b30b1787fd7f510c442322439b26c3db35` | historical 24k reports; no complete standard-protocol four-suite report |

The historical reports predate checkpoint hashing in the evaluator and often
store `checkpoints/last`. Their directory labels are useful evidence but do not
provide the same immutable identity as a modern report.

## Archived WEP-VLA v0.4 results

All rows below came from `song_real_libero/outputs/wepvla_v04*`. They use
5 Hz, action index 0, and execute 12 unless stated otherwise. A dash means the
suite was not completed, not a zero success rate.

| Archived run | Spatial | Object | LIBERO-10 | Goal |
| --- | ---: | ---: | ---: | ---: |
| `wepvla_v04_28k_after_3w` | 86/100 | 89/100 | — | — |
| `wepvla_v04_32k_after_3w_thresh_0.002` | 91/100 | 96/100 | — | — |
| `wepvla_v04_32k_after_3w_thresh_0.003` | 94/100 | 98/100 | 78/100 | 70/100 |
| `wepvla_v04_20k_after_32k_after_3w_thresh_0.002` | **99/100** | 89/100 | 53/80 | — |
| `wepvla_v04_24k_after_32k_after_3w_thresh_0.004` | 92/100 | 96/100 | 46/80 | — |
| `wepvla_v04_24k_after_32k_after_3w_thresh_0.003` | 89/100 | 95/100 | 36/50 | — |

For the historical 20k spatial run, successful episodes had median 123 steps;
three successful episodes finished after step 220 (283, 244, and 465). More
importantly, changing 5 Hz to 20 Hz changes how long every command is held, so
the standard result cannot be reconstructed merely by truncating the old log.

## Controlled standard-protocol experiments

These archived experiments explicitly used the current 20k model hash above,
the pre-fix LitePT/PointSeg sparse-input implementation, fixed policy
noise seed 0, environment seed 7, 20 Hz, the 220-step spatial horizon, and
gripper delta threshold 0.002.

| Mode | Action rows | Spatial | Per-task successes | Elapsed |
| --- | --- | ---: | --- | ---: |
| serial, batch 1 | index 0, execute 12 | 86/100 | `9,7,10,10,9,9,10,7,9,6` | 941.8 s |
| serial, batch 1 | index 1, execute 16 | **89/100** | `8,8,10,9,10,8,10,10,9,7` | 746.0 s |
| two independent models, batch 1 | index 0, execute 12 | 83/100 | `9,7,10,10,8,7,10,8,9,5` | 536.3 s |

Index 1 skips the near-identity first UMI row. Executing all 16 trained action
rows reduced model calls by about 31% and improved this controlled run by three
episodes. Earlier short-replan testing with execute 5 was substantially worse,
so non-blocking chunk execution is not the dominant failure source. The model
benefits from the temporal coherence of its trained chunk.

## Active same-checkpoint weak-state audit

The active profile keeps the immutable 24k checkpoint, action index 0, a base
execute length of 12, a tracking-conditioned maximum of 14, gripper delta
threshold 0.002, policy seed 0, and environment seed 7. Only the allowed control
frequency and horizon were changed to 0.75 Hz and 600 steps. Instead of rerunning
every 100-episode suite while tuning, the audit replayed historical failure
initial states exactly once after canonical hard-reset alignment and with the
same PointSeg/LitePT preprocessing function used to train the checkpoint.

No checkpoint averaging, model soup, parameter interpolation, action averaging,
or test-time ensemble is used. Every result below comes from the single 24k
checkpoint hash listed above and one Flow Matching sample per replan.

- Restoring checkpoint-compatible PointSeg/LitePT preprocessing was required.
  The deterministic-voxel experiment made Goal task 3 episode 2 execute the
  wrong objects without opening the drawer. The restored path opened the top
  drawer by 10.3 cm; the episode still failed because the policy put the bowl
  down before completing the drawer subgoal.
- With the final checkpoint-compatible controller state, all four known failures
  from the archived 24k Spatial/Object reports succeeded once: Spatial task 0
  episode 8 (75 steps) and task 5 episode 7 (106); Object task 2 episode 4 (133)
  and task 6 episode 3 (224). These are targeted regressions, not a fresh
  100-episode score.
- The bounded tracking continuation keeps 12 rows as the base, allows rows
  13-14 only for moderate tracking error, and replans instead of continuing when
  error exceeds 3 cm or 0.15 rad. This prevents a blocked robot from following
  an increasingly stale suffix.
- The transported-grasp extension executes 16 rows only after physical gripper
  width remains between 3 mm and 70 mm and the end effector has lifted at least
  15 mm from the closure point. Width alone confused a fixed drawer handle with
  a carried object. On the final checkpoint-compatible path, Long task 7
  episode 3 succeeded at step 377 with 51 extended rows, and Long task 0
  episode 1 succeeded at step 336 with 36 extended rows. A second historical
  Long failure, task 7 episode 6, had already succeeded with the earlier
  width-conditioned path at step 558. Globally executing 16 rows failed task 0
  episode 1, so the extension remains state-conditioned.
- Goal task 3 episode 2 still failed at 600 steps and left the top drawer only
  1.2 cm from its
  initial position; an apparent 478-step success came from the rejected
  deterministic-voxel implementation and is excluded because its modeling
  source hash differs. Long task 4 episode 2 also still failed with only its
  first predicate complete. These failures are not hidden by retries.
- Additional final-candidate successes include Goal task 6 episodes 2 and 7,
  Long task 0 episode 4, and Long task 9 episode 4. Remaining targeted failures
  include Goal task 5 episode 0, Long task 4 episode 2, and Long task 8 episode
  3. They exhibit incorrect subgoal/placement geometry, not controller tracking
  failure. Enabling delayed-release suffix preservation on Long task 4 episode
  2 executed 47 extra suffix rows but still missed the second plate by about
  14 cm, so that experimental path remains disabled.
- Goal task 5 episode 0 remains a genuine policy failure. Execute 12, 16, and 32
  all failed. Adaptive execution increased plate displacement from roughly
  1.2 cm to 5.1 cm, but moved it away from the goal region; after the initial
  interaction the policy contacted the table instead of maintaining plate
  contact. A controller-only replay that preserved every recorded arm command
  but held the gripper fully open moved the plate only 3.3 mm toward the target
  direction and still failed. More controller time or a task-specific gripper
  override therefore cannot repair this incorrect policy subgoal.

These targeted results justify the active profile but are not a replacement
for a final complete suite report. They must not be presented as a measured
99/100 Object score; the evidence is one confirmed remaining historical weak
state, not a fresh 100-episode sweep.

## Why low action loss does not imply nearly 100% success

`loss_action` is flow-matching velocity-field MSE averaged over sampled flow
time, action steps, and dimensions. Online inference then integrates the field,
converts the trajectory to controller targets, executes a chunk, observes a new
scene, and repeats. It does not directly optimize:

- final object placement tolerance;
- contact, grasp, or release success;
- task reward;
- robustness to flow noise or sparse-kernel numerical variation;
- accumulated closed-loop pose error;
- completion before a fixed benchmark horizon.

Consequently, an action loss around `3e-4` can coexist with a wrong subgoal,
one failed grasp, a few millimetres of terminal error, or a rollout that needs
more than the standard horizon. The rollout videos and action diagnostics show
both near-miss failures and genuine repeated-subgoal loops; this is not
explained by one gripper-timing bug.

## Gripper control

LIBERO's Panda gripper input is directional (`-1=open`, `+1=close`), not a
physical width target. The default evaluator therefore uses the temporal
derivative of predicted physical width, aligned to the row being executed:

```text
delta_i = predicted_width[i] - predicted_width[i - 1]
delta_i > +0.002  -> -1 (open)
delta_i < -0.002  -> +1 (close)
otherwise         ->  0 (hold current actuator target)
```

For the first selected row, `width[i - 1]` is taken from the preceding row of
the full predicted chunk when one exists; row zero uses itself. This avoids a
spurious event at the chunk boundary. After initial settling, the evaluator
also synchronizes `PandaGripper.current_action` and actuator controls with the
physical finger qpos, so a zero command actually holds the fully-open state.

`target_width` remains a diagnostic mode only. Closed-loop target tracking can
chatter when the measured gripper lags, contacts an object, overshoots, or the
next predicted chunk changes its target. There is no task-specific latch, rim
correction, or manually coded object rule in the standard path.

## Recommended commands

### Reproducible 0.75 Hz / 600-step single-suite comparison

Use this mode to compare checkpoints. It is intentionally serial.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh_post/checkpoints/last/pretrained_model \
  --suite libero_spatial \
  --all-tasks \
  --episodes 10 \
  --isolated-policy-workers 1 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --control-freq 0.75 \
  --max-steps 600 \
  --action-index 0 \
  --exec-action-steps 12 \
  --adaptive-exec-max-steps 14 \
  --adaptive-exec-position-error-threshold 0.009 \
  --adaptive-exec-rotation-error-threshold 0.10 \
  --adaptive-exec-position-error-max 0.03 \
  --adaptive-exec-rotation-error-max 0.15 \
  --grasp-exec-steps 16 \
  --grasp-width-min 0.003 \
  --grasp-width-max 0.07 \
  --grasp-lift-threshold 0.015 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --synchronize-gripper-controller-state \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode viewer3d \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_isolated_spatial
```

### Four GPUs, one suite per GPU, serial within each suite

This is the closest multi-GPU equivalent of four independent serial suite
runs. It uses one model per GPU.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /absolute/path/to/checkpoints/024000_after32k_after32k/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_goal \
  --suite libero_10 \
  --suite-gpu-ids 0,1,2,3 \
  --all-tasks \
  --episodes 10 \
  --isolated-policy-workers 1 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --control-freq 0.75 \
  --max-steps 600 \
  --action-index 0 \
  --exec-action-steps 12 \
  --adaptive-exec-max-steps 14 \
  --adaptive-exec-position-error-threshold 0.009 \
  --adaptive-exec-rotation-error-threshold 0.10 \
  --adaptive-exec-position-error-max 0.03 \
  --adaptive-exec-rotation-error-max 0.15 \
  --grasp-exec-steps 16 \
  --grasp-width-min 0.003 \
  --grasp-width-max 0.07 \
  --grasp-lift-threshold 0.015 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --synchronize-gripper-controller-state \
  --no-use-suite-max-steps \
  --no-recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --no-save-video \
  --output-dir /absolute/path/to/eval_4suite_strict
```

### Four GPUs, two independent models per GPU

Use this for faster private-policy-process evaluation on 24 GB cards. It loads
eight checkpoint copies in total and assigns each model five tasks. Do not also
enable task or episode parallelism.

```bash
# Use the same command as above, but replace the worker options with:
  --isolated-policy-workers 2 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1
```

One model used roughly 7–8 GB in the measured setup. Two copies per 24 GB GPU
worked; three copies are not recommended without measuring peak memory. Ten
independent task models per GPU do not fit.

## Modes that must not be mixed in one comparison table

- 0.75 Hz / 600 steps, archived 5 Hz / 1000 steps, and standard suite horizons;
- mutable `checkpoints/last` versus explicit checkpoint hashes;
- serial batch 1 versus dynamic inference batch 10;
- one model versus two independent policy workers;
- different action index / executed chunk length;
- different gripper delta thresholds;
- different initial scene settling or initial gripper state;
- single-rollout evaluation versus reset-and-retry or best-of-N evaluation;
- one policy sample per replan versus multi-sample/ensemble action selection;
- save-video/viewer runs versus headless timing measurements.

## Saved evidence

Modern reports include the requested and resolved checkpoint paths, model
SHA256, policy/config/code hashes, package versions, GPU/driver inventory,
determinism settings, control settings, suite horizons, and worker mode.
Per-episode `actions.npz` contains model actions, LIBERO commands, controller
targets, achieved poses, every complete predicted action chunk (including its
unexecuted suffix), observable object poses, tracking errors,
chunk-boundary errors, gripper diagnostics, per-step goal-predicate values,
per-chunk grasp-lift displacement and transported-grasp flags, non-robot
slide/hinge joint values, and end-effector/scene contact pairs. These
extra fields are post-hoc diagnostics only and never enter policy inference or
controller decisions. `progress.json` and
`evaluation_events.jsonl` are updated during
evaluation, including isolated-policy mode.

The evaluator records both `max_steps` and the actual robosuite
`environment_horizon`. It aborts before rollout if the environment horizon is
shorter than the requested limit. Raising the limit therefore makes one longer
episode; it never creates multiple attempts. The active profile stops at 600
steps; increasing Long task 4 episode 2 to 1000 steps did not complete its
second placement.

The fixed flow-noise seed also seeds random choices inside one model forward,
but it does not make sparse CUDA execution bitwise deterministic. The active
path deliberately retains the checkpoint's original sparse voxelization and
score selection; replacing them for determinism changed model function and
degraded behavior. Disabling LitePT order shuffling in evaluation removes one
augmentation source, but contact-rich rollouts can still amplify sparse-CUDA
differences. Retain input, code, package, driver, and checkpoint hashes in every
report, and treat separate-process or cross-hardware equality as statistical
reproducibility rather than bitwise identity.

This residual effect is measurable even on one host. Two Goal task-3 episode-1
runs had identical first-input hashes but their first action chunks differed by
up to about `0.0020`, with a `0.28 mm` difference in the first controller
target. One rollout failed at 600 steps while the other succeeded at step 278;
because the trajectories diverged before step 600, that comparison is not
evidence that a larger horizon caused the success.

## Targeted Goal diagnosis on the fixed 24k checkpoint

The strict single-rollout tests on 2026-07-23 rule out several tempting global
controller changes.  For `libero_goal` task 3, historical failure episode 2
still failed with each of the following changes applied independently:

- execute 14 rows per chunk instead of 12;
- change the deterministic flow-noise base seed from 0 to 1;
- construct 50,000 input points instead of 10,000.
- continue the optional suffix after any peak tracking error in the base segment,
  rather than using only the final base-waypoint error.

The 14-row run did not merely need more time: it failed to open the drawer at
all, whereas the 12-row run opened it by about 2.1 cm before switching to the
bowl.  The 50,000-point and peak-error continuation runs also failed to open
the drawer.  Consequently none of these settings is promoted to a global
default, and the failed peak-error experiment was removed from the production
control path.

Under the earlier 12-to-14 adaptive profile with grasp extension disabled,
task-3 episode 5 succeeded, while episodes 2, 6, 7, and 9 failed. Recorded
state separates the mechanisms:

- episodes 2 and 9 did not open the drawer sufficiently;
- episodes 6 and 7 opened it fully but released or moved the bowl into the
  wrong region;
- controller pose tracking was already accurate in episode 2, so additional
  target holding cannot repair that policy-level subgoal switch.

This is evidence of closed-loop strategy/generalization failures in this
checkpoint, not evidence for reset retries, task-specific corrections, or
privileged goal-state control.  The archived complete evaluation of the same
single checkpoint remains 98% Spatial, 98% Object, 77% Goal, and 79%
LIBERO-10.  A 99% claim for Goal or LIBERO-10 is therefore not supported by the
current weights.




MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-alignment current_minus_previous \
  --synchronize-gripper-controller-state \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/libero_setting/train_libero_fresh_post/checkpoints/last/pretrained_model \
  --suite libero_spatial \
  --all-tasks \
  --episodes 10 \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 16 \
  --adaptive-exec-max-steps 16 \
  --grasp-exec-steps 16 \
  --gripper-delta-threshold 0.002 \
  --max-steps 600 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_fixed16_spatial






  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v031_v7_adapter_libero_after_3w2/checkpoints/032000/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_10 \
  --suite libero_goal \
  --suite-gpu-ids 0,1,2,3 \
  --all-tasks \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 9 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 12 \
  --adaptive-exec-max-steps 16 \
  --grasp-exec-steps 16 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/32k_after_32k_freq_20






# GOOD isolated-policy-workers 
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v031_v7_adapter_libero_after_3w2_after3w2_after2w4/checkpoints/026000/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_10 \
  --suite libero_goal \
  --suite-gpu-ids 0,1,2,3 \
  --all-tasks \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 9 \
  --task-workers 1 \
  --episode-workers-per-task 1 \
  --inference-batch-size 1 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 16 \
  --adaptive-exec-max-steps 16 \
  --grasp-exec-steps 16 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/26000_freq_20

# GOOD multi thread
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v031_v7_adapter_libero_after_3w2_after3w2_after2w4/checkpoints/026000/pretrained_model \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_10 \
  --suite libero_goal \
  --suite-gpu-ids 0,1,2,3 \
  --all-tasks \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 10 \
  --episode-workers-per-task 1 \
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \d
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 16 \
  --adaptive-exec-max-steps 16 \
  --grasp-exec-steps 16 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/26000_freq_20_multithread


===================================================

# Perfect  Version
 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v041_libero_dataset_fixed_v1_after4k_after8k_after10k/checkpoints/030000/pretrained_model \
  --suite-gpu-ids 0,1,2,3 \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_10 \
  --suite libero_goal \
  --all-tasks \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 10 \
  --episode-workers-per-task 1\
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 1 \
  --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 \
  --grasp-exec-steps 24 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_best_action_index1_3






#########PARAMETER CORRECT##########
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wepvla_v042_0801_after56k/checkpoints/052000/pretrained_model \
  --suite-gpu-ids 0,1,2,3 \
  --suite libero_spatial \
  --suite libero_object \
  --suite libero_10 \
  --suite libero_goal \
  --all-tasks \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 10 \
  --episode-workers-per-task 1\
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
  --control-freq 20 \
  --action-index 1 \
  --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 \
  --grasp-exec-steps 24 \
  --max-steps 1000 \
  --no-use-suite-max-steps \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_52k_after_54k_wepvla_v042_0803_action_index1

=======================SMALL TEST============================



 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v041_libero_dataset_fixed_v1_after4k/checkpoints/008000/pretrained_model \
  --suite-gpu-ids 0 \
  --suite libero_10 \
  --task-id 3 \
  --task-id 4 \
  --task-id 6 \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 4 \
  --episode-workers-per-task 5\
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.004 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
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
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_temp_long_chunk24_freq20_thresh4_2






 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v041_libero_dataset_fixed_v1_after4k/checkpoints/008000/pretrained_model \
  --suite-gpu-ids 1 \
  --suite libero_10 \
  --task-id 3 \
  --task-id 4 \
  --task-id 6 \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 4 \
  --episode-workers-per-task 5\
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
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
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_temp_long_chunk24_freq20_thresh2_2

#####  GOAL

   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v041_libero_dataset_fixed_v1_after4k/checkpoints/008000/pretrained_model \
  --suite-gpu-ids 0 \
  --suite libero_goal \
  --task-id 3 \
  --task-id 5 \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 2 \
  --episode-workers-per-task 10\
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
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
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_temp_goal_chunk24_freq20_thresh2



   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v041_libero_dataset_fixed_v1_after4k/checkpoints/008000/pretrained_model \
  --suite-gpu-ids 1 \
  --suite libero_goal \
  --task-id 3 \
  --task-id 5 \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 2 \
  --episode-workers-per-task 10\
  --inference-batch-size 20 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.004 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
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
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_temp_goal_chunk24_freq20_thresh4



######################
#########PARAMETER CORRECT##########
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model \
  --suite-gpu-ids 0,1,2,3 \
  --suite libero_object \
  --suite libero_spatial \
  --suite libero_10 \
  --suite libero_goal \
  --all-tasks \
  --episodes 50 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 10 \
  --episode-workers-per-task 2 \
  --inference-batch-size 80 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
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
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/30k_eval_wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5_FULL



######################
#########PARAMETER CORRECT##########
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5   python \
  benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py \
  --config benchmarks/song_real_libero/configs/libero.json \
  --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5/checkpoints/030000/pretrained_model \
  --suite libero_goal \
  --all-tasks \
  --episodes 10 \
  --policy-noise-seed 0 \
  --env-seed 7 \
  --isolated-policy-workers 1 \
  --task-workers 10 \
  --episode-workers-per-task 5 \
  --inference-batch-size 50 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode delta_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --initial-gripper-open \
  --settle-keep-robot-fixed \
  --synchronize-gripper-controller-state \
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
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/30k_eval_wep_vla_v042_general_dataset_toolseg_after32k_mul3_after28k_lr5_goal_temp






PYOPENGL_PLATFORM=egl  CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5   python   benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py   --config benchmarks/song_real_libero/configs/libero.json   --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_after32k/checkpoints/028000/pretrained_model    --suite libero_10   --task-id 4    --task-id 6    --task-id 8    --task-id 9    --episodes 50   --policy-noise-seed 0   --env-seed 7   --isolated-policy-workers 1   --task-workers 4   --episode-workers-per-task 10  --inference-batch-size 40   --no-release-event-exec-enable   --waypoint-max-hold-steps 1   --gripper-control-mode delta_width   --gripper-delta-threshold 0.002   --gripper-delta-alignment current_minus_previous   --initial-gripper-open   --settle-keep-robot-fixed   --synchronize-gripper-controller-state   --control-freq 20   --action-index 0   --exec-action-steps 24   --adaptive-exec-max-steps 24   --grasp-exec-steps 24   --max-steps 1000   --no-use-suite-max-steps   --recreate-env-per-episode   --render-mode offscreen   --no-visualize-foreground   --save-video   --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_wep_vla_v042_general_dataset_toolseg_all_train_long4689_28k_after32k_after32k




PYOPENGL_PLATFORM=egl  CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5   python   benchmarks/song_real_libero/scripts/libero_setting/libero_pointcloud_eval.py   --config benchmarks/song_real_libero/configs/libero.json   --policy.path /opt/data/private/liusong/benchmarks/song_real_libero/outputs/wep_vla_v042_general_dataset_toolseg_after32k_after32k_after32k/checkpoints/028000/pretrained_model    --suite libero_goal   --task-id 2    --task-id 3    --task-id 5    --task-id 6  --task-id 9    --episodes 50   --policy-noise-seed 0   --env-seed 7   --isolated-policy-workers 1   --task-workers 5   --episode-workers-per-task 10  --inference-batch-size 40   --no-release-event-exec-enable   --waypoint-max-hold-steps 1   --gripper-control-mode delta_width   --gripper-delta-threshold 0.002   --gripper-delta-alignment current_minus_previous   --initial-gripper-open   --settle-keep-robot-fixed   --synchronize-gripper-controller-state   --control-freq 20   --action-index 0   --exec-action-steps 24   --adaptive-exec-max-steps 24   --grasp-exec-steps 24   --max-steps 600   --no-use-suite-max-steps   --recreate-env-per-episode   --render-mode offscreen   --no-visualize-foreground   --save-video   --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_wep_vla_v042_general_dataset_toolseg_all_train_goal23569_28k_after32k_after32k_after32k




######################
#########PARAMETER CORRECT##########
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
  --max-steps 800  \
  --no-use-suite-max-steps  \
  --isolated-policy-workers 1 \
  --task-workers 3 \
  --episode-workers-per-task 25 \
  --inference-batch-size 75 \
  --no-release-event-exec-enable \
  --waypoint-max-hold-steps 1 \
  --gripper-control-mode target_width \
  --gripper-delta-threshold 0.002 \
  --gripper-delta-alignment current_minus_previous \
  --control-freq 20 \
  --action-index 0 \
  --exec-action-steps 24 \
  --adaptive-exec-max-steps 24 \
  --grasp-exec-steps 24 \
  --recreate-env-per-episode \
  --render-mode offscreen \
  --no-visualize-foreground \
  --save-video \
  --output-dir benchmarks/song_real_libero/outputs/libero_setting/eval_strict_official_maxstep800



python -m pip install --no-deps --force-reinstall \
  "mujoco==2.3.1.post1"



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