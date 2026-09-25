# Single-view WorldFlow: robot-base EEF trajectory contract

## Definition

WorldFlow predicts the future commanded end-effector trajectory in the
**complete robot-base coordinate frame**. It does not predict simulator object
poses or an explicit dense point-flow field.

For any pose or point `X`, conversion is

```text
T_base_X = inverse(T_world_base) @ T_world_X
```

Both the rotation and translation of `T_world_base` are applied. The fixed
camera extrinsic is used only to reconstruct geometry. The current EEF pose
maps the segmented current-EEF foreground cloud into robot-base coordinates.

## Why this still represents object flow

The World branch receives task-relevant foreground points and is supervised by
the EEF trajectory in the same fixed frame. During grasp or contact, the EEF
contact point moves with the object, so its trajectory is a sparse controlled
point-flow trace on the object. The model learns this relationship implicitly;
no object ID, object-pose label, correspondence field, or dense flow target is
introduced.

Free-space approach and post-contact motion remain parts of the commanded EEF
trajectory. Therefore this target should be described as World-frame EEF flow
that implicitly carries object-motion information, not object ground-truth
flow.

## Independent double-flow contract

`worldflow_target_type="world_eef_trajectory"` enforces:

- `worldflow_reference_frame="robot_base"`
- foreground scene XYZ and EEF targets in the complete robot-base frame
- independent World and Ego flow priors
- token cross-attention as the only World/Ego interaction
- no endpoint residual/rate, analytic bridge loss, coordinate augmentation, or
  shared parameters

World uses an SE(3) geodesic flow and a direct six-dimensional twist head even
when the pretrained Ego branch retains its legacy pose chart.

For the focused 100-demo experiment, `worldflow_bootstrap_from_ego=true` is a
one-shot initialization only: compatible trained point encoders, adapters,
time embeddings, and (when configured) the complete Action Expert are copied
by value into distinct World parameters before training. No storage is shared.
The direct World SE(3) output head remains independently initialized, and the
World-to-Ego cross-attention output projection starts at zero so enabling the
branch initially preserves the loaded baseline policy function.

`worldflow_action_expert_mode="independent"` is required for the corrected
dual-expert experiment. The historical `"shared"` mode is retained only so old
checkpoints remain loadable. In shared mode, World owned its token front-end
but its trajectory tokens were still processed by the Ego Action Expert. Once
protected-Ego training froze that expert, World had to learn around a fixed
Ego-specific temporal mapping and could interact with Ego only after the final
expert layer. This was not a genuinely independent double-flow architecture.

In independent mode:

- Ego uses the frozen pretrained Action Expert and unchanged Ego suffix.
- World uses a separately parameterized Action Expert bootstrapped from Ego.
- World Expert input contains Ego/World global scene tokens and only World
  action tokens; it never consumes Ego action tokens.
- the completed Ego and World trajectory-token sequences exchange information
  through explicit bidirectional cross-attention.
- World supervision updates the World Expert directly, while the baseline Ego
  weights and buffers remain byte-identical.

## Focused protected-Ego result before the dual-expert correction

The shared-expert protected-Ego run was evaluated with the same fixed seeds and
hard gate (task 6 stops at its fourth failure because it can no longer exceed
the `46/50` baseline):

| step | task 6 result at stop | outcome |
|---:|---:|---|
| 260 | 41/45, 4 failures | disqualified |
| 520 | 35/40, 5 failures | disqualified |
| 780 | 39/44, 5 failures | disqualified |
| 1040 | 45/50, 5 failures | below baseline |
| 1300 | 46/50, 4 failures | equals baseline |
| 1564 | 46/50, 4 failures | equals baseline |
| 1824 | 39/43, 4 failures | disqualified |
| 2084 | 39/43, 4 failures | disqualified |

All pretrained Ego tensors were audited byte-identical at step 260. Therefore
the result rules out catastrophic forgetting but does not establish WorldFlow
benefit. Extending training past step 1564 made performance worse, so further
step-count or learning-rate tuning is not justified. The next experiment must
change the shared Action Expert bottleneck itself.

## Independent Action Expert result

The corrected independent-Expert run used the same 100 demonstrations, fixed
evaluation seeds, protected pretrained Ego weights, and the original 1564-step
schedule. Every pretrained Ego tensor remained byte-identical. The hard-gated
results were:

| step | task 6 | task 8 | outcome |
|---:|---:|---:|---|
| 260 | 44/50 | not run | below `46/50` baseline |
| 520 | 44/50 | not run | below baseline |
| 780 | 47/50 | 45/50 | `+1 / +0`, total `92/100` |
| 1040 | 46/50 | not run | equals baseline |
| 1300 | 40/44 at hard stop | not run | cannot exceed baseline |
| 1564 | 42/46 at hard stop | not run | cannot exceed baseline |

This removes the shared-Expert bottleneck but still does not establish a
stable WorldFlow gain. Paired failures show both genuine repairs and new
regressions. At step 780, task 6 repairs baseline episodes `2/36/43` but adds
`23/25`; task 8 repairs `7/20/23/34`, retains `30`, and adds
`11/13/26/37`. Thus World information changes behavior, but the historical
final-latent World-to-Ego attention trades one failure set for another.

Two structural causes remain:

1. After the current-EEF foreground cloud is transformed into robot-base
   coordinates, the achieved current EEF pose is discarded as a model
   condition. The same nearly static foreground and instruction can then map
   to different future EEF trajectories at different rollout phases. The
   final training diagnostic (about `6.1 cm` translation error) confirms that
   this under-conditioned World branch is not an execution-quality trajectory
   predictor.
2. A final hidden-state attention update has no coordinate semantics. It asks
   the frozen Ego pose9 head to interpret an unconstrained World latent instead
   of converting the complete World trajectory into the controller's current
   EEF frame.

The next physical-execution contract therefore adds an explicit
`worldflow.current_ee_pose` robot-base token to the independent World Expert.
For arm execution it uses the independently predicted absolute trajectory via

```text
T_current_EEF_target = inverse(T_base_current_EEF) @ T_base_target_EEF
```

and retains Ego's gripper trajectory. Ego-to-World and World-to-Ego token
attention remain available for interaction, but the arm output is no longer
an unaligned latent perturbation. The equation is a left coordinate change,
not global-origin conjugation, endpoint residual, residual rate, learned gate,
or trajectory average.

## Dataset sidecars

- `world_base_ee_poses/episode_XXXXXX.npy`: achieved current EEF pose in
  robot-base coordinates, used to transform the foreground cloud.
- `world_base_action_target_ee_poses/episode_XXXXXX.npy`: commanded model-EEF
  target from each aligned raw LIBERO action, expressed in robot-base
  coordinates. Action chunks index this array exactly like the Ego labels.

The dataset wrapper requires strict robot-base metadata and refuses camera-frame
or achieved-future fallbacks in the new mode.

## Focused experiment

The authoritative single-flow baseline is
`eval_FULL4-9705*` (`481/500` on LIBERO-10). The focused tasks are task 8
(`45/50`) and task 6 (`46/50`), the two lowest baseline tasks. Their complete
50-demo HDF5 files provide exactly 100 training episodes.

Double-flow evaluation must reuse baseline environment seed 7, policy-noise
seed 0, official fixed initialization, control frequency 20, action index 0,
24 execution steps, and 50 episodes per task.

## Current-pose-conditioned direct-World execution result

The independent World Expert was then conditioned on the achieved current EEF
pose in robot-base coordinates. Training kept the original 1564-step LR
schedule but was stopped after the complete step-1300 checkpoint, as required.
The moving 50-batch World translation error improved from about `7.57 cm` near
step 353 to `5.39 cm` near step 1229, confirming that the missing current-pose
condition was a real under-conditioning error.

Directly executing that World trajectory was nevertheless decisively invalid:

| step | task 6 hard-gate result | outcome |
|---:|---:|---|
| 260 | 0/28 | disqualified |
| 520 | 0/28 | disqualified |
| 780 | 0/28 | disqualified |
| 1040 | 0/28 | disqualified |
| 1300 | 0/28 | disqualified |

At step 1300, the median Cartesian controller tracking error was about `5.2
cm`; all 28 completed episodes consumed the long failure horizon. Thus the
failure is not a checkpoint-selection or training-duration issue. A roughly
5-cm World predictor cannot replace the pretrained Ego controller as the
executed arm trajectory.

The current-pose condition remains necessary, but the
`world_trajectory_arm_ego_gripper` output route is rejected. The next causal
diagnostic loads the same immutable checkpoint with
`--worldflow-action-fusion-override cross_attention`. This changes only the
final execution route back to the jointly conditioned Ego trajectory; it is
recorded as a non-benchmark diagnostic and never edits the checkpoint files.

That diagnostic was hard-stopped at task 6 with `37/42` and five failures. Its
best possible final score was only `45/50`, below the `46/50` baseline, so task
8 was not run. It proves the direct-World `0/28` result was caused by the final
execution route rather than a corrupt checkpoint, but it also rejects the old
unconstrained final-latent World-to-Ego attention.

## Physical trajectory interaction

`physical_trajectory_cross_attention` replaces both rejected paths. At every
flow step, the two Action Experts first run independently and decode their own
complete endpoint proposals. Without dividing by the remaining path length:

```text
Ego endpoint in current EEF  -> T_base_current @ T_current_Ego_endpoint
World endpoint in robot base -> inverse(T_base_current) @ T_base_World_endpoint
```

Thus World attends to Ego's complete proposal in robot-base coordinates, while
Ego attends to World's complete proposal in current-EEF coordinates. Only
these physically aligned pose9 trajectory tokens enter the bidirectional
attention. The original raw World/Ego hidden-state attention is bypassed.

The interacted Ego hidden sequence is decoded by the unchanged pretrained Ego
action head; the World sequence is decoded by its independent SE(3) head. Both
attention output matrices start at zero, so bootstrap exactly preserves both
independent predictors, then learns a full token-to-token interaction. This is
not endpoint residual correction, residual rate, a scalar gate, trajectory
averaging, or direct World execution.

A real four-GPU one-step training smoke completed with batch 24 per rank and
six workers per rank, saved and reloaded its checkpoint, and completed one
online model call. After the first optimizer update, both physical interaction
attention output matrices were nonzero, while the frozen pretrained Ego action
head remained byte-identical to the baseline (`max_abs_diff=0.0`). The smoke
artifacts are under `SMOKE_world_eef_physicaltraj_1step_20260816` and
`SMOKE_EVAL_world_eef_physicaltraj_1step_20260816`.

## Focused physical-interaction result

The formal run declared the same 1564-step schedule as the preceding controls,
saved steps `260/520/780/1040/1300`, and was interrupted only after the complete
step-1300 checkpoint was durable (the training process reached step 1303 while
the checkpoint was being written). The output is:

```text
/opt/data/private/liusong/benchmarks/song_real_libero/outputs/
world_eef_task6_task8_100ep_physicaltraj_4gpu_b24_schedule1564_stop1300
```

Every checkpoint used the fixed task6/task8 seeds and 28 same-task episode
workers. A checkpoint had to strictly exceed task 6 baseline `46/50` before
task 8 was run, and task 8 then had to strictly exceed `45/50`.

| step | task 6 | task 8 | outcome |
|---:|---:|---:|---|
| 260 | 39/43 at 4-failure stop | not run | cannot exceed baseline |
| 520 | 48/50 | 45/50 | `+2 / +0`, total `93/100`; not qualified |
| 780 | 44/50 | not run | below baseline |
| 1040 | 41/45 at 4-failure stop | not run | cannot exceed baseline |
| 1300 | 45/50 | not run | below baseline |

Step 520 is causal evidence that physically aligned World interaction can help,
but not stable evidence of a two-task improvement. Against baseline failures
task6 `{2,26,36,43}` and task8 `{7,20,23,30,34}`, step 520 repaired every one
of those episodes. It introduced new task6 failures `{3,23}` and new task8
failures `{13,15,17,36,38}`. World therefore changes decisions in a useful but
still non-robust way rather than merely reproducing the frozen Ego policy.

Performance was not monotonic after step 520, so extending the same run to step
1564 cannot be justified as a convergence continuation. No learning-rate,
gradient, residual-rate, gate, or task-specific patch is applied. Under the
focused-goal stop rule, this experiment is stopped without advancing to
multi-view or all-suite evaluation.

## Corrected stochastic origin and base-frame velocity contract

The physical-interaction result exposed two remaining principle-level errors
that are independent of learning rate and training duration.

First, the World flow used an SE(3) prior sampled near the robot-base identity,
while the Ego flow used a pose prior in the current EEF frame. The two random
origins therefore did not denote the same physical EEF pose. For an absolute
robot-base EEF trajectory the correct coordinate map is

```text
T_base_noise = T_base_current @ T_current_noise
```

not the rejected motion conjugation
`T_base_current @ T_current_noise @ inverse(T_base_current)`. The new
`left_compose_ego` mode couples only this stochastic origin. The independent
World and Ego experts still predict and integrate their own vector fields
afterwards.

Second, an absolute World EEF pose was still integrated with a left-trivialized
SE(3) spatial twist. Its translational channel contains the global-origin
lever-arm term `omega × position`, so rotating an EEF away from the robot-base
origin spuriously moves its position around that origin. The corrected
`base_decoupled` velocity is

```text
[position_velocity_on_robot_base_axes,
 angular_velocity_on_robot_base_axes]
```

Position is added directly. Orientation is left-rotated on robot-base axes,
but that rotation never changes position. Translation follows a straight line
and orientation follows the SO(3) geodesic. The target vector field is constant
and the endpoint is recovered by multiplying by the remaining time; no
`1/(1-t)` residual rate is learned or executed.

Backward compatibility is explicit: old checkpoints that do not serialize
`worldflow_world_eef_velocity_mode` retain `legacy_spatial_twist`. The new
focused recipe explicitly writes both:

```text
worldflow_noise_coupling=left_compose_ego
worldflow_world_eef_velocity_mode=base_decoupled
```

Validation completed on 2026-08-16:

- 108 focused WorldFlow/SmolVLA tests passed.
- the mathematical tests distinguish left composition from conjugation and
  verify that pure EEF rotation has no global-origin translation.
- a real 4-GPU, batch-24-per-rank, six-worker-per-rank update completed and
  saved/reloaded a checkpoint at
  `SMOKE_world_eef_physicaltraj_leftcompose_decoupled_1step_20260816`.
- the reloaded checkpoint completed repeated online LIBERO model calls through
  the corrected integration path. The long 1-step random-head rollout was
  intentionally stopped after the execution path was established; it is not a
  performance measurement.

## Corrected formal checkpoint screen

The left-compose, base-decoupled run was trained with the original 1564-step
scheduler declaration and stopped after saving step 1300. The fixed task6/task8
screen produced:

| step | task 6 | task 8 | result |
|---:|---:|---:|---|
| 260 | 47/50 | 43/50 | 90/100; rejected |
| 520 | 45/50 | not run | rejected on task 6 |
| 780 | 45/50 | not run | rejected on task 6 |
| 1040 | 44/50 | not run | rejected on task 6 |
| 1300 | 41/46 observed, 5 failures | not run | best possible 45/50; stopped early |

For step 1300, episodes `{4,9,11,23,25}` completed as genuine failures after
990 environment steps. Episodes `{32,36,37,39}` were still running when the
strict-improvement upper bound fell below baseline and were deliberately
cancelled; their generated `KeyboardInterrupt` records are infrastructure
cancellations, not additional policy failures. No checkpoint improves both
tasks, so this corrected structural experiment does not qualify for ablation
or further training under the focused-goal rule.

## Four-GPU checkpoint evaluation

Future checkpoint sweeps use
`scripts/run_world_eef_multicheckpoint_eval.sh`. Up to four evaluator parents
load different checkpoints concurrently, one per GPU, and then remain resident
behind an evaluation gate. The scheduler releases one resident model at a time
with all 28 episode workers. This overlaps checkpoint loading without reducing
the established single-checkpoint simulation parallelism or oversubscribing
the 30-thread host with four simultaneous 28-worker environment pools.

Each parent receives both `--task-id 6` and `--task-id 8` while retaining
`--task-workers 1`, so the two tasks execute sequentially against the same
loaded model rather than loading the checkpoint twice. Example:

```bash
WORLD_EEF_TRAINING_DIR=/path/to/training \
WORLD_EEF_EXPERIMENT_DIR=/path/to/experiment \
WORLD_EEF_STEPS='000260 000520 000780 001040 001300' \
WORLD_EEF_GPU_IDS='0,1,2,3' \
bash benchmarks/song_real_libero/scripts/run_world_eef_multicheckpoint_eval.sh
```

Set `WORLD_EEF_DRY_RUN=1` to inspect GPU assignment, worker allocation, paths,
and complete evaluator commands without launching an evaluation.

## Independent dual-flow calibration after the rejection

The rejected run exposed a more basic violation than interaction quality. Its
last 100 updates averaged 7.20 mm / 0.63 degrees for Ego but 46.51 mm / 6.60
degrees for World (6.46x / 10.49x worse). The run froze the complete pretrained
Ego path, weighted the direct World flow and endpoint objectives by only
0.02/0.002, and nevertheless enabled physical bidirectional interaction from
the first update. A centimetre-accuracy stream cannot provide a useful control
correction to a millimetre-accuracy stream.

The target is not the source of this gap. On 16 uniformly spaced frames (416
valid chunk targets), the post-UMI Ego target transformed analytically by the
current EEF-to-base pose matches the stored robot-base World target to
4.45e-8 m mean translation error and zero measured rotation error. The frozen
baseline's prediction has identical error in either coordinate system: 9.73 mm
mean translation and 1.26 degrees mean rotation. Thus an approximately 10 mm
World predictor is demonstrably reachable with the existing labels.

The new `independent_parallel` calibration contract therefore does the
following before any interaction training:

- freezes only the pretrained VLM; both Action Experts, both point/action
  paths, both output heads, and all other active non-VLM modules are trainable;
- gives Ego and World their own complete direct trajectory supervision while
  bypassing both directions of cross-attention;
- uses equal base learning rates and no 0.02/0.002 attenuation of World direct
  supervision (`worldflow_loss_weight=1`, `worldflow_geo_loss_weight=1`);
- retains identity-initialized physical interaction modules in the checkpoint
  so they can be enabled after calibration without reinitializing either flow;
- gates interaction on unweighted physical endpoint errors, never on weighted
  scalar losses. The default 100-update gate requires Ego and World mean
  translation errors <=12 mm, World/Ego translation ratio <=1.5, World mean
  rotation <=2 degrees, and World/Ego rotation ratio <=1.5.

The gate implementation is
`benchmarks/song_real_libero/scripts/audit_worldflow_error_alignment.py`.
`run_world_eef_task6_task8_focused.sh pipeline` now stops after this gate and no
longer evaluates an uncalibrated World model.

A four-GPU one-update smoke run is stored at
`/opt/data/private/liusong/benchmarks/song_real_libero/outputs/SMOKE_world_eef_independent_parallel_equal_1step_20260816`.
It loaded 304M trainable parameters out of 666M total. Byte-level checkpoint
comparison found zero changed elements in all 350,165,184 VLM parameters,
while the Ego Action Expert/head and the independent World Expert/scene path
all changed. The first-update physical errors were 8.99 mm for Ego and 423.8 mm
for the newly initialized World head, which is intentionally isolated until it
passes the physical gate.

## Independent calibration result and physical-metric correction

The complete 1300-update independent calibration run is stored at
`world_eef_task6_task8_100ep_independent_parallel_strict_equal_4gpu_b24_schedule1564_stop1300`.
Its final 100-update audit failed the interaction gate:

| stream | translation | rotation |
|---|---:|---:|
| Ego | 7.20 mm | 0.585 degrees |
| World | 56.69 mm | 7.115 degrees |
| World / Ego | 7.87x | 12.16x |

The run stopped without environment evaluation. More training with the same
objective is not justified: World translation remained near 5.5--5.7 cm for
hundreds of updates while the schedule decayed to its terminal learning rate.

The scalar logs expose the underlying dimensional error. Near step 520,
`loss_worldflow_geo=0.1263` was almost exactly the World rotation error of
7.19 degrees expressed in radians (0.1255). A 4.5 cm translation error entered
the old Smooth-L1 endpoint term at only about `1e-3`. Thus "translation
weight 1, rotation weight 1" was not physical equality: it added squared
metres to radians and allowed rotation to account for virtually the entire
objective.

Corrected robot-base World-EEF calibration now uses a rigid-probe metric. Six
symmetric points at the fixed 10 cm EEF/tool radius convert both parts of the
base-decoupled field to physical point velocity:

```text
flow error = norm(delta p_dot_base)
           + mean_probe norm(delta omega_base x (R_state r_probe))
```

The endpoint term likewise adds the EEF-origin translation error to the mean
orientation-induced displacement of the same EEF-attached probes. Keeping the
two symmetric components additive prevents accidental cancellation between a
translation and a rotation at an individual probe. Consequently:

- translation and rotation supervision share one physical unit (metres);
- rotation never moves the EEF origin around the robot-base origin;
- the metric contains no learned gate, residual rate, loss-balance sweep, or
  coordinate-dependent conjugation;
- separate translation and rotation-probe metre errors are logged so another
  dimensional imbalance cannot hide behind a scalar total.

The interaction gate remains unchanged and still uses the unweighted EEF
translation/rotation errors. Passing the new training objective alone is not
evidence of success.

The complete rigid-probe run improved World translation but still failed its
final 100-update gate:

| stream | translation | rotation |
|---|---:|---:|
| Ego | 7.19 mm | 0.585 degrees |
| World | 32.43 mm | 20.51 degrees |
| World / Ego | 4.51x | 35.07x |

It therefore also stopped without environment evaluation. The four physical
loss components stayed well-scaled, so the remaining gap is not another
metre/radian weighting failure. It comes from a deeper representation
asymmetry: the loaded Ego Expert predicts a pretrained 9D position +
rotation-6D Euclidean flow, whereas World was given a random 6D
`[p_dot_base, omega_base]` head and an SO(3) geodesic path. Independent
parameters had accidentally become an independently redefined and much harder
prediction problem.

## Symmetric base-pose9 calibration

`base_pose9_euclidean` removes that asymmetry while keeping the complete World
trajectory in robot-base coordinates. Let `C = T_base_current` be fixed for
one observation. Left-transforming both Euclidean pose9 endpoints applies the
same base rotation to position and to each rotation-6D column, plus the same
base translation to both endpoint positions. Therefore interpolation and
velocity commute exactly with the coordinate change:

```text
x_world(t) = left_pose9(C, x_ego(t))
u_world    = rotate_pose9_velocity(C.R, u_ego)
```

This is not an Ego residual and no runtime tensor or parameter is shared. The
World branch still owns its base-frame scene encoder, current-base-pose token,
Action Expert, action projections and output head. The structural advantages
are:

- both branches now solve the same 9D flow-matching problem;
- the independent World output head is copied by value from the trained Ego
  head instead of being random;
- position is still added directly on base axes, so rotating orientation never
  rotates EEF position around the base origin;
- endpoint projection to SO(3) and the rigid-probe metre loss remain active;
- there is no `1/(1-t)`, residual, learned rate, gate, or conjugation.

Tests verify the complete raw pose9 path and target velocity are exactly left
equivariant, and an exact predicted velocity gives zero World flow and endpoint
loss. Interaction remains disabled until the unchanged physical hard gate
passes.

### Complete symmetric calibration result

The four-GPU `base_pose9_euclidean` run completed all 1300 updates and was
stopped by the physical alignment gate before any environment evaluation.  The
last-100-update audit was:

| stream | translation | rotation |
|---|---:|---:|
| Ego | 7.06 mm | 0.583 degrees |
| World | 23.80 mm | 5.417 degrees |
| World / Ego | 3.37x | 9.29x |

The representation correction is real: it improved the previous rigid-probe
World result from 32.43 mm / 20.51 degrees to 23.80 mm / 5.42 degrees.  It did
not, however, meet the required 12 mm / 2 degree and 1.5x alignment gates.  The
curve was already effectively flat by steps 780--1300, so extending the same
optimization or changing a local loss/learning-rate multiplier is not a valid
next experiment.

One apparent carrier discrepancy was explicitly ruled out.  The serialized
parquet action is expressed in the episode-first-EEF frame, but the policy
preprocessor's `UMIProcessor` rebases both the observation state and action
chunk into the *current* EEF frame before the model forward.  Consequently the
model's current-EEF-to-base left transform is the correct action carrier.  The
dataset-side relative-pose audit confirms this to approximately `1e-7`; using
the episode origin again inside the model would apply the transform twice.

The remaining structural question is how the two complete policies should be
bridged.  It must not be answered by redefining the World policy as an internal
Ego policy: World is itself a conventional point-cloud VLA whose observation
and action are both expressed on robot-base axes.

## Raw-chart correction and rejected semantic collapse

The preceding symmetric run still contained a decisive implementation error.
The formal Ego policy uses an unconstrained Gaussian in all pose9 channels,
then follows a Euclidean chord.  The original `left_compose_ego_pose_to_world`
first called `pose9_to_matrix`, which Gram--Schmidt projected those arbitrary
rotation-6D noise columns onto SO(3).  Consequently its claimed exact
equivariance held only for already-valid SE(3) samples, not for the actual
training distribution.

The corrected raw-chart frame map is affine for every Gaussian sample:

```text
p_base  = R_current p_ego + t_current
r1_base = R_current r1_ego
r2_base = R_current r2_ego
```

It therefore commutes exactly with both Euclidean interpolation and the
pose9 velocity target without projecting the flow state.  A regression now
uses deliberately unconstrained Gaussian rotation columns and proves both
the affine path identity and that the old projected result differs.

A diagnostic implementation then converted the World action state back into
Ego coordinates inside the Expert, reused Ego foreground/action tokens and the
Ego gripper state, and rotated the copied output back to base axes.  Its smokes
showed why the loss gap disappeared:

| implementation | Ego translation | World translation | World rotation |
|---|---:|---:|---:|
| projected raw noise, unconstrained base Expert | 8.99 mm | 423.8 mm | -- |
| raw affine noise transport only | 8.85 mm | 212.2 mm | 105.8 degrees |
| internal World-to-Ego action wrapper | 8.91 mm | 77.9 mm | 5.23 degrees |
| duplicated Ego function plus base-context adapter | 7.98 mm | 7.95 mm | 0.743 degrees |

The final numerical equality is not a valid World result.  It was obtained by
collapsing the two policies to the same Ego semantics before their proposals
were formed.  The associated 1300-step run was stopped immediately and this
wrapper was removed.

The retained coordinate contract is instead:

- Ego consumes current-EEF-frame geometry and predicts an Ego-frame action;
- World consumes robot-base-frame geometry and predicts a robot-base-frame
  action as a conventional point-cloud VLA;
- each stream owns a same-structure `LitePTEncoder` and `PointActionAdapter`;
- World has no private language-mean or global-scene residual shortcut;
- neither flow state nor either supervision target is converted into the other
  branch's coordinate semantics.

For clarity, absolute poses and motion operators use different maps.  If
`C=T_base_current`, an Ego absolute-relative proposal `B` and a World absolute
EEF proposal `W` satisfy `W=C B` (left composition).  The corresponding
world-frame motion operator is

```text
G = W C^-1 = C B C^-1
```

so *this derived motion representation* is where conjugation belongs. Applying
the conjugation as the World branch's absolute action target would change the
branch semantics and remains prohibited.

## Symmetric PointActionExpert conjugate bridge

The formal architecture uses two coordinate-specific but structurally matched
point front-ends and exactly one joint Action Expert call:

```text
Ego XYZRGB -> PointSeg -> Ego foreground/background
             -> Ego LitePT + Ego PointActionAdapter -> Ego action tokens ----\
                                                                              +-> one joint
World XYZRGB (robot-base) -> World foreground/background                      |   PointActionExpert
             -> World LitePT + World PointActionAdapter -> World action tokens-/       |
                                                                                 Ego/World heads
```

The original selected foreground and background clouds are both retained.
Foreground LitePT tokens condition each branch's PointActionAdapter.  The
foreground and background global features become two scene tokens per branch,
so the joint Expert receives symmetric `{foreground, background}` sensory
roles instead of silently dropping World background.

The unchanged flow states also produce two coordinate-alignment encodings:

```text
Ego common motion   = pose9(C B_t C^-1)
World common motion = pose9(W_t C^-1)
```

They are computed after each branch has retained its own input representation;
they do not replace `B_t` or `W_t`.  One shared zero-initialized full matrix
projects either pose9 representation into the Expert width and adds it to the
corresponding point-conditioned action token as a coordinate tag.  It is not
a scalar gate, residual rate, target conversion, or learned replacement for
either flow.

The exact Expert input sequence is:

```text
unchanged pretrained image/language prefix
-> Ego foreground -> Ego background -> Ego action[0:T]
-> World foreground -> World background -> World action[0:T]
```

The pretrained VLM keeps its established internal image/language ordering;
the formal mode does not append a duplicate copy of the Ego scene globals to
that prefix.  Every scene token receives both a flow identity embedding
(`Ego`/`World`) and a shared role embedding (`foreground`/`background`).
Paired explicit positions are `foreground=0`, `background=1`, and
`action[t]=2+t` for both flows.  Therefore corresponding roles and time steps
have the same geometric position while their flow embeddings remain distinct.

The sequence is processed in **one** `PointActionExpert` forward with two
attention domains. The non-action domain contains image/language plus both
flows' foreground/background tokens and is globally bidirectional. The action
domain contains all 64 Ego/World action tokens and is also globally
bidirectional: it is not causal over trajectory time. Cross-domain visibility
is block-causal: every action query can read every valid non-action key, while
non-action queries cannot read action keys. This prevents noisy flow states
from rewriting perceptual conditioning while still allowing every action token
to jointly use both coordinate streams and all observations. Separate Ego and
World output heads consume only their own action slices and preserve the two
complete targets and losses; scene outputs are conditioning states only. There
is no independent `world_lm_expert`, no historical pre-Expert linear residual
bridge, and no post-Expert latent cross-attention.

The formal mode name is
`point_action_expert_conjugate_bridge`.  Its enforced contract is:

- `worldflow_action_expert_mode=shared`;
- `worldflow_noise_coupling=left_compose_ego`;
- `worldflow_world_eef_velocity_mode=base_pose9_euclidean`;
- robot-base World scene/action semantics and current-EEF Ego semantics;
- only the VLM is frozen; both LitePT paths, both PointActionAdapters, the
  shared Action Expert, both heads, scene projections and the conjugate
  coordinate projection are trainable.

The previous proposal-after-Expert interaction remains loadable only for old
checkpoint compatibility and is no longer the formal successor design.

The earlier implementation that invoked the same Expert parameter instance
twice was rejected after code audit: parameter sharing is not token fusion and
does not permit per-sample layer-wise interaction. Its partial training output
was stopped and retained.

Two real four-GPU 10-step smokes then separated an over-complete joint suffix
from the compact formal layout:

| layout at step 10 | Ego translation | Ego rotation | World translation | World rotation |
|---|---:|---:|---:|---:|
| separate Ego/World motion sequences, 132 suffix tokens (rejected) | 45.93 mm | 3.440 degrees | 177.8 mm | 107.5 degrees |
| four scene + paired Ego/World action tokens, 68 suffix tokens | 37.69 mm | 3.251 degrees | 194.3 mm | 93.3 degrees |

These short runs are execution diagnostics, not convergence or policy evidence.
The compact run completed multiple DDP updates with no unused parameters. Its
saved checkpoint has 145 shared-Expert tensors, zero independent-World Expert
or historical bridge tensors, and all `6480/6480` entries of the initially-zero
conjugate coordinate matrix changed after ten updates. Artifacts are under
`SMOKE_world_eef_joint_pointactionexpert_fg_bg_pairedpos_compact_4gpu_b24_10steps_20260816`.

After enforcing the exact Ego-scene/Ego-action/World-scene/World-action order,
removing the duplicate Ego globals from the VLM prefix, and adding the shared
foreground/background role embedding, a fresh four-GPU 10-step smoke also
completed without unused parameters or OOM. The saved role embedding changed
in all `1440/1440` entries, the conjugate projection changed in all
`6480/6480` weight entries, and the checkpoint still contains 145 shared-Expert
tensors with no independent World Expert or historical linear bridge. A real
cached-inference environment smoke loaded this checkpoint and completed 11
model calls / 10 environment steps. These artifacts are under
`SMOKE_world_eef_joint_pointactionexpert_exact_sequence_4gpu_b24_10steps_20260816`
and
`SMOKE_world_eef_joint_pointactionexpert_exact_sequence_cached_inference_process_20steps_20260816`.

That exact-sequence run still used one fully bidirectional suffix and is
superseded by the final two-domain block-causal visibility contract above. A
third fresh four-GPU 10-step smoke verified the final mask in DDP, and its
checkpoint completed a real cached-inference environment smoke with 11 model
calls / 10 environment steps. Final-mask artifacts are under
`SMOKE_world_eef_joint_pointactionexpert_blockcausal_4gpu_b24_10steps_20260816`
and
`SMOKE_world_eef_joint_pointactionexpert_blockcausal_cached_inference_20steps_20260816`.
