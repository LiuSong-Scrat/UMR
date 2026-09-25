# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 32 #50
    n_action_steps: int = 16 #50
    # Some recorder formats store observation[i] after action[i] has already
    # been applied.  Their first causal control target is therefore
    # action[i + 1].  Shift action chunk lookup at the dataset boundary rather
    # than training a stale token and compensating with action_index at runtime.
    action_chunk_start_offset: int = 0

    # Shorter state and action vectors will be padded
    max_state_dim: int = 10
    max_action_dim: int = 10

    # Normalization mapping for Flow Matching (required for action normalization)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Point-cloud and RGB views are independently selectable. None preserves
    # legacy checkpoints by coupling RGB selection to point-cloud selection.
    camera_views: str = "agentview"
    rgb_camera_views: str | None = None
    # Optional scene-point budget ratios in camera_views order. None preserves
    # the legacy equal split, so existing multi-view checkpoints keep the exact
    # composition they were trained with. Example: "9,1" retains 90% of the
    # agentview scene budget while adding 10% eye-in-hand coverage.
    camera_view_weights: str | None = None
    # Multi-view composition policy. ``legacy_budget`` preserves every old
    # checkpoint exactly. ``fps`` forms an equal union of every view's scene
    # points, keeps one gripper tail, and downsamples on-device with FPS.
    # ``voxel_fps`` first removes repeated occupied voxels from the shared
    # spatial union, then applies the same FPS contract.
    # ``voxel_cover_fps`` retains one representative of every occupied voxel
    # when possible, then fills remaining detail slots in union-FPS order.
    # ``multiscale_novelty_union`` protects every primary fine voxel and adds
    # one secondary representative only for coverage that remains novel at a
    # configurable coarser scale; the geometry determines the camera share.
    # ``consensus_multiscale_novelty_union`` additionally lets overlapping
    # views choose a real union-medoid representative inside each protected
    # primary fine voxel, while retaining coarse-only secondary novelty.
    # ``transport_novelty_union`` preserves every primary occupied voxel and
    # inserts secondary novel voxels through local one-to-one replacements.
    # ``full_union`` preserves every scene point from every selected view and
    # appends the primary gripper tail once. It changes only the input length;
    # no model module or checkpoint parameter is added.
    # ``primary_residual`` keeps the primary cloud byte-identical and encodes
    # the second view through a zero-initialized matrix residual.
    camera_view_fusion: str = "legacy_budget"
    # Deprecated compatibility field. Reduction-based multi-view adapters now
    # derive their output length from the native per-view input length instead
    # of imposing a dataset-specific fixed point count. Old checkpoints may
    # still deserialize their recorded numeric value, but it does not constrain
    # the policy core's accepted input length.
    camera_view_fps_target_points: int | None = None
    camera_view_fps_gripper_points: int = 500
    camera_view_voxel_size: float = 0.005
    camera_view_coarse_novelty_scale: float = 3.0
    multiview_pretrained_lr_multiplier: float = 1.0
    multiview_residual_lr_multiplier: float = 1.0
    # Discriminative fine-tuning for input-layer multi-view fusion. All
    # parameters remain trainable; the point-input path can adapt faster than
    # the pretrained action path to reduce catastrophic forgetting.
    multiview_input_pretrained_lr_multiplier: float = 1.0
    multiview_input_point_lr_multiplier: float = 1.0
    # When jointly adapting an Ego/World double-flow checkpoint to a new
    # point-cloud input distribution, place both coordinate streams' direct
    # point consumers in the point-input optimizer group.  This changes only
    # training-time optimizer membership; the forward graph and old-checkpoint
    # path are unchanged.  It is opt-in so historical runs remain exact.
    multiview_input_symmetric_point_path_adaptation: bool = False
    # Training-only input augmentation: deterministically expose roughly half
    # the frames as the original primary view and half as the all-view fused
    # cloud. Inference still uses every configured view. No model module or
    # learned gate is introduced.
    multiview_input_view_dropout_enable: bool = False
    multiview_input_view_dropout_seed: int = 20260812
    # Expand training to two complementary entries per frame: primary-only
    # and all-view input. This gives both input domains full data coverage
    # without changing model layers or inference behavior.
    multiview_input_view_dropout_paired_coverage: bool = False
    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    # Number of denoising steps during flow matching inference.
    # Recommended values: 8-10 (fast), 20-30 (balanced), 50+ (high-quality)
    num_steps: int = 10
    # The official large-data recipe samples continuous times from
    # Beta(1.5, 1). For small-data fitting, ``integration_grid`` instead
    # trains exactly the Euler times used by sample_actions:
    # {0, 1/num_steps, ..., (num_steps-1)/num_steps}.
    flow_time_sampling: str = "beta"
    # Optional small-data emphasis on the first Euler evaluation. At t=0 the
    # noisy action contains no ground-truth trajectory information, so this is
    # the only grid point that must infer the whole chunk from observations.
    # Zero preserves uniform integration-grid sampling.
    flow_time_zero_probability: float = 0.0

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True
    # Keep proprioception out of the learned prefix by default. The World-Ego
    # branch uses current EEF pose only as an analytic coordinate carrier.
    encode_robot_state: bool = False

    # Checkpoint compatibility fields used by older Song/UMI training runs.
    # The current inference path configures point-cloud processing externally,
    # but retaining these metadata fields allows those checkpoints to load.
    use_umi_processor: bool = False
    official_point_cloud_prefix_enable: bool = False
    point_cloud_feature_dim: int = 64
    point_cloud_grid_size: float = 0.005

    # Frozen-VLM image/point adapter mode. This keeps the official SmolVLM
    # vision/language backbone unchanged and frozen, while the existing point
    # encoders, point/action fusion and Action Expert remain trainable.
    vla_adapter_enable: bool = False
    vla_adapter_freeze_vlm: bool = True

    # Song point-cloud foreground/background conditioning.
    pointseg_enable: bool = False
    pointseg_checkpoint_path: str | None = None
    pointseg_backbone_type: str = "litept"
    pointseg_grid_size: float = 0.01
    pointseg_feature_dim: int = 64
    pointseg_foreground_ratio: float = 0.08
    pointseg_background_ratio: float = 0.25
    pointseg_min_foreground_points: int = 512
    pointseg_min_background_points: int = 512
    pointseg_aux_loss_weight: float = 0.20
    pointseg_use_temporal_priors_as_input: bool = False
    pointseg_use_pseudo_selection: bool = True
    # Preserve pretrained LitePT/PointSeg population statistics during
    # small-data fine-tuning. BatchNorm affine parameters remain trainable.
    pointseg_freeze_batchnorm_stats: bool = False
    point_action_fusion_enable: bool = True
    point_action_fusion_heads: int = 4
    point_action_fusion_dropout: float = 0.0

    # Relative importance of the physical pose9 + gripper action groups in the
    # standard flow-matching objective. Values are normalized to preserve the
    # overall loss scale. Defaults reproduce the original per-dimension MSE.
    action_loss_translation_weight: float = 1.0
    action_loss_rotation_weight: float = 1.0
    action_loss_gripper_weight: float = 1.0
    # Standard SmolVLA assumes normalized Euclidean action channels. Song's
    # identity-normalized pose9 instead contains a 6D rotation representation,
    # so zero-centred Gaussian vectors are not valid rotation states. This
    # option samples a valid SE(3) pose9 in both training and inference while
    # retaining the standard Euclidean flow-matching objective.
    pose9_action_noise_enable: bool = False
    pose9_action_noise_trans_scale: float = 0.15
    pose9_action_noise_rot_scale: float = 0.35
    pose9_action_noise_gripper_scale: float = 0.05

    # Joint World–Ego trajectory branch. PointSeg selects foreground XYZRGB in
    # the Ego/body frame, then the current EEF pose analytically maps those
    # exact points into World. World owns an independent LitePT + PointAction
    # front-end. Historical modes share the official Action Expert; the
    # calibrated design instead gives World an independently parameterized
    # copy. Explicit fusion modes may exchange complete trajectory tokens only
    # after both streams have formed their own proposals. The branch is active
    # in training and inference.
    worldflow_enable: bool = False
    # Supervision semantics for the independent World branch. ``legacy_eef``
    # retains historical checkpoints whose World target is derived through
    # the old carrier/conjugacy path. ``world_eef_trajectory`` predicts the
    # commanded EEF trajectory directly in the fixed robot-base frame. With
    # the foreground cloud in that same frame, the trajectory acts as sparse,
    # task-relevant point-flow supervision without explicit object poses or
    # dense scene-flow labels.
    worldflow_target_type: str = "legacy_eef"
    # Velocity chart for an absolute robot-base EEF trajectory. The legacy
    # spatial twist rotates/translates about the global origin and therefore
    # contains a position-dependent omega-cross-p lever arm. ``base_decoupled``
    # instead predicts [p_dot_base, omega_base]: translation and orientation
    # are integrated independently while both remain expressed on base axes.
    # The legacy default preserves old checkpoints that do not serialize this
    # field. ``base_pose9_euclidean`` keeps the complete pose in robot-base
    # coordinates but uses the same 9D position + rotation-6D Euclidean flow
    # as the pretrained Ego branch. Under the known current-EEF left transform
    # that path is exactly linearly equivariant to Ego, so an independent World
    # Expert does not have to relearn an unrelated 6D SO(3) flow and random
    # output head before the two streams can reach comparable accuracy.
    worldflow_world_eef_velocity_mode: str = "legacy_spatial_twist"
    # Canonical fixed frame used by the World branch. Historical checkpoints
    # used the first fixed camera. New World-EEF checkpoints use the robot
    # base so scene geometry is invariant to camera placement and directly
    # comparable across calibrated fixed cameras. This frame is never the UMI
    # episode origin and never the current EEF frame.
    worldflow_reference_frame: str = "pointcloud_reference_camera"
    worldflow_feature_dim: int = 64
    worldflow_grid_size: float = 0.01
    # Keep the pretrained World LitePT population statistics fixed during
    # fine-tuning while leaving its BatchNorm affine parameters and every
    # World/Ego path trainable. This removes a learning-rate-independent
    # train/inference drift in the residual feature distribution.
    worldflow_freeze_batchnorm_stats: bool = False
    worldflow_loss_weight: float = 0.05
    worldflow_geo_loss_weight: float = 0.05
    worldflow_bridge_loss_weight: float = 0.05
    worldflow_trans_weight: float = 1.0
    worldflow_rot_weight: float = 1.0
    # Physical radius of the symmetric rigid probes used to supervise an
    # absolute robot-base EEF trajectory.  Translation and angular errors have
    # incompatible units (metres versus radians); evaluating the velocity and
    # endpoint displacement of points attached to the EEF converts both to
    # metres without a tunable scalar loss balance.  Ten centimetres is a
    # robot/tool geometry constant, not an optimization multiplier.
    worldflow_eef_probe_radius_m: float = 0.10
    worldflow_se3_head_enable: bool = False
    worldflow_equiv_loss_weight: float = 0.02
    # Discriminative fine-tuning keeps every path plastic while permitting an
    # existing World representation and its zero-initialized physical residual
    # output head to adapt on different time scales. ``None`` preserves the
    # historical two-group behavior by using ``worldflow_new_lr_multiplier``
    # for the residual head as well. Every explicit multiplier must be positive:
    # zero would silently freeze a path.
    worldflow_pretrained_lr_multiplier: float = 1.0
    worldflow_new_lr_multiplier: float = 1.0
    worldflow_residual_lr_multiplier: float | None = None
    # Preserve an already strong Ego policy while learning the independent
    # World branch and the explicit bidirectional interaction modules.  This
    # prevents a small focused dataset (for example 100 demonstrations) from
    # rewriting the pretrained Action Expert, Ego point path, or action head.
    # World remains fully trainable and can still affect Ego through the
    # zero-initialized World-to-Ego cross-attention path.
    worldflow_freeze_pretrained_ego: bool = False
    # Training-only stochastic depth for the physical World-to-Ego correction.
    # A dropped sample is optimized through the unchanged Ego action path;
    # retained samples use the complete bidirectional World/Ego path. This adds
    # no parameter or inference-time branch and uses the same action objective
    # for both cases. A value of 1 is forbidden because it would prevent the
    # World correction from receiving action gradients.
    worldflow_training_world_to_ego_dropout_probability: float = 0.0
    # Optional residual-boosting gradient routing. On samples whose physical
    # World-to-Ego correction is retained, treat the current Ego prediction as
    # a fixed boosting anchor and update the World/residual path through the
    # unchanged action loss. Dropped samples continue to update the ordinary
    # Ego action path. This is training-only, adds no parameter or loss, and
    # leaves the complete inference forward exactly unchanged.
    worldflow_training_residual_anchor_stop_gradient: bool = False
    # Training-only asymmetric PCGrad for the two stochastic-depth paths.
    # The ordinary Ego-only samples define the protected shared gradient.  If
    # the World-corrected samples produce an opposing gradient, only that
    # opposing component is projected out; aligned World gradients and all
    # World-only parameter gradients are retained.  Both paths still optimize
    # the same action loss from one forward, every parameter remains trainable,
    # and inference is unchanged.
    worldflow_training_ego_priority_gradient_projection: bool = False
    # Stronger shared-backbone variant: on pretrained/common parameters the
    # World-corrected gradient is projected onto the non-negative one-
    # dimensional span of the ordinary Ego gradient. Orthogonal or opposing
    # World components cannot rewrite the baseline representation, while the
    # dedicated World/bidirectional/residual parameter groups retain their
    # complete gradients. Ego/common parameters still train from the original
    # Ego action loss, so no branch is frozen.
    worldflow_training_shared_gradient_ego_tangent_projection: bool = False
    # Continuous-time parameterization for endpoint residual boosting. The
    # World head predicts a bounded residual *rate* and the physical SE(3)
    # endpoint correction is multiplied by the remaining flow time (1-t).
    # Consequently the correction vanishes at the terminal boundary and the
    # velocity reconstruction cannot amplify a time-independent residual by
    # 1/(1-t). This is a fixed analytical boundary condition, not a learned
    # gate; it adds no parameter or auxiliary objective. Disabled by default
    # so historical checkpoints preserve their exact inference function.
    worldflow_endpoint_residual_rate_parameterization: bool = False
    # Express the endpoint residual twist in the Ego carrier frame instead of
    # the arbitrary World coordinate frame. A rigid reparameterization of the
    # World frame then leaves the target six-vector unchanged, rather than
    # requiring the point network to learn the SE(3) adjoint transformation of
    # a spatial World twist. The correction is still composed physically on
    # SE(3), uses the same residual head and action loss, and adds no parameter,
    # gate, auxiliary objective, or inference-time branch. Disabled by default
    # for exact historical-checkpoint compatibility.
    worldflow_endpoint_residual_ego_frame_parameterization: bool = False
    # Express the endpoint correction as a body twist at the predicted Ego
    # endpoint: B_corrected = B_ego Exp(xi_body).  The corresponding target
    # Log(B_ego^-1 B_target) is right-invariant and follows the endpoint's own
    # axes, while remaining independent of an arbitrary World-coordinate
    # reparameterization.  This reuses the existing six-dimensional head and
    # action loss; it adds no parameter, gate, auxiliary objective, or second
    # forward.  Disabled by default for exact historical-checkpoint
    # compatibility and mutually exclusive with the carrier-frame left twist
    # above.
    worldflow_endpoint_residual_body_frame_parameterization: bool = False
    # Interpret the same six-dimensional World head as a right-trivialized
    # body-twist *velocity* at the current Ego Flow state rather than as a
    # finite correction at the predicted endpoint.  The induced tangent
    # dB/dt = B hat(xi_body) is lifted back into the pretrained pose9/rotation6d
    # gauge before it is added to the Ego Euclidean vector field.  Its axes are
    # therefore defined by the known current state, not by a still-changing
    # endpoint prediction.  This is coordinate invariant, reuses the existing
    # residual head and action loss, and adds no parameter, gate, auxiliary
    # objective, or second forward.  Disabled by default for exact historical
    # checkpoint compatibility and mutually exclusive with every finite
    # endpoint-residual parameterization above.
    worldflow_body_velocity_residual_parameterization: bool = False
    # Training-only reparameterization of the complete World coordinate frame.
    # One random rigid transform A is applied consistently to World points,
    # the Ego-to-World carrier C, and every World trajectory state, so the
    # physical Ego action represented by C^-1 G C is unchanged. This replaces
    # the canonical World representation for the ordinary single action-loss
    # forward; it adds no second view, auxiliary objective, parameter, gate, or
    # inference-time operation.
    worldflow_training_coordinate_frame_augmentation: bool = False
    # Optional one-shot initialization used when adapting an Ego checkpoint:
    # copy shape-compatible Ego action/point modules into the World stream,
    # while keeping both streams trainable. This is not applied when loading a
    # trained WorldFlow checkpoint unless explicitly requested by the caller.
    worldflow_bootstrap_from_ego: bool = False
    # Deprecated checkpoint-compatibility field. World/Ego scalar gates are
    # forbidden; only None is accepted. Keeping the parser field lets older
    # gate-free checkpoints that serialized null continue to load.
    worldflow_ego_residual_gate_init: float | None = None
    # 0 keeps the complete predicted foreground. A positive value is an
    # optional memory cap applied after PointSeg foreground selection.
    worldflow_max_points: int = 0
    # Command-target sidecars keep the World target exactly aligned with the
    # Ego action chunk. False preserves legacy datasets that only contain
    # achieved poses; production WorldFlow recipes should enable this guard.
    worldflow_require_action_target_sidecar: bool = False
    # Legacy v0.5 CLI fields. v0.5.1+ shares the Ego Action Expert, so these
    # values are parsed for old commands but do not instantiate another expert.
    worldflow_action_expert_layers: int = -1
    worldflow_action_expert_dropout: float = 0.0
    # WorldFlow denoises an SE(3) spatial transform. Unit Gaussian twists
    # correspond to metre-scale translations and roughly 90 degree rotations,
    # which is far outside a tabletop robot's action distribution. Keep the
    # noise on the same physical scale as the reachable trajectory.
    worldflow_noise_trans_scale: float = 0.15
    worldflow_noise_rot_scale: float = 0.20
    # Noise relationship between the two action streams.
    #
    # ``independent`` reproduces the original experimental implementation:
    # Ego body motion and World spatial motion start from unrelated random
    # transforms. ``left_compose_ego`` is the physically correct prior for an
    # absolute robot-base EEF trajectory: if C is the observed current
    # EEF-to-base pose and B_0 is the Ego prior expressed in the current EEF
    # frame, the World prior is W_0 = C B_0. Only the stochastic origin is
    # coupled; the two vector fields are integrated independently afterwards.
    # ``conjugate_ego`` samples one physically valid Ego SE(3)
    # prior B_0 and derives the World prior exactly as G_0 = C B_0 C^{-1},
    # where C is the current EEF-to-World pose.  The latter is the recommended
    # stochastic double-flow contract. ``projected_ego_chart`` instead keeps a
    # legacy checkpoint's exact Euclidean Ego noise distribution, projects its
    # rotation-6D chart to SO(3), and conjugates only the World prior. This is
    # the checkpoint-compatible coupling contract: it leaves the Ego input
    # byte-for-byte unchanged while removing an unrelated World random origin.
    # ``projected_ego_path`` strengthens that contract at every denoising time:
    # World state is always the analytic conjugate of the current projected Ego
    # chart, rather than following a second, incompatible Euclidean chord.
    worldflow_noise_coupling: str = "independent"
    # ``global`` is the historical spatial-transform frame G=CBC^-1. Its
    # translation contains a world-origin lever-arm term. ``current_ee`` keeps
    # world-aligned axes but places the origin at the current EEF, yielding an
    # exactly invertible, better-conditioned conjugacy without that term.
    worldflow_frame_origin: str = "global"
    # Coordinate frame used only to encode the World scene points. ``carrier``
    # preserves the historical contract and reuses ``worldflow_frame_origin``.
    # ``global`` keeps the scene in the fixed dataset/world frame even when the
    # action-flow carrier uses ``current_ee`` to remove its conjugation
    # lever-arm. This separates sensory information from action
    # parameterization: it restores absolute scene position without changing
    # the well-conditioned local World action state or adding model parameters.
    worldflow_scene_frame_origin: str = "carrier"
    # ``cross_attention`` preserves historical checkpoints: World affects the
    # Ego head only through token exchange. ``symmetric_twist`` additionally
    # maps the World spatial twist back to Ego coordinates with the exact SE(3)
    # adjoint and averages the two physical predictions with fixed 1:1 weight.
    # ``conjugate_residual`` derives an exact World baseline from Ego and lets
    # a zero-initialized World head predict a residual correction. The
    # checkpoint-compatible ``conjugate_residual_consensus`` splits that
    # correction equally between the two coordinate descriptions, then derives
    # World exactly from the corrected Ego twist. ``conjugate_residual_boosting``
    # has the same forward contract but stops the direct Ego-score gradient in
    # the World auxiliary objective. This makes the World residual learn the
    # remaining Ego error instead of allowing the two predictors to cancel.
    # ``endpoint_geodesic_consensus`` retains the checkpoint's legacy pose9
    # Ego flow.  Ego and World independently predict the same physical
    # endpoint, the World endpoint is conjugated back to Ego coordinates, and
    # their fixed 1:1 SE(3) geodesic midpoint is converted back to an Ego
    # pose9 velocity.  This gives World a coordinate-defined path to Ego
    # without a latent residual or learned gate.
    # ``endpoint_residual_boosting`` instead keeps the pretrained Ego endpoint
    # as the exact zero-residual function and uses the existing zero-initialized
    # World twist head to predict only the remaining SE(3) endpoint error.  The
    # corrected World endpoint is conjugated back to Ego and represented in the
    # original pose9 chart.  It has no fixed mixing coefficient: World learns a
    # physical correction, not a second absolute policy to average with Ego.
    # These are deterministic geometry/fixed fusion contracts, not learned gates.
    # ``point_action_expert_conjugate_bridge`` keeps two coordinate-semantic
    # front-ends (each LitePT + PointAction adapter), converts their current
    # flow states to the same robot-base motion operator by exact conjugation,
    # exchanges them through bidirectional full-matrix token adapters, and runs
    # both action streams through one shared Action Expert parameter instance.
    # Neither branch's input state or supervision target is rewritten.
    # ``independent_parallel`` is the calibration phase for a true dual-flow
    # model: both complete trajectories are directly supervised and every
    # non-VLM module remains trainable, but neither flow can alter the other.
    # Physical interaction is enabled only after their unweighted endpoint
    # errors are comparable. This is a curriculum, not a learned gate.
    worldflow_action_fusion: str = "cross_attention"
    # ``shared`` is the historical implementation: World owns its geometric
    # front-end but its action tokens are processed by the Ego Action Expert.
    # ``independent`` gives World a separately parameterized Action Expert,
    # bootstrapped once from Ego and then trained only by the World/interaction
    # objectives.  Ego and World exchange information only through the explicit
    # bidirectional cross-attention path after their respective experts.
    worldflow_action_expert_mode: str = "shared"
    # A robot-base trajectory is not Markov without the achieved current EEF
    # pose: after the foreground cloud is mapped out of the current-EEF frame,
    # that pose can no longer be recovered from foreground geometry alone.
    # New World-trajectory models expose it as an explicit World Expert token.
    # Disabled by default so historical WorldFlow checkpoints remain loadable.
    worldflow_current_ee_pose_token: bool = False
    worldflow_augmentation_trans_scale: float = 0.20
    worldflow_augmentation_rot_scale: float = 0.75
    # Legacy Dense-ObjectFlow options retained only so old command lines and
    # configs remain parseable. The independent branch does not consume role
    # scores, predict point flow, or run Kabsch.
    worldflow_min_transport_points: int = 3
    worldflow_transport_score_threshold: float = 0.05

    # ET-SEED-style SE(3) action generation.
    se3_enable: bool = False
    # ``direct_twist`` uses dedicated randomly initialized 7D Ego / 6D World
    # heads. ``projected_pose9`` reuses the pretrained pose9 velocity heads and
    # analytically projects their instantaneous rigid-body derivative onto a
    # spatial SE(3) twist. ``pose9_endpoint`` instead preserves the old flow
    # head's endpoint semantics: x_t + (1-t) v_pose9 is projected to SE(3), then
    # converted to the exact left-trivialized geodesic velocity from x_t.
    # ``pose9_chart_endpoint`` additionally keeps the original v0.4.2
    # Euclidean pose9 chart as the Action Expert input while carrying a
    # separate physical state on SE(3). This preserves checkpoint input and
    # endpoint semantics without giving up group integration or World/Ego
    # conjugacy. All modes use a manifold-valued physical state.
    se3_twist_head_mode: str = "direct_twist"
    se3_noise_trans_scale: float = 0.15
    se3_noise_rot_scale: float = 0.75
    se3_noise_gripper_scale: float = 0.05
    se3_pose_loss_weight: float = 1.0
    se3_gripper_loss_weight: float = 1.0
    se3_endpoint_loss_weight: float = 0.25
    se3_final_correction_enable: bool = False
    se3_final_correction_loss_weight: float = 0.20
    se3_equivariance_loss_weight: float = 0.02

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 100 #1_000
    scheduler_decay_steps: int = 60_000 #30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    # Optional raw SmolVLM directory/repository or SmolVLA policy checkpoint
    # used only as the source of the VLM weights. `vlm_model_name` remains the
    # source of the architecture and processor when a policy checkpoint is used.
    vlm_weights_path: str | None = None
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights
    # One-shot initialization used only when make_policy creates a fresh
    # `--policy.type=smolvla` model. Full checkpoints loaded through
    # `--policy.path` already contain these tensors and are never overwritten.
    load_action_expert_weights: bool = False
    # Defaults to vlm_weights_path when omitted. The source must be a complete
    # SmolVLA policy checkpoint, not a raw SmolVLM repository.
    action_expert_weights_path: str | None = None
    # The official 32 action channels need not share semantics with Song's
    # physical pose9 + gripper channels. Keep the task-specific projections
    # random by default while reusing the transformer and timestep MLP.
    load_action_expert_projection_weights: bool = False

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        # Preserve compatibility with checkpoints produced before the
        # multi-view fusion names were standardized.
        if self.camera_view_fusion == "global_local_geometry":
            self.camera_view_fusion = "legacy_budget"
        if self.camera_view_fusion not in {
            "legacy_budget",
            "fps",
            "voxel_fps",
            "voxel_cover_fps",
            "novelty_union",
            "multiscale_novelty_union",
            "consensus_multiscale_novelty_union",
            "transport_novelty_union",
            "uniform_union",
            "full_union",
            "primary_residual",
        }:
            raise ValueError(
                "camera_view_fusion must be 'legacy_budget', 'fps', 'voxel_fps', "
                "'voxel_cover_fps', 'novelty_union', 'multiscale_novelty_union', "
                "'consensus_multiscale_novelty_union', "
                "'transport_novelty_union', "
                "'uniform_union', 'full_union', "
                "or 'primary_residual'; "
                f"got {self.camera_view_fusion!r}."
            )
        if self.camera_view_fps_target_points is not None:
            if int(self.camera_view_fps_target_points) <= 0:
                raise ValueError("camera_view_fps_target_points must be positive when set.")
            if not 0 <= int(self.camera_view_fps_gripper_points) < int(
                self.camera_view_fps_target_points
            ):
                raise ValueError(
                    "camera_view_fps_gripper_points must be in "
                    "[0, camera_view_fps_target_points)."
                )
        if not math.isfinite(float(self.camera_view_voxel_size)) or float(self.camera_view_voxel_size) <= 0.0:
            raise ValueError("camera_view_voxel_size must be finite and positive.")
        if (
            not math.isfinite(float(self.camera_view_coarse_novelty_scale))
            or float(self.camera_view_coarse_novelty_scale) <= 1.0
        ):
            raise ValueError("camera_view_coarse_novelty_scale must be finite and greater than one.")
        if self.camera_view_fusion in {
            "fps",
            "voxel_fps",
            "voxel_cover_fps",
            "novelty_union",
            "multiscale_novelty_union",
            "consensus_multiscale_novelty_union",
            "transport_novelty_union",
            "uniform_union",
            "full_union",
            "primary_residual",
        } and self.camera_view_weights not in {None, ""}:
            raise ValueError(
                f"camera_view_weights cannot be combined with camera_view_fusion={self.camera_view_fusion!r}; "
                "equal-union fusion does not use a hand-chosen view ratio."
            )
        if float(self.multiview_input_pretrained_lr_multiplier) <= 0:
            raise ValueError("multiview_input_pretrained_lr_multiplier must be positive; zero would freeze parameters.")
        if float(self.multiview_input_point_lr_multiplier) <= 0:
            raise ValueError("multiview_input_point_lr_multiplier must be positive; zero would freeze parameters.")
        if self.multiview_input_symmetric_point_path_adaptation and not self.worldflow_enable:
            raise ValueError(
                "multiview_input_symmetric_point_path_adaptation requires worldflow_enable=True."
            )
        if self.multiview_input_symmetric_point_path_adaptation and self.camera_view_fusion not in {
            "fps",
            "voxel_fps",
            "voxel_cover_fps",
            "novelty_union",
            "multiscale_novelty_union",
            "consensus_multiscale_novelty_union",
            "transport_novelty_union",
            "uniform_union",
            "full_union",
        }:
            raise ValueError(
                "multiview_input_symmetric_point_path_adaptation requires a supported "
                "input-layer multi-view fusion."
            )
        if self.multiview_input_view_dropout_enable:
            if self.camera_view_fusion not in {
                "fps",
                "novelty_union",
                "multiscale_novelty_union",
                "consensus_multiscale_novelty_union",
                "transport_novelty_union",
                "uniform_union",
                "full_union",
            }:
                raise ValueError(
                    "multiview_input_view_dropout_enable currently requires "
                    "camera_view_fusion='fps', 'novelty_union', 'multiscale_novelty_union', "
                    "'consensus_multiscale_novelty_union', "
                    "'transport_novelty_union', 'uniform_union', or 'full_union'."
                )
            views = tuple(part.strip() for part in str(self.camera_views).split(",") if part.strip())
            if len(views) < 2:
                raise ValueError("multiview_input_view_dropout_enable requires at least two camera_views.")
        elif self.multiview_input_view_dropout_paired_coverage:
            raise ValueError(
                "multiview_input_view_dropout_paired_coverage requires "
                "multiview_input_view_dropout_enable=true."
            )
        if self.camera_view_fusion == "primary_residual":
            views = tuple(part.strip() for part in str(self.camera_views).split(",") if part.strip())
            if len(views) != 2:
                raise ValueError(
                    "camera_view_fusion='primary_residual' requires exactly two camera_views; "
                    f"got {views}."
                )
            if float(self.multiview_pretrained_lr_multiplier) <= 0:
                raise ValueError("multiview_pretrained_lr_multiplier must be positive.")
            if float(self.multiview_residual_lr_multiplier) <= 0:
                raise ValueError("multiview_residual_lr_multiplier must be positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if int(self.action_chunk_start_offset) < 0:
            raise ValueError("action_chunk_start_offset must be non-negative.")
        if int(self.action_chunk_start_offset) > 0:
            warnings.warn(
                "ACTION TEMPORAL CONTRACT: action_chunk_start_offset="
                f"{int(self.action_chunk_start_offset)} means predicted token 0 is trained from "
                f"dataset action[i+{int(self.action_chunk_start_offset)}]. Online inference must "
                "execute predicted token 0 (action_index=0); do not apply the offset a second time.",
                stacklevel=2,
            )
        if self.flow_time_sampling not in {"beta", "uniform", "integration_grid"}:
            raise ValueError(
                "flow_time_sampling must be one of beta, uniform, integration_grid; "
                f"got {self.flow_time_sampling!r}."
            )
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if not 0.0 <= float(self.flow_time_zero_probability) < 1.0:
            raise ValueError("flow_time_zero_probability must be in [0, 1).")
        if self.flow_time_zero_probability > 0 and self.flow_time_sampling != "integration_grid":
            raise ValueError(
                "flow_time_zero_probability is only supported with "
                "flow_time_sampling='integration_grid'."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.se3_enable:
            action_norm = self.normalization_mapping.get("ACTION")
            if action_norm is not NormalizationMode.IDENTITY:
                raise ValueError("se3_enable=True requires ACTION normalization to be IDENTITY.")
            if self.max_action_dim < 10:
                raise ValueError("se3_enable=True requires max_action_dim >= 10.")
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError("se3_enable=True is not supported with RTC enabled in v1.")
            if self.se3_twist_head_mode not in {
                "direct_twist",
                "projected_pose9",
                "pose9_endpoint",
                "pose9_chart_endpoint",
            }:
                raise ValueError(
                    "se3_twist_head_mode must be 'direct_twist', 'projected_pose9', "
                    "'pose9_endpoint', or 'pose9_chart_endpoint'; "
                    f"got {self.se3_twist_head_mode!r}."
                )
            for name in (
                "se3_noise_trans_scale",
                "se3_noise_rot_scale",
                "se3_noise_gripper_scale",
            ):
                if float(getattr(self, name)) < 0:
                    raise ValueError(f"{name} must be non-negative.")
        if self.se3_enable and self.pose9_action_noise_enable:
            raise ValueError(
                "se3_enable and pose9_action_noise_enable are mutually exclusive: "
                "se3_enable already supplies the complete manifold-valued SE(3) prior and flow."
            )
        for name in (
            "action_loss_translation_weight",
            "action_loss_rotation_weight",
            "action_loss_gripper_weight",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.pose9_action_noise_enable:
            action_norm = self.normalization_mapping.get("ACTION")
            if action_norm is not NormalizationMode.IDENTITY:
                raise ValueError(
                    "pose9_action_noise_enable=True requires ACTION normalization to be IDENTITY."
                )
            if self.max_action_dim < 10:
                raise ValueError("pose9_action_noise_enable=True requires max_action_dim >= 10.")
            for name in (
                "pose9_action_noise_trans_scale",
                "pose9_action_noise_rot_scale",
                "pose9_action_noise_gripper_scale",
            ):
                if float(getattr(self, name)) < 0:
                    raise ValueError(f"{name} must be non-negative.")
        if self.worldflow_se3_head_enable:
            if not self.worldflow_enable:
                raise ValueError("worldflow_se3_head_enable=True requires worldflow_enable=True.")
            if not self.se3_enable:
                raise ValueError("worldflow_se3_head_enable=True requires se3_enable=True.")
        if self.worldflow_training_coordinate_frame_augmentation and not self.worldflow_enable:
            raise ValueError(
                "worldflow_training_coordinate_frame_augmentation=True requires worldflow_enable=True."
            )
        if self.worldflow_training_ego_priority_gradient_projection and not self.worldflow_enable:
            raise ValueError(
                "worldflow_training_ego_priority_gradient_projection=True requires worldflow_enable=True."
            )
        if self.worldflow_training_shared_gradient_ego_tangent_projection and not self.worldflow_enable:
            raise ValueError(
                "worldflow_training_shared_gradient_ego_tangent_projection=True requires "
                "worldflow_enable=True."
            )
        if self.worldflow_endpoint_residual_rate_parameterization and not self.worldflow_enable:
            raise ValueError(
                "worldflow_endpoint_residual_rate_parameterization=True requires "
                "worldflow_enable=True."
            )
        if self.worldflow_endpoint_residual_ego_frame_parameterization and not self.worldflow_enable:
            raise ValueError(
                "worldflow_endpoint_residual_ego_frame_parameterization=True requires "
                "worldflow_enable=True."
            )
        if self.worldflow_endpoint_residual_body_frame_parameterization and not self.worldflow_enable:
            raise ValueError(
                "worldflow_endpoint_residual_body_frame_parameterization=True requires "
                "worldflow_enable=True."
            )
        if self.worldflow_body_velocity_residual_parameterization and not self.worldflow_enable:
            raise ValueError(
                "worldflow_body_velocity_residual_parameterization=True requires "
                "worldflow_enable=True."
            )
        if self.worldflow_freeze_pretrained_ego and not self.worldflow_enable:
            raise ValueError(
                "worldflow_freeze_pretrained_ego=True requires worldflow_enable=True."
            )
        if self.worldflow_enable:
            if self.worldflow_action_expert_mode not in {"shared", "independent"}:
                raise ValueError(
                    "worldflow_action_expert_mode must be 'shared' or 'independent', "
                    f"got {self.worldflow_action_expert_mode!r}."
                )
            if self.worldflow_target_type not in {
                "legacy_eef",
                "world_eef_trajectory",
            }:
                raise ValueError(
                    "worldflow_target_type must be 'legacy_eef' or "
                    f"'world_eef_trajectory', got {self.worldflow_target_type!r}."
                )
            if self.worldflow_world_eef_velocity_mode not in {
                "legacy_spatial_twist",
                "base_decoupled",
                "base_pose9_euclidean",
            }:
                raise ValueError(
                    "worldflow_world_eef_velocity_mode must be "
                    "'legacy_spatial_twist', 'base_decoupled', or "
                    "'base_pose9_euclidean', got "
                    f"{self.worldflow_world_eef_velocity_mode!r}."
                )
            if self.worldflow_reference_frame not in {
                "pointcloud_reference_camera",
                "robot_base",
            }:
                raise ValueError(
                    "worldflow_reference_frame must be 'pointcloud_reference_camera' or "
                    f"'robot_base', got {self.worldflow_reference_frame!r}."
                )
            if self.worldflow_target_type == "world_eef_trajectory":
                world_trajectory_contract_errors = []
                if self.worldflow_reference_frame != "robot_base":
                    world_trajectory_contract_errors.append("worldflow_reference_frame='robot_base'")
                if self.worldflow_scene_frame_origin != "global":
                    world_trajectory_contract_errors.append("worldflow_scene_frame_origin='global'")
                if self.worldflow_frame_origin != "global":
                    world_trajectory_contract_errors.append("worldflow_frame_origin='global'")
                if self.worldflow_noise_coupling not in {"independent", "left_compose_ego"}:
                    world_trajectory_contract_errors.append(
                        "worldflow_noise_coupling='independent' or 'left_compose_ego'"
                    )
                if self.worldflow_action_fusion not in {
                    "independent_parallel",
                    "cross_attention",
                    "point_action_expert_conjugate_bridge",
                    "physical_trajectory_cross_attention",
                    "world_trajectory_arm_ego_gripper",
                }:
                    world_trajectory_contract_errors.append(
                        "worldflow_action_fusion='independent_parallel', 'cross_attention', "
                        "'point_action_expert_conjugate_bridge', "
                        "'physical_trajectory_cross_attention', or "
                        "'world_trajectory_arm_ego_gripper'"
                    )
                if self.worldflow_bridge_loss_weight != 0:
                    world_trajectory_contract_errors.append("worldflow_bridge_loss_weight=0")
                if self.worldflow_equiv_loss_weight != 0:
                    world_trajectory_contract_errors.append("worldflow_equiv_loss_weight=0")
                if self.worldflow_training_coordinate_frame_augmentation:
                    world_trajectory_contract_errors.append(
                        "worldflow_training_coordinate_frame_augmentation=False"
                    )
                if self.worldflow_residual_lr_multiplier is not None:
                    world_trajectory_contract_errors.append("worldflow_residual_lr_multiplier=None")
                if self.worldflow_world_eef_velocity_mode in {
                    "base_decoupled",
                    "base_pose9_euclidean",
                } and (
                    not math.isclose(self.worldflow_trans_weight, 1.0)
                    or not math.isclose(self.worldflow_rot_weight, 1.0)
                ):
                    world_trajectory_contract_errors.append(
                        "worldflow_trans_weight=worldflow_rot_weight=1 for the fixed physical-probe metric"
                    )
                if world_trajectory_contract_errors:
                    raise ValueError(
                        "world_eef_trajectory WorldFlow requires the strict independent-branch "
                        "contract: " + ", ".join(world_trajectory_contract_errors) + "."
                    )
            if self.worldflow_ego_residual_gate_init is not None:
                raise ValueError(
                    "World/Ego residual gates are unsupported; "
                    "worldflow_ego_residual_gate_init must be None."
                )
            if self.worldflow_freeze_pretrained_ego and not math.isclose(
                self.worldflow_pretrained_lr_multiplier,
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(
                    "worldflow_freeze_pretrained_ego=True requires "
                    "worldflow_pretrained_lr_multiplier=1.0 because frozen parameters "
                    "are excluded from the optimizer rather than assigned a nominal learning rate."
                )
            if self.worldflow_pretrained_lr_multiplier <= 0:
                raise ValueError("worldflow_pretrained_lr_multiplier must be positive; zero would freeze Ego.")
            if self.worldflow_new_lr_multiplier <= 0:
                raise ValueError("worldflow_new_lr_multiplier must be positive; zero would freeze World.")
            if (
                self.worldflow_residual_lr_multiplier is not None
                and self.worldflow_residual_lr_multiplier <= 0
            ):
                raise ValueError(
                    "worldflow_residual_lr_multiplier must be positive; zero would freeze the residual head."
                )
            if not 0.0 <= self.worldflow_training_world_to_ego_dropout_probability < 1.0:
                raise ValueError(
                    "worldflow_training_world_to_ego_dropout_probability must be in [0,1); "
                    "1 would remove all World-to-Ego action gradients."
                )
            action_norm = self.normalization_mapping.get("ACTION")
            if action_norm is not NormalizationMode.IDENTITY:
                raise ValueError(
                    "worldflow_enable=True requires ACTION normalization to be IDENTITY because "
                    "the World-Ego bridge interprets Ego pose9 predictions as physical SE(3) transforms."
                )
            if self.worldflow_feature_dim <= 0:
                raise ValueError("worldflow_feature_dim must be positive.")
            if not self.point_action_fusion_enable:
                raise ValueError(
                    "worldflow_enable=True requires point_action_fusion_enable=True so both "
                    "coordinate streams provide point-fused action tokens to the shared expert."
                )
            if self.worldflow_max_points < 0:
                raise ValueError("worldflow_max_points must be non-negative.")
            if self.worldflow_noise_trans_scale < 0:
                raise ValueError("worldflow_noise_trans_scale must be non-negative.")
            if self.worldflow_noise_rot_scale < 0:
                raise ValueError("worldflow_noise_rot_scale must be non-negative.")
            if self.worldflow_eef_probe_radius_m <= 0:
                raise ValueError("worldflow_eef_probe_radius_m must be positive.")
            if self.worldflow_augmentation_trans_scale < 0:
                raise ValueError("worldflow_augmentation_trans_scale must be non-negative.")
            if self.worldflow_augmentation_rot_scale < 0:
                raise ValueError("worldflow_augmentation_rot_scale must be non-negative.")
            if self.worldflow_noise_coupling not in {
                "independent",
                "left_compose_ego",
                "conjugate_ego",
                "projected_ego_chart",
                "projected_ego_path",
            }:
                raise ValueError(
                    "worldflow_noise_coupling must be 'independent', 'left_compose_ego', "
                    "'conjugate_ego', 'projected_ego_chart', or 'projected_ego_path'; "
                    f"got {self.worldflow_noise_coupling!r}."
                )
            if self.worldflow_frame_origin not in {"global", "current_ee"}:
                raise ValueError(
                    "worldflow_frame_origin must be 'global' or 'current_ee'; "
                    f"got {self.worldflow_frame_origin!r}."
                )
            if self.worldflow_scene_frame_origin not in {"carrier", "global"}:
                raise ValueError(
                    "worldflow_scene_frame_origin must be 'carrier' or 'global'; "
                    f"got {self.worldflow_scene_frame_origin!r}."
                )
            if (
                self.worldflow_scene_frame_origin == "global"
                and self.worldflow_training_coordinate_frame_augmentation
            ):
                raise ValueError(
                    "worldflow_scene_frame_origin='global' is incompatible with random "
                    "World-coordinate reparameterization: the fixed global scene frame is "
                    "the intended information source."
                )
            if self.worldflow_action_fusion not in {
                "independent_parallel",
                "cross_attention",
                "point_action_expert_conjugate_bridge",
                "symmetric_twist",
                "conjugate_residual",
                "conjugate_residual_consensus",
                "conjugate_residual_boosting",
                "endpoint_geodesic_consensus",
                "endpoint_residual_boosting",
                "physical_trajectory_cross_attention",
                "world_trajectory_arm_ego_gripper",
            }:
                raise ValueError(
                    "worldflow_action_fusion must be 'independent_parallel', 'cross_attention', "
                    "'point_action_expert_conjugate_bridge', "
                    "'symmetric_twist', "
                    "'conjugate_residual', 'conjugate_residual_consensus', or "
                    "'conjugate_residual_boosting', 'endpoint_geodesic_consensus', or "
                    "'endpoint_residual_boosting', or "
                    "'physical_trajectory_cross_attention', or "
                    "'world_trajectory_arm_ego_gripper'; "
                    f"got {self.worldflow_action_fusion!r}."
                )
            if self.worldflow_action_fusion == "point_action_expert_conjugate_bridge":
                shared_expert_contract_errors = []
                if self.worldflow_target_type != "world_eef_trajectory":
                    shared_expert_contract_errors.append("worldflow_target_type='world_eef_trajectory'")
                if self.worldflow_action_expert_mode != "shared":
                    shared_expert_contract_errors.append("worldflow_action_expert_mode='shared'")
                if self.worldflow_noise_coupling != "left_compose_ego":
                    shared_expert_contract_errors.append("worldflow_noise_coupling='left_compose_ego'")
                if self.worldflow_world_eef_velocity_mode != "base_pose9_euclidean":
                    shared_expert_contract_errors.append(
                        "worldflow_world_eef_velocity_mode='base_pose9_euclidean'"
                    )
                if self.worldflow_current_ee_pose_token:
                    shared_expert_contract_errors.append("worldflow_current_ee_pose_token=False")
                if self.se3_enable:
                    shared_expert_contract_errors.append("se3_enable=False")
                if shared_expert_contract_errors:
                    raise ValueError(
                        "point_action_expert_conjugate_bridge requires symmetric World/Ego "
                        "front-ends meeting in one shared Action Expert: "
                        + ", ".join(shared_expert_contract_errors)
                        + "."
                    )
            if self.worldflow_action_fusion == "independent_parallel":
                calibration_contract_errors = []
                if self.worldflow_target_type != "world_eef_trajectory":
                    calibration_contract_errors.append("worldflow_target_type='world_eef_trajectory'")
                if self.worldflow_action_expert_mode != "independent":
                    calibration_contract_errors.append("worldflow_action_expert_mode='independent'")
                if not self.worldflow_current_ee_pose_token:
                    calibration_contract_errors.append("worldflow_current_ee_pose_token=True")
                if self.worldflow_freeze_pretrained_ego:
                    calibration_contract_errors.append("worldflow_freeze_pretrained_ego=False")
                if self.se3_enable:
                    calibration_contract_errors.append("se3_enable=False")
                if calibration_contract_errors:
                    raise ValueError(
                        "independent_parallel requires two independently supervised, fully trainable "
                        "action flows before interaction: "
                        + ", ".join(calibration_contract_errors)
                        + "."
                    )
            if self.worldflow_current_ee_pose_token and (
                self.worldflow_target_type != "world_eef_trajectory"
                or self.worldflow_action_expert_mode != "independent"
            ):
                raise ValueError(
                    "worldflow_current_ee_pose_token=True requires "
                    "worldflow_target_type='world_eef_trajectory' and "
                    "worldflow_action_expert_mode='independent'."
                )
            if self.worldflow_action_fusion == "world_trajectory_arm_ego_gripper":
                execution_contract_errors = []
                if self.worldflow_target_type != "world_eef_trajectory":
                    execution_contract_errors.append("worldflow_target_type='world_eef_trajectory'")
                if self.worldflow_action_expert_mode != "independent":
                    execution_contract_errors.append("worldflow_action_expert_mode='independent'")
                if not self.worldflow_current_ee_pose_token:
                    execution_contract_errors.append("worldflow_current_ee_pose_token=True")
                if self.se3_enable:
                    execution_contract_errors.append("se3_enable=False")
                if execution_contract_errors:
                    raise ValueError(
                        "world_trajectory_arm_ego_gripper requires the physical execution contract: "
                        + ", ".join(execution_contract_errors)
                        + "."
                    )
            if self.worldflow_action_fusion == "physical_trajectory_cross_attention":
                interaction_contract_errors = []
                if self.worldflow_target_type != "world_eef_trajectory":
                    interaction_contract_errors.append("worldflow_target_type='world_eef_trajectory'")
                if self.worldflow_action_expert_mode != "independent":
                    interaction_contract_errors.append("worldflow_action_expert_mode='independent'")
                if not self.worldflow_current_ee_pose_token:
                    interaction_contract_errors.append("worldflow_current_ee_pose_token=True")
                if self.se3_enable:
                    interaction_contract_errors.append("se3_enable=False")
                if interaction_contract_errors:
                    raise ValueError(
                        "physical_trajectory_cross_attention requires independent complete "
                        "World/Ego trajectories in aligned physical frames: "
                        + ", ".join(interaction_contract_errors)
                        + "."
                    )
            if (
                self.worldflow_endpoint_residual_rate_parameterization
                and self.worldflow_action_fusion != "endpoint_residual_boosting"
            ):
                raise ValueError(
                    "worldflow_endpoint_residual_rate_parameterization=True requires "
                    "worldflow_action_fusion='endpoint_residual_boosting'."
                )
            if (
                self.worldflow_endpoint_residual_ego_frame_parameterization
                and self.worldflow_action_fusion != "endpoint_residual_boosting"
            ):
                raise ValueError(
                    "worldflow_endpoint_residual_ego_frame_parameterization=True requires "
                    "worldflow_action_fusion='endpoint_residual_boosting'."
                )
            if (
                self.worldflow_endpoint_residual_body_frame_parameterization
                and self.worldflow_action_fusion != "endpoint_residual_boosting"
            ):
                raise ValueError(
                    "worldflow_endpoint_residual_body_frame_parameterization=True requires "
                    "worldflow_action_fusion='endpoint_residual_boosting'."
                )
            if (
                self.worldflow_body_velocity_residual_parameterization
                and self.worldflow_action_fusion != "endpoint_residual_boosting"
            ):
                raise ValueError(
                    "worldflow_body_velocity_residual_parameterization=True requires "
                    "worldflow_action_fusion='endpoint_residual_boosting'."
                )
            if (
                self.worldflow_endpoint_residual_ego_frame_parameterization
                and self.worldflow_endpoint_residual_body_frame_parameterization
            ):
                raise ValueError(
                    "Select only one endpoint residual frame parameterization: "
                    "carrier-frame left twist or endpoint body-frame right twist."
                )
            if self.worldflow_body_velocity_residual_parameterization and any(
                (
                    self.worldflow_endpoint_residual_rate_parameterization,
                    self.worldflow_endpoint_residual_ego_frame_parameterization,
                    self.worldflow_endpoint_residual_body_frame_parameterization,
                )
            ):
                raise ValueError(
                    "Body-velocity residuals are mutually exclusive with finite endpoint-residual "
                    "rate and frame parameterizations."
                )
            if self.worldflow_action_fusion in {
                "symmetric_twist",
                "conjugate_residual",
                "conjugate_residual_consensus",
                "conjugate_residual_boosting",
            } and not self.se3_enable:
                raise ValueError(f"worldflow_action_fusion={self.worldflow_action_fusion!r} requires se3_enable=True.")
            if self.worldflow_action_fusion in {
                "endpoint_geodesic_consensus",
                "endpoint_residual_boosting",
            } and (
                self.se3_enable or self.worldflow_noise_coupling != "projected_ego_path"
            ):
                raise ValueError(
                    f"worldflow_action_fusion={self.worldflow_action_fusion!r} requires the "
                    "checkpoint-compatible legacy Ego chart (se3_enable=False) and "
                    "worldflow_noise_coupling='projected_ego_path'."
                )
            if (
                self.worldflow_training_world_to_ego_dropout_probability > 0
                and self.worldflow_action_fusion != "endpoint_residual_boosting"
            ):
                raise ValueError(
                    "worldflow_training_world_to_ego_dropout_probability is supported only for "
                    "worldflow_action_fusion='endpoint_residual_boosting'."
                )
            if self.worldflow_training_residual_anchor_stop_gradient and (
                self.worldflow_action_fusion != "endpoint_residual_boosting"
                or self.worldflow_training_world_to_ego_dropout_probability <= 0
            ):
                raise ValueError(
                    "worldflow_training_residual_anchor_stop_gradient requires training-time "
                    "World-to-Ego dropout and "
                    "worldflow_action_fusion='endpoint_residual_boosting'."
                )
            if self.worldflow_training_ego_priority_gradient_projection and (
                self.worldflow_action_fusion != "endpoint_residual_boosting"
                or self.worldflow_training_world_to_ego_dropout_probability <= 0
                or not self.worldflow_training_residual_anchor_stop_gradient
            ):
                raise ValueError(
                    "worldflow_training_ego_priority_gradient_projection requires "
                    "endpoint_residual_boosting with positive training-time World-to-Ego "
                    "dropout and residual-anchor stop-gradient."
                )
            if self.worldflow_training_shared_gradient_ego_tangent_projection and (
                self.worldflow_action_fusion != "endpoint_residual_boosting"
                or self.worldflow_training_world_to_ego_dropout_probability <= 0
                or not self.worldflow_training_residual_anchor_stop_gradient
            ):
                raise ValueError(
                    "worldflow_training_shared_gradient_ego_tangent_projection requires "
                    "endpoint_residual_boosting with positive training-time World-to-Ego "
                    "dropout and residual-anchor stop-gradient."
                )
            if (
                self.worldflow_training_ego_priority_gradient_projection
                and self.worldflow_training_shared_gradient_ego_tangent_projection
            ):
                raise ValueError(
                    "Select only one WorldFlow shared-gradient projection strategy."
                )
            if self.worldflow_noise_coupling == "conjugate_ego" and not (
                self.pose9_action_noise_enable or self.se3_enable
            ):
                raise ValueError(
                    "worldflow_noise_coupling='conjugate_ego' requires "
                    "pose9_action_noise_enable=True or se3_enable=True so the Ego prior is a valid "
                    "SE(3) transform."
                )
            if self.worldflow_noise_coupling in {
                "projected_ego_chart",
                "projected_ego_path",
            } and (
                self.pose9_action_noise_enable or self.se3_enable
            ):
                raise ValueError(
                    "projected Ego chart coupling is only for the legacy Euclidean Ego chart; "
                    "use 'conjugate_ego' with a valid-pose9 or SE(3) prior."
                )
            if self.worldflow_noise_coupling in {
                "left_compose_ego",
                "conjugate_ego",
                "projected_ego_chart",
                "projected_ego_path",
            } and (
                self.worldflow_noise_trans_scale != 0.15 or self.worldflow_noise_rot_scale != 0.20
            ):
                warnings.warn(
                    "worldflow_noise_trans_scale/rot_scale are ignored when "
                    "the World prior is derived from the Ego prior and current pose.",
                    stacklevel=2,
                )
            ego_random_prior = (
                any(
                    float(getattr(self, name)) > 0.0
                    for name in (
                        "se3_noise_trans_scale",
                        "se3_noise_rot_scale",
                        "se3_noise_gripper_scale",
                    )
                )
                if self.se3_enable
                else self.pose9_action_noise_enable
                and any(
                    float(getattr(self, name)) > 0.0
                    for name in (
                        "pose9_action_noise_trans_scale",
                        "pose9_action_noise_rot_scale",
                        "pose9_action_noise_gripper_scale",
                    )
                )
            )
            if (
                self.worldflow_noise_coupling == "independent"
                and self.worldflow_target_type == "legacy_eef"
            ):
                detail = (
                    " Both priors are valid poses, but their random origins still describe different "
                    "physical trajectories."
                    if ego_random_prior
                    else " The legacy Ego prior is also an unconstrained rotation-6D vector rather than "
                    "an SE(3) pose."
                )
                warnings.warn(
                    "WorldFlow and Ego use independent random priors. This preserves legacy behavior "
                    "but does not satisfy G_0=C B_0 C^{-1}." + detail + " Use "
                    "pose9_action_noise_enable=True (or se3_enable=True) together with "
                    "worldflow_noise_coupling='conjugate_ego' for geometrically coupled double-flow training.",
                    stacklevel=2,
                )
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError("worldflow_enable=True is not compatible with RTC.")
            if self.worldflow_action_expert_layers != -1 or self.worldflow_action_expert_dropout != 0.0:
                warnings.warn(
                    "worldflow_action_expert_layers/dropout are legacy v0.5 options and are ignored "
                    "because World and Ego now share one Action Expert.",
                    stacklevel=2,
                )

        if self.se3_final_correction_enable:
            warnings.warn(
                "se3_final_correction_enable is kept only for CLI compatibility and is ignored. "
                "The PCA-style final correction branch has been removed from the active policy path.",
                stacklevel=2,
            )
        if self.vla_adapter_enable and not self.load_vlm_weights:
            warnings.warn(
                "vla_adapter_enable=True requires pretrained VLM weights; "
                "overriding load_vlm_weights=True.",
                stacklevel=2,
            )
            self.load_vlm_weights = True
        if self.load_action_expert_weights:
            action_expert_source = self.action_expert_weights_path or self.vlm_weights_path
            if action_expert_source is None or str(action_expert_source).strip().lower() in {
                "",
                "0",
                "false",
                "none",
                "off",
            }:
                raise ValueError(
                    "load_action_expert_weights=True requires action_expert_weights_path "
                    "or vlm_weights_path pointing to a complete SmolVLA policy checkpoint."
                )
        if self.vla_adapter_enable and self.vla_adapter_freeze_vlm:
            if not self.train_expert_only:
                warnings.warn(
                    "vla_adapter_enable=True with vla_adapter_freeze_vlm=True freezes the VLM; "
                    "overriding train_expert_only=True.",
                    stacklevel=2,
                )
                self.train_expert_only = True
            self.freeze_vision_encoder = True

    def flow_contract_summary(self) -> str:
        """Describe the final resolved temporal and flow-origin contract."""

        if self.se3_enable:
            scales = (
                float(self.se3_noise_trans_scale),
                float(self.se3_noise_rot_scale),
                float(self.se3_noise_gripper_scale),
            )
            if self.se3_twist_head_mode == "pose9_chart_endpoint":
                origin = (
                    "pose9_chart_zero_projected_to_se3"
                    if scales == (0.0, 0.0, 0.0)
                    else "v0.4.2_pose9_chart_gaussian_projected_to_se3"
                )
                origin = (
                    f"{origin}(trans={scales[0]},rot6d={scales[1]},"
                    f"gripper={scales[2]})"
                )
            else:
                origin = "se3_identity_deterministic" if scales == (0.0, 0.0, 0.0) else "se3_manifold_random"
                origin = f"{origin}(trans_m={scales[0]},rot_rad={scales[1]},gripper_m={scales[2]})"
            flow = f"se3_geodesic_left_trivialized(head={self.se3_twist_head_mode})"
        elif self.pose9_action_noise_enable:
            scales = (
                float(self.pose9_action_noise_trans_scale),
                float(self.pose9_action_noise_rot_scale),
                float(self.pose9_action_noise_gripper_scale),
            )
            origin = "pose9_identity_deterministic" if scales == (0.0, 0.0, 0.0) else "pose9_valid_se3_random"
            origin = f"{origin}(trans_m={scales[0]},rot_rad={scales[1]},gripper_m={scales[2]})"
            flow = "pose9_euclidean"
        else:
            origin = "v0.4.2_raw_channel_gaussian(std=0.1)"
            flow = "channel_euclidean"
        if self.worldflow_enable:
            target_contract = self.worldflow_target_type
            if target_contract == "legacy_eef":
                target_contract = (
                    "legacy_eef_commanded_required"
                    if self.worldflow_require_action_target_sidecar
                    else "legacy_eef_fallback_allowed"
                )
            world = (
                f",worldflow={self.worldflow_noise_coupling},"
                f"worldflow_targets={target_contract},"
                f"worldflow_world_eef_velocity={self.worldflow_world_eef_velocity_mode},"
                f"worldflow_reference={self.worldflow_reference_frame},"
                f"worldflow_action_expert={self.worldflow_action_expert_mode},"
                f"worldflow_action_fusion={self.worldflow_action_fusion},"
                f"worldflow_current_pose_token={self.worldflow_current_ee_pose_token},"
                f"worldflow_eef_probe_radius_m={self.worldflow_eef_probe_radius_m}"
            )
        else:
            world = ",worldflow=disabled"
        return (
            f"action_chunk_start_offset={int(self.action_chunk_start_offset)},"
            f"online_action_index=0,ego_origin={origin},ego_flow={flow}{world},"
            f"num_steps={int(self.num_steps)}"
        )

    def validate_features(self) -> None:
        selected = set(self.selected_rgb_camera_views)
        for key in list(self.input_features):
            if not key.startswith(f"{OBS_IMAGES}."):
                continue
            camera = key[len(OBS_IMAGES) + 1 :]
            # ``front`` predates camera-view metadata in RLBench checkpoints.
            # Keep it when loading those legacy configs; explicit newer
            # checkpoints still validate ``front`` in _parse_camera_views.
            if camera in {"agentview", "robot0_eye_in_hand"} and camera not in selected:
                del self.input_features[key]

        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    @staticmethod
    def _parse_camera_views(value, *, field_name: str) -> tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            parts = [str(part).strip() for part in value]
        else:
            text = str(value).strip().strip("[]")
            parts = [part.strip().strip("\"'") for part in text.split(",")]
        views = tuple(part for part in parts if part) or ("agentview",)
        supported = {"agentview", "front", "robot0_eye_in_hand"}
        unknown = [view for view in views if view not in supported]
        if unknown:
            raise ValueError(
                f"Unsupported {field_name} view(s) {unknown}; supported views are {sorted(supported)}."
            )
        if len(set(views)) != len(views):
            raise ValueError(f"{field_name} contains duplicates: {views}.")
        return views

    @property
    def selected_camera_views(self) -> tuple[str, ...]:
        return self._parse_camera_views(self.camera_views, field_name="camera_views")

    @property
    def selected_rgb_camera_views(self) -> tuple[str, ...]:
        value = self.camera_views if self.rgb_camera_views is None else self.rgb_camera_views
        return self._parse_camera_views(value, field_name="rgb_camera_views")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        start = int(self.action_chunk_start_offset)
        return list(range(start, start + self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
