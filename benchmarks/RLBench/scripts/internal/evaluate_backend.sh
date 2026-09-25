#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLBENCH_ROOT="${RLBENCH_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TOOLS_DIR="${RLBENCH_ROOT}/scripts/tools"

# Checkpoints may contain paths from the training machine. Keep the model
# assets beside the benchmark by default, while allowing explicit overrides.
VLM_MODEL_ROOT="${VLM_MODEL_ROOT:-${RLBENCH_ROOT}/../vlm_model}"
export VLM_MODEL_NAME="${VLM_MODEL_NAME:-${VLM_MODEL_ROOT}/SmolVLM2-500M-Video-Instruct}"
export VLM_WEIGHTS_PATH="${VLM_WEIGHTS_PATH:-${VLM_MODEL_ROOT}/smolvla_base}"

# Internal compatibility backend. New evaluations should use ./evaluate_profile.sh, which
# resolves checkpoint-task parameters from rlbench_eval_registry.json.

# Tasks. Passing --tasks/--task on the command line overrides this list.

# 这里选择保不保留视频
eval_save_video="${EVAL_SAVE_VIDEO:-1}"
eval_save_action_visualizations="${EVAL_SAVE_ACTION_VISUALIZATIONS:-1}"
eval_failure_artifacts_only="${EVAL_FAILURE_ARTIFACTS_ONLY:-1}"
eval_save_frame_pointclouds="${EVAL_SAVE_FRAME_POINTCLOUDS:-0}"
eval_frame_pointcloud_every_n_frames="${EVAL_FRAME_POINTCLOUD_EVERY_N_FRAMES:-2}"

eval_episodes=10
eval_policy_path="${EVAL_POLICY_PATH:-}"
eval_planner_max_time_ms="${EVAL_PLANNER_MAX_TIME_MS:-${RLBENCH_PLANNER_MAX_TIME_MS:-50}}"
# 如果是delta模式，这个参数启用，表示夹爪开闭的阈值是相对于上一次模型调用的夹爪宽度。
eval_gripper_mode="delta_width_initial_sync" # absolute_width | libero_delta (legacy) | delta_width_initial_sync
eval_gripper_delta_threshold="${EVAL_GRIPPER_DELTA_THRESHOLD:-0.003}"
eval_gripper_delta_open_threshold="${EVAL_GRIPPER_DELTA_OPEN_THRESHOLD:-0.0025}"
eval_gripper_delta_close_threshold="${EVAL_GRIPPER_DELTA_CLOSE_THRESHOLD:-0.003}"
eval_gripper_delta_alignment="current_minus_previous"
# 0808/022000 在 DISPLAY=:99 上验证过的 Panda 夹爪协议：模型输出宽度
# 大于 0.01 m 判为打开。water_plants 在任务预设中单独使用 0.075 m。
eval_gripper_open_threshold="${EVAL_GRIPPER_OPEN_THRESHOLD:-0.04}"
# 设为 1 后，第一次关闭夹爪后，整个 episode 后续都强制保持关闭。
eval_gripper_lock_after_close=0       # 0 = normal; 1 = never reopen after close
eval_water_plant_collision="${EVAL_WATER_PLANT_COLLISION:-enabled}" # disabled | enabled; physical plant blocking
eval_water_drop_collision="${EVAL_WATER_DROP_COLLISION:-original}"   # disabled | original
eval_simulation_timestep=0
eval_exec_action_steps="${EVAL_EXEC_ACTION_STEPS:-28}"
eval_num_points=20000
eval_gripper_points="${EVAL_GRIPPER_POINTS:-500}"
eval_gripper_len="${EVAL_GRIPPER_LEN:-0.11}"
eval_gripper_template="reap"
eval_add_gripper_cloud=1 # 1 = merge virtual gripper points into model input; 0 = scene only
eval_max_model_calls="${EVAL_MAX_MODEL_CALLS:-10}"
# A rejected target keeps the current simulator state, discards the remainder
# of the stale chunk, and asks the policy for a fresh action chunk.

eval_controller_error_retries=0
eval_controller_error_retry_timeout_seconds=0
eval_controller_error_mode="continue_episode" # retry_episode | continue_episode
eval_controller_continue_max_errors="${EVAL_CONTROLLER_CONTINUE_MAX_ERRORS:-200}"
eval_action_index=0
if [[ -n "${EVAL_COLLISION_CHECKING+x}" ]]; then
    eval_collision_checking_explicit=1
else
    eval_collision_checking_explicit=0
fi
eval_collision_checking="${EVAL_COLLISION_CHECKING:-0}" # umbrella defaults to enabled; other tasks default to disabled
eval_verbose_terminal=0                   # 0 = progress summaries only; 1 = print evaluator details

# Runtime and resources.
python_bin="${PYTHON:-python}"
eval_rlbench_headless="${EVAL_RLBENCH_HEADLESS:-}"
coppeliasim_root="${COPPELIASIM_ROOT:-${RLBENCH_ROOT}/../CoppeliaSim}"
eval_base_dir="${EVAL_BASE_DIR:-${RLBENCH_ROOT}/eval}"
eval_root="${EVAL_ROOT:-}"
eval_log_dir="${EVAL_LOG_DIR:-}"
eval_gpu_ids="${EVAL_GPU_IDS:-0}"
eval_display_base="${EVAL_DISPLAY_BASE:-99}"

# Model and episode selection.
eval_variation=0
eval_seed=100
eval_model_noise_seed=20260801
eval_device="cuda"
eval_pointseg_device=""
eval_dataset_root=""
eval_dataset_episodes=""

# Action chunk and arm controller.
eval_arm_action_mode="planning"          # ik | planning | joint_velocity_waypoint | franka_ik_servo | franka_cartesian_servo
eval_execution_mode="dataset_step"       # adaptive | dataset_step | bounded_step
# PointACT-compatible controller post-processing. A successfully reached
# waypoint keeps the normal chunk cadence. A failed waypoint discards the
# remaining stale rows and invokes the model again from the latest observation.
eval_clip_within_workspace="${EVAL_CLIP_WITHIN_WORKSPACE:-0}"
eval_mover_max_tries="${EVAL_MOVER_MAX_TRIES:-2}"
eval_mover_position_tolerance="${EVAL_MOVER_POSITION_TOLERANCE:-0.001}"
eval_mover_rotation_tolerance="${EVAL_MOVER_ROTATION_TOLERANCE:-0.05491}"
eval_mover_gripper_position_tolerance="${EVAL_MOVER_GRIPPER_POSITION_TOLERANCE:-0.001}"
eval_mover_gripper_rotation_tolerance="${EVAL_MOVER_GRIPPER_ROTATION_TOLERANCE:-0.05491}"
eval_reinfer_on_control_error="${EVAL_REINFER_ON_CONTROL_ERROR:-1}"
eval_gripper_after_reach="${EVAL_GRIPPER_AFTER_REACH:-1}"
eval_pointact_pyrep_compat="${EVAL_POINTACT_PYREP_COMPAT:-1}"
eval_max_eef_position_step=0
eval_max_eef_rotation_step=0
eval_jacobian_position_threshold=0.008
eval_jacobian_rotation_threshold=0.05
eval_jacobian_midpoint_max_depth=4
eval_waypoint_position_tolerance=0.002
eval_waypoint_rotation_tolerance=0.03
eval_waypoint_max_control_steps=64

# Joint-velocity waypoint controller (only used when arm mode is enabled above).
eval_joint_velocity_kp=2.0
eval_joint_velocity_max_speed=1.0
eval_joint_velocity_joint_tolerance=0.01
eval_joint_velocity_stall_steps=8

# Point-cloud and gripper input.
eval_image_size=256
# Video and per-chunk PLY/PNG action diagnostics are enabled by default. The
# default failure-only retention policy skips MP4 encoding for successful
# episodes and removes their temporary action visualizations after success is
# known. Set EVAL_FAILURE_ARTIFACTS_ONLY=0 to retain diagnostics for every run.

eval_video_width=512
eval_video_height=512
eval_video_fps=20
eval_video_refresh_rgb="${EVAL_VIDEO_REFRESH_RGB:-0}" # opt-in: explicit renders can perturb legacy camera observations
eval_save_control_log="${EVAL_SAVE_CONTROL_LOG:-0}"
eval_log_control_details="${EVAL_LOG_CONTROL_DETAILS:-0}"
eval_save_action_records="${EVAL_SAVE_ACTION_RECORDS:-1}"
eval_save_action_chunks="${EVAL_SAVE_ACTION_CHUNKS:-0}"

eval_action_vis_every_n_frames="${EVAL_ACTION_VIS_EVERY_N_FRAMES:-20}"
eval_action_vis_max_points=50000
eval_action_vis_image_width=768
eval_action_vis_point_mode="prob"        # full | prob
eval_visualize_foreground=0
eval_draw_pour_point="0"              # auto | 0 | 1
eval_draw_task_stages="${EVAL_DRAW_TASK_STAGES:-0}"  # auto | 0 | 1; water_plants stage overlay
eval_draw_phone_success_sensor="${EVAL_DRAW_PHONE_SUCCESS_SENSOR:-0}"  # auto | 0 | 1; phone_on_base sensor overlay

task_list=(
    close_box
    close_fridge
    close_laptop_lid
    phone_on_base
    stack_wine
    sweep_to_dustpan
    take_frame_off_hanger
    take_umbrella_out_of_umbrella_stand
    toilet_seat_down
    water_plants
)

# Defaults above are the validated common protocol. Keep task presets limited
# to intentional deviations so a plain invocation reproduces the 2026-08-23
# DISPLAY=:99 evaluation without a long list of command-line overrides.
declare -a preset_close_box=()
declare -a preset_close_fridge=()
declare -a preset_close_laptop_lid=()
declare -a preset_phone_on_base=()
declare -a preset_stack_wine=()
declare -a preset_sweep_to_dustpan=()
declare -a preset_take_frame_off_hanger=()
declare -a preset_take_umbrella_out_of_umbrella_stand=()
declare -a preset_toilet_seat_down=()
declare -a preset_water_plants=()

get_task_preset() {
    local task_name="$1"
    local preset_name="preset_${task_name}"
    TASK_PRESET_ARGS=()
    if [[ -n "${EVAL_CHECKPOINT_TASK_REGISTRY:-}" || -n "${EVAL_CHECKPOINT_ID:-}" ]]; then
        if [[ -z "${EVAL_CHECKPOINT_TASK_REGISTRY:-}" || -z "${EVAL_CHECKPOINT_ID:-}" ]]; then
            echo "EVAL_CHECKPOINT_TASK_REGISTRY and EVAL_CHECKPOINT_ID must be set together" >&2
            return 2
        fi
        local registry_resolver="${TOOLS_DIR}/eval_registry.py"
        if [[ ! -f "${registry_resolver}" ]]; then
            echo "Registry resolver not found: ${registry_resolver}" >&2
            return 2
        fi
        mapfile -d '' -t TASK_PRESET_ARGS < <(
            "${PYTHON}" "${registry_resolver}" \
                --registry "${EVAL_CHECKPOINT_TASK_REGISTRY}" \
                args --checkpoint "${EVAL_CHECKPOINT_ID}" --task "${task_name}" --format null
        )
        return 0
    fi
    if declare -p "${preset_name}" &>/dev/null; then
        eval "TASK_PRESET_ARGS=(\"\${${preset_name}[@]}\")"
    fi
}

get_task_collision_args() {
    local task_name="$1"
    local task_arg
    # Registry task arguments are the most specific layer.  If they already
    # select a collision mode, do not prepend the global/default opposite flag:
    # argparse treats the pair as mutually exclusive instead of last-wins.
    for task_arg in "${TASK_PRESET_ARGS[@]}"; do
        if [[ "${task_arg}" == "--collision-checking" || "${task_arg}" == "--no-collision-checking" ]]; then
            TASK_COLLISION_ARGS=()
            return 0
        fi
    done
    TASK_COLLISION_ARGS=("${COLLISION_ARGS[@]}")
}

get_task_planner_max_time_ms() {
    TASK_PLANNER_MAX_TIME_MS="${EVAL_PLANNER_MAX_TIME_MS}"
    local -a planner_args=("${TASK_PRESET_ARGS[@]}" "${EVAL_ARGS[@]}")
    local planner_index
    for ((planner_index = 0; planner_index < ${#planner_args[@]}; planner_index++)); do
        case "${planner_args[planner_index]}" in
            --planner-max-time-ms)
                TASK_PLANNER_MAX_TIME_MS="${planner_args[planner_index + 1]}"
                planner_index=$((planner_index + 1))
                ;;
            --planner-max-time-ms=*)
                TASK_PLANNER_MAX_TIME_MS="${planner_args[planner_index]#*=}"
                ;;
        esac
    done
}

# action-index：模型每次输出一个包含 32 行动作的 action chunk。
# 设为 0 表示从 chunk[0] 开始执行，不跳过第一行。
#
# exec-action-steps：每次模型预测后，从 action-index 开始连续执行几个 waypoint。
# 当前值 16 表示从 action-index 开始执行连续 16 个 waypoint。
# 这些 waypoint 都相对于“本次模型调用开始时”的同一个 EEF 坐标系。
# Python 代码使用同一个 chunk_anchor_world，把四个相对位姿分别转换成世界绝对位姿。
# dataset_step + planning 不改变 chunk cadence；当前 PointACT Mover 会在未到位时
# 对同一个 waypoint 最多重发 10 次，但不会重新调用模型。
# 需要 adaptive 或 planning 时，可以在命令末尾显式追加对应参数覆盖默认值。
#
# gripper-mode：当前默认使用 absolute_width。
# 模型输出物理夹爪宽度，宽度大于 EVAL_GRIPPER_OPEN_THRESHOLD 时打开，否则闭合。
# 默认绝对宽度阈值为 0.01 m；delta 模式只使用 EVAL_GRIPPER_DELTA_THRESHOLD，
# 不读取 absolute-width 的 open threshold。
#
# 以下 max-eef、jacobian 和 waypoint 参数由 adaptive 模式使用，用于插值、
# 重复跟踪和 waypoint 到达判定。
#
# max-eef-position-step 和 max-eef-rotation-step：是否永久截断完整 waypoint。
# 0 表示关闭截断，执行器最终仍然以模型预测的完整世界位姿为目标。
# 如果 position 设置为 0.03，距离大于 3 cm 的模型目标会被永久截成 3 cm。
# 当前要求是完整执行每个 waypoint，因此这两个参数都设为 0。
#
# jacobian-position-threshold：单次发给 Jacobian IK 的最大平移变化。
# 当前设置 0.008 米，即 8 mm。当前实际 EEF 距离 waypoint 超过 8 mm 时，
# 代码先取当前位置和目标的中点；如果中点仍超过 8 mm，就继续取更近的中点。
#
# jacobian-rotation-threshold：单次发给 Jacobian IK 的最大旋转变化。
# 当前设置 0.05 rad，约 2.86 度。旋转超过该值时也会反复插入中间姿态。
# 上面两个 threshold 决定“什么时候插中点”，不是“waypoint 是否完成”的判据。
#
# jacobian-midpoint-max-depth：已经足够小的子目标仍发生 IK 错误时的保护。
# IK 失败后，在当前实际位姿和失败目标之间额外递归插入中点。
# 当前值 4 表示最多递归二分 4 层，仍失败就放弃当前 chunk 并重新预测。
#
# waypoint-position-tolerance 和 waypoint-rotation-tolerance：最终到达判据。
# 只有实际 EEF 与完整 waypoint 的位置误差 <= 0.002 米（2 mm），
# 旋转误差 <= 0.03 rad（约 1.72 度），并且夹爪开闭状态一致，
# 当前 waypoint 才算完成，之后才能执行同一 chunk 中的下一个 waypoint。
#
# waypoint-max-control-steps：一个 waypoint 最多允许的实际环境控制动作数量。
# 中点动作、重复跟踪动作和 IK 失败后递归产生的动作都属于这个执行过程。
# 当前上限为 64，超过后仍未达到最终容差，就放弃当前 chunk 并重新预测，
# 防止不可达目标或控制器不收敛造成无限循环。
#
# video-width 和 video-height：最终保存的 MP4 视频分辨率。
# 当前设置为 512 x 512。这里只缩放保存到视频中的画面，
# 不会改变模型输入的 256 x 256 front RGB，也不会改变相机点云分辨率。
# video-fps 只控制视频播放帧率，不改变 RLBench 仿真控制频率。
#
# save-control-log：保存模型输出、坐标转换结果和实际机器人状态到本次评测输出目录
# 的 control.log，但不刷屏终端。追加 --log-control-details 才同时打印这些明细。
# [model-state] 是模型调用开始时的实际世界 state 和送入模型的 identity state。
# [model-action] 是模型原始相对 EEF 10D action 和转换后的世界绝对目标。
# [control-before] 是真正送入 task_env.step() 的世界坐标 8D action。
# [control-after] 是执行该 action 后 RLBench 返回的实际世界 state。
# [waypoint-error] 是当前实际 state 到完整 waypoint 的位置和旋转误差。
# [control-ik-failed] 会记录发生 Jacobian IK 错误时的 state、action 和异常文本。
# 终端保留原始浮点精度；control.log 中的所有浮点数最多保存两位小数。
#
# 本 baseline 使用 dataset_step + planning，并启用同目标 Mover 重试。
# 训练标签的夹爪是绝对物理宽度，不是 LIBERO 的 delta 事件；宽度 > 0.04 m
# 发送 RLBench open，否则发送 closed。adaptive + libero_delta 仍可通过命令行覆盖。
#
# save-action-visualizations：每隔 32 个实际环境控制帧保存完整 action chunk。
# PLY 使用模型实际看到的当前 EEF 坐标系点云，并在同一坐标系画出 32 个
# action 的位置轨迹和姿态三轴。轨迹从蓝/青色开始，中间绿色，最后红色。
# 同一次LitePT前向传播得到的 operation_prob 会叠加到场景点：蓝=低前景概率，
# 青/黄=中等，红=高。action轨迹仍保持自己的时间渐变色。PLY 额外保存
# operation_prob、selection_score、source_point_index、point_kind；point_kind=0
# 是场景点，point_kind=1 是未来action轨迹/姿态标记，后者的概率字段为nan。
# action-vis-point-mode有两个值：prob按operation_prob热力着色；full保留全部
# 场景点的原始RGB。两种模式都会画未来32步action。当前默认使用prob；需要
# 看原始全景点时，在命令末尾传入 --action-vis-point-mode full 即可覆盖。
# PNG 使用 RLBench front 相机内外参，把 32 个世界目标投影到真实 RGB 图像，
# 并标出 1 到 32。action 可视化只保存 PLY 和 PNG，不额外生成 JSON。
# 需要保存每次模型调用的原始 action chunk 时，在命令末尾追加 --save-action-chunks；
# 文件会写到本次 run/task/action_chunks/episode_XXX/frame_XXXXXX_model_call_XXXX.npy。
# frame 0 保存一次；之后每个 CoppeliaSim physics step 保存一帧，包括规划器
# path.step() 和夹爪开合/释放后的等待步，因此视频帧数不再等于高层 task_env.step() 数。
#
# 评测任务始终串行运行：任务按 TASKS 顺序执行，同一时刻只占用一个 GPU
# 和一个 DISPLAY。每个任务仍使用独立的输出目录和日志文件。

configure_coppeliasim() {
    coppeliasim_root="${COPPELIASIM_ROOT:-${coppeliasim_root}}"
    qt_platform_path="${QT_QPA_PLATFORM_PLUGIN_PATH:-${coppeliasim_root}}"
    qt_plugin_path="${QT_PLUGIN_PATH:-}"
    qt_xcb_gl_integration="${QT_XCB_GL_INTEGRATION:-}"
    glx_vendor_library="${__GLX_VENDOR_LIBRARY_NAME:-}"
    eval_rlbench_headless="${EVAL_RLBENCH_HEADLESS:-1}"
    if [[ ! -x "${coppeliasim_root}/coppeliaSim" ]]; then
        echo "CoppeliaSim executable not found: ${coppeliasim_root}/coppeliaSim" >&2
        exit 2
    fi
}

configure_coppeliasim
# Internal aliases keep the existing CLI/environment interface; edit only the
# lowercase block above.
TASK_LIST=("${task_list[@]}")
PYTHON="${python_bin}"
COPPELIASIM_ROOT="${coppeliasim_root}"
QT_QPA_PLATFORM_PLUGIN_PATH="${qt_platform_path}"
QT_PLUGIN_PATH="${qt_plugin_path}"
QT_XCB_GL_INTEGRATION="${qt_xcb_gl_integration}"
__GLX_VENDOR_LIBRARY_NAME="${glx_vendor_library}"
EVAL_BASE_DIR="${eval_base_dir}"
EVAL_ROOT="${eval_root}"
EVAL_LOG_DIR="${eval_log_dir}"
EVAL_GPU_IDS="${eval_gpu_ids}"
EVAL_DISPLAY_BASE="${eval_display_base}"
EVAL_PLANNER_MAX_TIME_MS="${eval_planner_max_time_ms}"
EVAL_POLICY_PATH="${EVAL_POLICY_PATH:-${eval_policy_path}}"
EVAL_EPISODES="${EVAL_EPISODES:-${eval_episodes}}"
EVAL_VARIATION="${eval_variation}"
EVAL_SEED="${eval_seed}"
EVAL_MODEL_NOISE_SEED="${EVAL_MODEL_NOISE_SEED:-${eval_model_noise_seed}}"
EVAL_DEVICE="${eval_device}"
EVAL_POINTSEG_DEVICE="${eval_pointseg_device}"
EVAL_DATASET_ROOT="${eval_dataset_root}"
EVAL_DATASET_EPISODES="${eval_dataset_episodes}"
EVAL_MAX_MODEL_CALLS="${EVAL_MAX_MODEL_CALLS:-${eval_max_model_calls}}"
EVAL_ACTION_INDEX="${EVAL_ACTION_INDEX:-${eval_action_index}}"
EVAL_EXEC_ACTION_STEPS="${EVAL_EXEC_ACTION_STEPS:-${eval_exec_action_steps}}"
EVAL_SIMULATION_TIMESTEP="${EVAL_SIMULATION_TIMESTEP:-${eval_simulation_timestep}}"
EVAL_ARM_ACTION_MODE="${EVAL_ARM_ACTION_MODE:-${eval_arm_action_mode}}"
EVAL_EXECUTION_MODE="${EVAL_EXECUTION_MODE:-${eval_execution_mode}}"
EVAL_CLIP_WITHIN_WORKSPACE="${eval_clip_within_workspace}"
EVAL_MOVER_MAX_TRIES="${eval_mover_max_tries}"
EVAL_MOVER_POSITION_TOLERANCE="${eval_mover_position_tolerance}"
EVAL_MOVER_ROTATION_TOLERANCE="${eval_mover_rotation_tolerance}"
EVAL_MOVER_GRIPPER_POSITION_TOLERANCE="${eval_mover_gripper_position_tolerance}"
EVAL_MOVER_GRIPPER_ROTATION_TOLERANCE="${eval_mover_gripper_rotation_tolerance}"
EVAL_REINFER_ON_CONTROL_ERROR="${eval_reinfer_on_control_error}"
EVAL_GRIPPER_AFTER_REACH="${eval_gripper_after_reach}"
EVAL_POINTACT_PYREP_COMPAT="${eval_pointact_pyrep_compat}"
EVAL_COLLISION_CHECKING="${EVAL_COLLISION_CHECKING:-${eval_collision_checking}}"
EVAL_MAX_EEF_POSITION_STEP="${eval_max_eef_position_step}"
EVAL_MAX_EEF_ROTATION_STEP="${eval_max_eef_rotation_step}"
EVAL_JACOBIAN_POSITION_THRESHOLD="${eval_jacobian_position_threshold}"
EVAL_JACOBIAN_ROTATION_THRESHOLD="${eval_jacobian_rotation_threshold}"
EVAL_JACOBIAN_MIDPOINT_MAX_DEPTH="${eval_jacobian_midpoint_max_depth}"
EVAL_WAYPOINT_POSITION_TOLERANCE="${eval_waypoint_position_tolerance}"
EVAL_WAYPOINT_ROTATION_TOLERANCE="${eval_waypoint_rotation_tolerance}"
EVAL_WAYPOINT_MAX_CONTROL_STEPS="${eval_waypoint_max_control_steps}"
EVAL_JOINT_VELOCITY_KP="${eval_joint_velocity_kp}"
EVAL_JOINT_VELOCITY_MAX_SPEED="${eval_joint_velocity_max_speed}"
EVAL_JOINT_VELOCITY_JOINT_TOLERANCE="${eval_joint_velocity_joint_tolerance}"
EVAL_JOINT_VELOCITY_STALL_STEPS="${eval_joint_velocity_stall_steps}"
EVAL_NUM_POINTS="${EVAL_NUM_POINTS:-${eval_num_points}}"
EVAL_GRIPPER_POINTS="${EVAL_GRIPPER_POINTS:-${eval_gripper_points}}"
EVAL_GRIPPER_TEMPLATE="${EVAL_GRIPPER_TEMPLATE:-${eval_gripper_template}}"
EVAL_ADD_GRIPPER_CLOUD="${EVAL_ADD_GRIPPER_CLOUD:-${eval_add_gripper_cloud}}"
EVAL_IMAGE_SIZE="${EVAL_IMAGE_SIZE:-${eval_image_size}}"
EVAL_GRIPPER_MODE="${EVAL_GRIPPER_MODE:-${eval_gripper_mode}}"
EVAL_GRIPPER_DELTA_THRESHOLD="${eval_gripper_delta_threshold}"
EVAL_GRIPPER_DELTA_ALIGNMENT="${eval_gripper_delta_alignment}"
EVAL_GRIPPER_OPEN_THRESHOLD="${EVAL_GRIPPER_OPEN_THRESHOLD:-${eval_gripper_open_threshold}}"
EVAL_GRIPPER_LOCK_AFTER_CLOSE="${EVAL_GRIPPER_LOCK_AFTER_CLOSE:-${eval_gripper_lock_after_close}}"
EVAL_WATER_PLANT_COLLISION="${eval_water_plant_collision}"
EVAL_WATER_DROP_COLLISION="${eval_water_drop_collision}"
EVAL_SAVE_VIDEO="${eval_save_video}"
EVAL_FAILURE_ARTIFACTS_ONLY="${eval_failure_artifacts_only}"
EVAL_VIDEO_WIDTH="${eval_video_width}"
EVAL_VIDEO_HEIGHT="${eval_video_height}"
EVAL_VIDEO_FPS="${eval_video_fps}"
EVAL_VIDEO_REFRESH_RGB="${eval_video_refresh_rgb}"
EVAL_SAVE_CONTROL_LOG="${eval_save_control_log}"
EVAL_LOG_CONTROL_DETAILS="${eval_log_control_details}"
EVAL_SAVE_ACTION_RECORDS="${eval_save_action_records}"
EVAL_SAVE_ACTION_CHUNKS="${eval_save_action_chunks}"
EVAL_SAVE_ACTION_VISUALIZATIONS="${eval_save_action_visualizations}"
EVAL_SAVE_FRAME_POINTCLOUDS="${eval_save_frame_pointclouds}"
EVAL_FRAME_POINTCLOUD_EVERY_N_FRAMES="${eval_frame_pointcloud_every_n_frames}"
EVAL_ACTION_VIS_EVERY_N_FRAMES="${eval_action_vis_every_n_frames}"
EVAL_ACTION_VIS_MAX_POINTS="${eval_action_vis_max_points}"
EVAL_ACTION_VIS_IMAGE_WIDTH="${eval_action_vis_image_width}"
EVAL_ACTION_VIS_POINT_MODE="${eval_action_vis_point_mode}"
EVAL_VISUALIZE_FOREGROUND="${eval_visualize_foreground}"
EVAL_DRAW_POUR_POINT="${eval_draw_pour_point}"
EVAL_DRAW_TASK_STAGES="${eval_draw_task_stages}"
EVAL_DRAW_PHONE_SUCCESS_SENSOR="${eval_draw_phone_success_sensor}"
EVAL_VERBOSE_TERMINAL="${eval_verbose_terminal}"
EVAL_SCRIPT="${TOOLS_DIR}/official_eval.py"
ORIGINAL_ARGS=("$@")
# Multi-task evaluations run up to five tasks concurrently by default. The
# worker count is capped below by the number of requested tasks.
REQUESTED_EVAL_ROOT="${EVAL_ROOT}"
REQUESTED_EVAL_WORKERS="${EVAL_WORKERS:-5}"
# Reuse the display supplied by the caller. The previous fast evaluation used
# DISPLAY=:99; forcing the stale :105 server can leave RLBench OpenGL cameras
# with an incomplete framebuffer and produce all-zero RGB frames.
if ! [[ "${REQUESTED_EVAL_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_WORKERS must be a positive integer" >&2
    exit 2
fi

# --tasks followed by one or more task names still overrides TASK_LIST.
# Comma-separated names also work. Other arguments are forwarded to every task.
TASKS=()
EVAL_ARGS=()
# Planning overrides keep collision checking disabled by default because
# several benchmark tasks require deliberate contact with task geometry.
# Pass --collision-checking explicitly for collision-free diagnostics.
if [[ "${EVAL_COLLISION_CHECKING}" == "1" ]]; then
    COLLISION_ARGS=(--collision-checking)
else
    COLLISION_ARGS=(--no-collision-checking)
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose-terminal)
            eval_verbose_terminal=1
            shift
            ;;
        --no-verbose-terminal)
            eval_verbose_terminal=0
            shift
            ;;
        --episodes)
            EVAL_ARGS+=("$1")
            shift
            [[ $# -gt 0 ]] || { echo "--episodes requires a value" >&2; exit 2; }
            EVAL_EPISODES="$1"
            EVAL_ARGS+=("$1")
            shift
            ;;
        --episodes=*)
            EVAL_EPISODES="${1#*=}"
            EVAL_ARGS+=("$1")
            shift
            ;;
        --tasks|--task)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                IFS=',' read -r -a TASK_PARTS <<< "$1"
                for TASK_PART in "${TASK_PARTS[@]}"; do
                    [[ -n "$TASK_PART" ]] && TASKS+=("$TASK_PART")
                done
                shift
            done
            ;;
        --tasks=*|--task=*)
            IFS=',' read -r -a TASK_PARTS <<< "${1#*=}"
            for TASK_PART in "${TASK_PARTS[@]}"; do
                [[ -n "$TASK_PART" ]] && TASKS+=("$TASK_PART")
            done
            shift
            ;;
        --collision-checking|--no-collision-checking)
            COLLISION_ARGS=("$1")
            eval_collision_checking_explicit=1
            shift
            ;;
        --controller-error-retries)
            EVAL_ARGS+=("$1")
            shift
            [[ $# -gt 0 ]] || { echo "$1 requires a value" >&2; exit 2; }
            eval_controller_error_retries="$1"
            shift
            EVAL_ARGS+=("${eval_controller_error_retries}")
            ;;
        --controller-error-retries=*)
            eval_controller_error_retries="${1#*=}"
            EVAL_ARGS+=("$1")
            shift
            ;;
        --controller-error-retry-timeout-seconds)
            EVAL_ARGS+=("$1")
            shift
            [[ $# -gt 0 ]] || { echo "$1 requires a value" >&2; exit 2; }
            eval_controller_error_retry_timeout_seconds="$1"
            shift
            EVAL_ARGS+=("${eval_controller_error_retry_timeout_seconds}")
            ;;
        --controller-error-retry-timeout-seconds=*)
            eval_controller_error_retry_timeout_seconds="${1#*=}"
            EVAL_ARGS+=("$1")
            shift
            ;;
        --controller-error-mode)
            EVAL_ARGS+=("$1")
            shift
            [[ $# -gt 0 ]] || { echo "$1 requires a value" >&2; exit 2; }
            eval_controller_error_mode="$1"
            shift
            EVAL_ARGS+=("${eval_controller_error_mode}")
            ;;
        --controller-error-mode=*)
            eval_controller_error_mode="${1#*=}"
            EVAL_ARGS+=("$1")
            shift
            ;;
        --controller-continue-max-errors)
            EVAL_ARGS+=("$1")
            shift
            [[ $# -gt 0 ]] || { echo "$1 requires a value" >&2; exit 2; }
            eval_controller_continue_max_errors="$1"
            shift
            EVAL_ARGS+=("${eval_controller_continue_max_errors}")
            ;;
        --controller-continue-max-errors=*)
            eval_controller_continue_max_errors="${1#*=}"
            EVAL_ARGS+=("$1")
            shift
            ;;
        --eval-workers)
            shift
            REQUESTED_EVAL_WORKERS="$1"
            shift
            ;;
        --eval-workers=*)
            REQUESTED_EVAL_WORKERS="${1#*=}"
            shift
            ;;
        --eval-gpu-ids)
            shift
            EVAL_GPU_IDS="$1"
            shift
            ;;
        --eval-gpu-ids=*)
            EVAL_GPU_IDS="${1#*=}"
            shift
            ;;
        --allow-gpu-sharing)
            shift
            ;;
        --no-gpu-sharing)
            shift
            ;;
        --eval-display-base)
            shift
            EVAL_DISPLAY_BASE="$1"
            shift
            ;;
        --eval-display-base=*)
            EVAL_DISPLAY_BASE="${1#*=}"
            shift
            ;;
        *)
            EVAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# The policy can be supplied either through the environment or as a forwarded
# CLI argument. Resolve the latter before constructing the fixed argument list.
for ((arg_index = 0; arg_index < ${#EVAL_ARGS[@]}; arg_index++)); do
    case "${EVAL_ARGS[arg_index]}" in
        --policy-path)
            EVAL_POLICY_PATH="${EVAL_ARGS[arg_index + 1]}"
            ;;
        --policy-path=*)
            EVAL_POLICY_PATH="${EVAL_ARGS[arg_index]#*=}"
            ;;
    esac
done
if [[ -z "${EVAL_POLICY_PATH}" ]]; then
    echo "A policy is required. Set EVAL_POLICY_PATH or pass --policy-path." >&2
    exit 2
fi

configure_coppeliasim
PYTHON="${python_bin}"
COPPELIASIM_ROOT="${coppeliasim_root}"
QT_QPA_PLATFORM_PLUGIN_PATH="${qt_platform_path}"
QT_PLUGIN_PATH="${qt_plugin_path}"
QT_XCB_GL_INTEGRATION="${qt_xcb_gl_integration}"
__GLX_VENDOR_LIBRARY_NAME="${glx_vendor_library}"
EVAL_RLBENCH_HEADLESS="${eval_rlbench_headless}"
export EVAL_RLBENCH_HEADLESS

if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=("${TASK_LIST[@]}")
fi

# Use the task preset in single-task output names as well. This is only a
# label; the Python task config remains the source of truth for final values.
ROOT_LABEL_EPISODES="${EVAL_EPISODES}"
ROOT_LABEL_POLICY_PATH="${EVAL_POLICY_PATH}"
if (( ${#TASKS[@]} == 1 )); then
    get_task_preset "${TASKS[0]}"
    for ((preset_index = 0; preset_index < ${#TASK_PRESET_ARGS[@]}; preset_index++)); do
        case "${TASK_PRESET_ARGS[preset_index]}" in
            --episodes)
                ROOT_LABEL_EPISODES="${TASK_PRESET_ARGS[preset_index + 1]}"
                ;;
            --episodes=*)
                ROOT_LABEL_EPISODES="${TASK_PRESET_ARGS[preset_index]#*=}"
                ;;
            --policy-path)
                ROOT_LABEL_POLICY_PATH="${TASK_PRESET_ARGS[preset_index + 1]}"
                ;;
            --policy-path=*)
                ROOT_LABEL_POLICY_PATH="${TASK_PRESET_ARGS[preset_index]#*=}"
                ;;
        esac
    done
fi
for ((arg_index = 0; arg_index < ${#EVAL_ARGS[@]}; arg_index++)); do
    case "${EVAL_ARGS[arg_index]}" in
        --episodes)
            ROOT_LABEL_EPISODES="${EVAL_ARGS[arg_index + 1]}"
            ;;
        --episodes=*)
            ROOT_LABEL_EPISODES="${EVAL_ARGS[arg_index]#*=}"
            ;;
        --policy-path)
            ROOT_LABEL_POLICY_PATH="${EVAL_ARGS[arg_index + 1]}"
            ;;
        --policy-path=*)
            ROOT_LABEL_POLICY_PATH="${EVAL_ARGS[arg_index]#*=}"
            ;;
    esac
done

if ! [[ "${REQUESTED_EVAL_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_WORKERS must be a positive integer" >&2
    exit 2
fi
if [[ -z "${EVAL_DISPLAY_BASE}" ]]; then
    if [[ "${DISPLAY:-}" =~ ^:([0-9]+)(\.[0-9]+)?$ ]]; then
        EVAL_DISPLAY_BASE="${BASH_REMATCH[1]}"
    else
        EVAL_DISPLAY_BASE="99"
    fi
fi
if ! [[ "${EVAL_DISPLAY_BASE}" =~ ^[0-9]+$ ]]; then
    echo "EVAL_DISPLAY_BASE must be a display number" >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${EVAL_GPU_IDS}"
if [[ -z "${EVAL_GPU_IDS}" || ${#GPU_IDS[@]} == 0 ]]; then
    echo "EVAL_GPU_IDS must contain at least one GPU id" >&2
    exit 2
fi
for gpu_id in "${GPU_IDS[@]}"; do
    if [[ -z "${gpu_id}" ]]; then
        echo "EVAL_GPU_IDS contains an empty GPU id: ${EVAL_GPU_IDS}" >&2
        exit 2
    fi
done
SERIAL_GPU_ID="${GPU_IDS[0]}"
if (( REQUESTED_EVAL_WORKERS > ${#TASKS[@]} )); then
    REQUESTED_EVAL_WORKERS="${#TASKS[@]}"
fi
SERIAL_WORKER_COUNT="${REQUESTED_EVAL_WORKERS}"
if (( REQUESTED_EVAL_WORKERS > 1 )); then
    echo "[eval-parallel] tasks=${#TASKS[@]} workers=${SERIAL_WORKER_COUNT} gpu=${SERIAL_GPU_ID} display_base=:${EVAL_DISPLAY_BASE}"
else
    echo "[eval-serial] tasks=${#TASKS[@]} gpu=${SERIAL_GPU_ID} display=:${EVAL_DISPLAY_BASE}"
fi

EVAL_FIXED_ARGS=(
    --policy-path "${EVAL_POLICY_PATH}"
    --episodes "${EVAL_EPISODES}"
    --variation "${EVAL_VARIATION}"
    --seed "${EVAL_SEED}"
    --max-model-calls "${EVAL_MAX_MODEL_CALLS}"
    --controller-error-retries "${eval_controller_error_retries}"
    --controller-error-retry-timeout-seconds "${eval_controller_error_retry_timeout_seconds}"
    --controller-error-mode "${eval_controller_error_mode}"
    --controller-continue-max-errors "${eval_controller_continue_max_errors}"
    --action-index "${EVAL_ACTION_INDEX}"
    --exec-action-steps "${EVAL_EXEC_ACTION_STEPS}"
    --simulation-timestep "${EVAL_SIMULATION_TIMESTEP}"
    --arm-action-mode "${EVAL_ARM_ACTION_MODE}"
    --execution-mode "${EVAL_EXECUTION_MODE}"
    --mover-max-tries "${EVAL_MOVER_MAX_TRIES}"
    --mover-position-tolerance "${EVAL_MOVER_POSITION_TOLERANCE}"
    --mover-rotation-tolerance "${EVAL_MOVER_ROTATION_TOLERANCE}"
    --mover-gripper-position-tolerance "${EVAL_MOVER_GRIPPER_POSITION_TOLERANCE}"
    --mover-gripper-rotation-tolerance "${EVAL_MOVER_GRIPPER_ROTATION_TOLERANCE}"
    --num-points "${EVAL_NUM_POINTS}"
    --gripper-points "${EVAL_GRIPPER_POINTS}"
    --gripper-len "${eval_gripper_len}"
    --gripper-template "${EVAL_GRIPPER_TEMPLATE}"
    --gripper-delta-threshold "${EVAL_GRIPPER_DELTA_THRESHOLD}"
    --gripper-delta-open-threshold "${eval_gripper_delta_open_threshold}"
    --gripper-delta-close-threshold "${eval_gripper_delta_close_threshold}"
    --gripper-delta-alignment "${EVAL_GRIPPER_DELTA_ALIGNMENT}"
    --gripper-mode "${EVAL_GRIPPER_MODE}"
    --gripper-open-threshold "${EVAL_GRIPPER_OPEN_THRESHOLD}"
    --water-plant-collision "${EVAL_WATER_PLANT_COLLISION}"
    --water-drop-collision "${EVAL_WATER_DROP_COLLISION}"
    --max-eef-position-step "${EVAL_MAX_EEF_POSITION_STEP}"
    --max-eef-rotation-step "${EVAL_MAX_EEF_ROTATION_STEP}"
    --jacobian-position-threshold "${EVAL_JACOBIAN_POSITION_THRESHOLD}"
    --jacobian-rotation-threshold "${EVAL_JACOBIAN_ROTATION_THRESHOLD}"
    --jacobian-midpoint-max-depth "${EVAL_JACOBIAN_MIDPOINT_MAX_DEPTH}"
    --waypoint-position-tolerance "${EVAL_WAYPOINT_POSITION_TOLERANCE}"
    --waypoint-rotation-tolerance "${EVAL_WAYPOINT_ROTATION_TOLERANCE}"
    --waypoint-max-control-steps "${EVAL_WAYPOINT_MAX_CONTROL_STEPS}"
    --joint-velocity-kp "${EVAL_JOINT_VELOCITY_KP}"
    --joint-velocity-max-speed "${EVAL_JOINT_VELOCITY_MAX_SPEED}"
    --joint-velocity-joint-tolerance "${EVAL_JOINT_VELOCITY_JOINT_TOLERANCE}"
    --joint-velocity-stall-steps "${EVAL_JOINT_VELOCITY_STALL_STEPS}"
    --franka-ik-max-iterations "${EVAL_FRANKA_IK_MAX_ITERATIONS:-300}"
    --franka-ik-tolerance "${EVAL_FRANKA_IK_TOLERANCE:-0.01}"
    --franka-ik-damping "${EVAL_FRANKA_IK_DAMPING:-0.0001}"
    --device "${EVAL_DEVICE}"
    --image-size "${EVAL_IMAGE_SIZE}"
    --video-fps "${EVAL_VIDEO_FPS}"
    --video-width "${EVAL_VIDEO_WIDTH}"
    --video-height "${EVAL_VIDEO_HEIGHT}"
    --action-vis-every-n-frames "${EVAL_ACTION_VIS_EVERY_N_FRAMES}"
    --frame-pointcloud-every-n-frames "${EVAL_FRAME_POINTCLOUD_EVERY_N_FRAMES}"
    --action-vis-max-points "${EVAL_ACTION_VIS_MAX_POINTS}"
    --action-vis-image-width "${EVAL_ACTION_VIS_IMAGE_WIDTH}"
    --action-vis-point-mode "${EVAL_ACTION_VIS_POINT_MODE}"
)
EVAL_FIXED_ARGS+=(--sync-virtual-gripper-tcp --simulator-robustness-optimizations-song)
case "${EVAL_CLIP_WITHIN_WORKSPACE}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--clip-within-workspace) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-clip-within-workspace) ;;
    *) echo "EVAL_CLIP_WITHIN_WORKSPACE must be 0 or 1" >&2; exit 2 ;;
esac
case "${EVAL_REINFER_ON_CONTROL_ERROR}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--reinfer-on-control-error) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-reinfer-on-control-error) ;;
    *) echo "EVAL_REINFER_ON_CONTROL_ERROR must be 0 or 1" >&2; exit 2 ;;
esac
case "${EVAL_GRIPPER_AFTER_REACH}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--gripper-after-reach) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-gripper-after-reach) ;;
    *) echo "EVAL_GRIPPER_AFTER_REACH must be 0 or 1" >&2; exit 2 ;;
esac
case "${EVAL_POINTACT_PYREP_COMPAT}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--pointact-pyrep-compat) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-pointact-pyrep-compat) ;;
    *) echo "EVAL_POINTACT_PYREP_COMPAT must be 0 or 1" >&2; exit 2 ;;
esac
case "${EVAL_ADD_GRIPPER_CLOUD}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--add-gripper-cloud) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-add-gripper-cloud) ;;
    *) echo "EVAL_ADD_GRIPPER_CLOUD must be 0 or 1" >&2; exit 2 ;;
esac
case "${EVAL_GRIPPER_LOCK_AFTER_CLOSE}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--gripper-lock-after-close) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-gripper-lock-after-close) ;;
    *) echo "EVAL_GRIPPER_LOCK_AFTER_CLOSE must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${EVAL_MODEL_NOISE_SEED}" ]]; then
    EVAL_FIXED_ARGS+=(--model-noise-seed "${EVAL_MODEL_NOISE_SEED}")
fi
if [[ -n "${EVAL_POINTSEG_DEVICE}" ]]; then
    EVAL_FIXED_ARGS+=(--pointseg-device "${EVAL_POINTSEG_DEVICE}")
fi
if [[ -n "${EVAL_DATASET_ROOT}" ]]; then
    EVAL_FIXED_ARGS+=(--dataset-root "${EVAL_DATASET_ROOT}")
fi
if [[ -n "${EVAL_DATASET_EPISODES}" ]]; then
    EVAL_FIXED_ARGS+=(--dataset-episodes "${EVAL_DATASET_EPISODES}")
fi
[[ "${EVAL_SAVE_VIDEO}" == "1" ]] && EVAL_FIXED_ARGS+=(--save-video)
case "${EVAL_FAILURE_ARTIFACTS_ONLY}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--failure-artifacts-only) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-failure-artifacts-only) ;;
    *) echo "EVAL_FAILURE_ARTIFACTS_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
[[ "${EVAL_VIDEO_REFRESH_RGB}" == "1" ]] && EVAL_FIXED_ARGS+=(--video-refresh-rgb)
[[ "${EVAL_VIDEO_REFRESH_RGB}" == "0" ]] && EVAL_FIXED_ARGS+=(--no-video-refresh-rgb)
[[ "${EVAL_SAVE_CONTROL_LOG}" == "1" ]] && EVAL_FIXED_ARGS+=(--save-control-log)
[[ "${EVAL_LOG_CONTROL_DETAILS}" == "1" ]] && EVAL_FIXED_ARGS+=(--log-control-details)
[[ "${EVAL_SAVE_ACTION_RECORDS}" == "1" ]] && EVAL_FIXED_ARGS+=(--save-action-records)
[[ "${EVAL_SAVE_ACTION_RECORDS}" == "0" ]] && EVAL_FIXED_ARGS+=(--no-save-action-records)
[[ "${EVAL_SAVE_ACTION_CHUNKS}" == "1" ]] && EVAL_FIXED_ARGS+=(--save-action-chunks)
[[ "${EVAL_SAVE_ACTION_CHUNKS}" == "0" ]] && EVAL_FIXED_ARGS+=(--no-save-action-chunks)
[[ "${EVAL_SAVE_ACTION_VISUALIZATIONS}" == "1" ]] && EVAL_FIXED_ARGS+=(--save-action-visualizations)
[[ "${EVAL_SAVE_FRAME_POINTCLOUDS}" == "1" ]] && EVAL_FIXED_ARGS+=(--save-frame-pointclouds)
[[ "${EVAL_VISUALIZE_FOREGROUND}" == "1" ]] && EVAL_FIXED_ARGS+=(--visualize-foreground)
case "${EVAL_DRAW_POUR_POINT}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--draw-pour-point) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-draw-pour-point) ;;
    auto|AUTO|Auto) ;;
    *) echo "EVAL_DRAW_POUR_POINT must be auto, 0, or 1" >&2; exit 2 ;;
esac
case "${EVAL_DRAW_TASK_STAGES}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--draw-task-stages) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-draw-task-stages) ;;
    auto|AUTO|Auto) ;;
    *) echo "EVAL_DRAW_TASK_STAGES must be auto, 0, or 1" >&2; exit 2 ;;
esac
case "${EVAL_DRAW_PHONE_SUCCESS_SENSOR}" in
    1|true|True|TRUE) EVAL_FIXED_ARGS+=(--draw-phone-success-sensor) ;;
    0|false|False|FALSE) EVAL_FIXED_ARGS+=(--no-draw-phone-success-sensor) ;;
    auto|AUTO|Auto) ;;
    *) echo "EVAL_DRAW_PHONE_SUCCESS_SENSOR must be auto, 0, or 1" >&2; exit 2 ;;
esac

# Each parallel task gets its own display. Existing reachable displays are
# reused; otherwise Xvfb/X is started for every task slot.
NEED_X_START=false
display_number="${EVAL_DISPLAY_BASE}"
display=":${display_number}"
display_count="${SERIAL_WORKER_COUNT}"
if (( display_count < 1 )); then display_count=1; fi
if [[ "${EVAL_USE_XVFB:-0}" == "1" ]] && ! command -v Xvfb >/dev/null 2>&1; then
    echo "Cannot start Xvfb: command Xvfb was not found" >&2
    exit 2
fi
for display_offset in $(seq 0 $((display_count - 1))); do
    display_number=$((EVAL_DISPLAY_BASE + display_offset))
    display=":${display_number}"
    if DISPLAY="${display}" xdpyinfo >/dev/null 2>&1; then
        continue
    fi
    socket="/tmp/.X11-unix/X${display_number}"
    if [[ -e "${socket}" || -e "/tmp/.X${display_number}-lock" ]]; then
        if [[ "${EVAL_USE_XVFB:-0}" == "1" ]]; then
            echo "Display ${display} has a stale X socket or lock" >&2
            exit 2
        fi
        echo "${display} exists but is not reachable: ${socket}" >&2
        exit 2
    fi
    NEED_X_START=true
    if [[ "${EVAL_USE_XVFB:-0}" == "1" ]]; then
        Xvfb "${display}" -screen 0 1280x1024x24 -nolisten tcp -noreset \
            -ac +extension GLX +render \
            </dev/null >"/tmp/rlbench_eval_xvfb${display_number}.log" 2>&1 &
    else
        if ! command -v X >/dev/null 2>&1; then
            echo "Cannot start X displays: command X was not found" >&2
            exit 2
        fi
        sudo -v
        sudo nohup X "${display}" -nolisten tcp -noreset -ac \
            </dev/null >"/tmp/rlbench_eval_x${display_number}.log" 2>&1 &
    fi
done
if [[ "${NEED_X_START}" == "true" ]]; then sleep 3; fi
for display_offset in $(seq 0 $((display_count - 1))); do
    display_number=$((EVAL_DISPLAY_BASE + display_offset))
    DISPLAY=":${display_number}" xdpyinfo >/dev/null || {
        echo "X display :${display_number} is not reachable" >&2
        exit 2
    }
done

# One outer directory represents one complete multi-task evaluation. Each
# serial task receives its own child directory through --run-dir.
checkpoint_name="$(basename "$(dirname "${ROOT_LABEL_POLICY_PATH}")")"
if [[ "${checkpoint_name}" =~ ^0*[0-9]+$ ]]; then
    checkpoint_suffix="$((10#${checkpoint_name}))"
else
    checkpoint_suffix="${checkpoint_name}"
fi
if [[ -n "${REQUESTED_EVAL_ROOT}" ]]; then
    EVAL_ROOT="${REQUESTED_EVAL_ROOT}"
    if [[ -e "${EVAL_ROOT}" ]]; then
        echo "Evaluation root already exists: ${EVAL_ROOT}" >&2
        exit 2
    fi
else
    eval_stamp="$(date +%Y%m%d_%H%M%S)"
    if [[ "${EVAL_RUN_NAMING_STYLE:-legacy}" == "canonical_v1" ]]; then
        if (( ${#TASKS[@]} == 1 )); then
            eval_task_field="${TASKS[0]}"
            eval_episode_field="${ROOT_LABEL_EPISODES}eps"
        else
            eval_task_field="${#TASKS[@]}tasks"
            eval_episode_field="${EVAL_EPISODES}eps"
        fi
        eval_run_name="${eval_stamp}__${EVAL_TRAINING_SETTING:-unknown_training}__ckpt-${EVAL_CHECKPOINT_STEP:-${checkpoint_suffix}}__${eval_task_field}__${eval_episode_field}"
        EVAL_ROOT="${EVAL_BASE_DIR}/${eval_run_name}"
    elif (( ${#TASKS[@]} == 1 )); then
        EVAL_ROOT="${EVAL_BASE_DIR}/eval_${eval_stamp}_1tasks_${TASKS[0]}_${ROOT_LABEL_EPISODES}eps_${checkpoint_suffix}"
    else
        EVAL_ROOT="${EVAL_BASE_DIR}/eval_${eval_stamp}_${#TASKS[@]}tasks_${EVAL_EPISODES}eps_${checkpoint_suffix}"
    fi
    suffix=1
    while [[ -e "${EVAL_ROOT}" ]]; do
        if [[ "${EVAL_RUN_NAMING_STYLE:-legacy}" == "canonical_v1" ]]; then
            EVAL_ROOT="${EVAL_BASE_DIR}/${eval_run_name}__${suffix}"
        elif (( ${#TASKS[@]} == 1 )); then
            EVAL_ROOT="${EVAL_BASE_DIR}/eval_${eval_stamp}_1tasks_${TASKS[0]}_${ROOT_LABEL_EPISODES}eps_${checkpoint_suffix}_${suffix}"
        else
            EVAL_ROOT="${EVAL_BASE_DIR}/eval_${eval_stamp}_${#TASKS[@]}tasks_${EVAL_EPISODES}eps_${checkpoint_suffix}_${suffix}"
        fi
        suffix=$((suffix + 1))
    done
fi
mkdir -p "${EVAL_ROOT}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${EVAL_ROOT}/logs}"
mkdir -p "${EVAL_LOG_DIR}"

printf 'EVAL_RLBENCH_HEADLESS=%q DISPLAY=%q QT_QPA_PLATFORM=%q QT_QPA_PLATFORM_PLUGIN_PATH=%q QT_PLUGIN_PATH=%q QT_XCB_GL_INTEGRATION=%q __GLX_VENDOR_LIBRARY_NAME=%q QT_X11_NO_MITSHM=%q CUDA_VISIBLE_DEVICES=%q COPPELIASIM_ROOT=%q RLBENCH_PLANNER_MAX_TIME_MS=%q EVAL_CHECKPOINT_TASK_REGISTRY=%q EVAL_CHECKPOINT_ID=%q EVAL_TRAINING_SETTING=%q EVAL_CHECKPOINT_STEP=%q bash %q ' \
    "${EVAL_RLBENCH_HEADLESS}" \
    "${DISPLAY:-}" "${QT_QPA_PLATFORM:-xcb}" "${QT_QPA_PLATFORM_PLUGIN_PATH}" "${QT_PLUGIN_PATH}" \
    "${QT_XCB_GL_INTEGRATION}" "${__GLX_VENDOR_LIBRARY_NAME}" "${QT_X11_NO_MITSHM:-1}" \
    "${CUDA_VISIBLE_DEVICES:-${SERIAL_GPU_ID}}" "${COPPELIASIM_ROOT}" \
    "${EVAL_PLANNER_MAX_TIME_MS}" "${EVAL_CHECKPOINT_TASK_REGISTRY:-}" \
    "${EVAL_CHECKPOINT_ID:-}" "${EVAL_TRAINING_SETTING:-}" \
    "${EVAL_CHECKPOINT_STEP:-}" "$0" > "${EVAL_ROOT}/command.txt"
printf '%q ' "${ORIGINAL_ARGS[@]}" >> "${EVAL_ROOT}/command.txt"
printf '\n' >> "${EVAL_ROOT}/command.txt"
{
    printf 'RLBench multi-task evaluation\n'
    printf 'run_root=%s\n' "${EVAL_ROOT}"
    printf 'tasks=%s\n' "${TASKS[*]}"
    printf 'checkpoint_id=%s\n' "${EVAL_CHECKPOINT_ID:-legacy-unregistered}"
    printf 'training_setting=%s\n' "${EVAL_TRAINING_SETTING:-legacy-unknown}"
    printf 'checkpoint_step=%s\n' "${EVAL_CHECKPOINT_STEP:-${checkpoint_suffix:-unknown}}"
    printf 'checkpoint_task_registry=%s\n' "${EVAL_CHECKPOINT_TASK_REGISTRY:-none}"
    printf 'checkpoint_and_effective_parameters=see each task/config.json\n'
    printf 'command=see command.txt\n'
} > "${EVAL_ROOT}/README.txt"

"${PYTHON}" - "${EVAL_ROOT}" "${EVAL_LOG_DIR}" "${EVAL_DISPLAY_BASE}" "${SERIAL_WORKER_COUNT}" "${REQUESTED_EVAL_WORKERS}" "${EVAL_GPU_IDS}" \
    "${COPPELIASIM_ROOT}" "${PYTHON}" "${DISPLAY:-}" "${QT_QPA_PLATFORM:-xcb}" \
    "${QT_QPA_PLATFORM_PLUGIN_PATH}" "${QT_PLUGIN_PATH}" "${QT_XCB_GL_INTEGRATION}" \
    "${__GLX_VENDOR_LIBRARY_NAME}" "${QT_X11_NO_MITSHM:-1}" "${EVAL_RLBENCH_HEADLESS}" \
"${SERIAL_GPU_ID}" "${EVAL_PLANNER_MAX_TIME_MS}" "${EVAL_WATER_PLANT_COLLISION}" "${EVAL_POLICY_PATH}" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    root,
    log_dir,
    display_base,
    workers,
    requested_workers,
    gpu_ids,
    coppeliasim_root,
    python_path,
    display,
    qt_platform,
    qt_plugin_path,
    qt_plugin_root,
    qt_xcb_gl_integration,
    glx_vendor_library,
    qt_no_mitshm,
    rlbench_headless,
    cuda_visible_devices,
    planner_max_time_ms,
    water_plant_collision,
    policy_path,
    *tasks,
) = sys.argv[1:]
resolved_policy_path = Path(policy_path).expanduser().resolve()
checkpoint_path = resolved_policy_path.parent
command_file = Path(root).resolve() / "command.txt"
invocation_command = command_file.read_text(encoding="utf-8").strip()
config = {
    "run_root": str(Path(root).resolve()),
    "tasks": tasks,
    "policy_path": str(resolved_policy_path),
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_name": checkpoint_path.name,
    "eval_workers": int(workers),
    "requested_eval_workers": int(requested_workers),
    "parallel_evaluation": int(workers) > 1,
    "eval_display_base": int(display_base),
    "eval_gpu_ids": gpu_ids,
    "success_log": str(Path(root).resolve() / "successlog"),
    "log_dir": str(Path(log_dir).resolve()),
    "planner_max_time_ms": planner_max_time_ms,
    "water_plant_collision": water_plant_collision,
    "environment": {
        "display": display,
        "qt_qpa_platform": qt_platform,
        "qt_qpa_platform_plugin_path": qt_plugin_path,
        "qt_plugin_path": qt_plugin_root,
        "qt_xcb_gl_integration": qt_xcb_gl_integration,
        "glx_vendor_library": glx_vendor_library,
        "qt_x11_no_mitshm": qt_no_mitshm,
        "rlbench_headless": rlbench_headless,
        "cuda_visible_devices": cuda_visible_devices,
        "coppeliasim_root": coppeliasim_root,
        "python": python_path,
    },
    "command_file": str(command_file),
    "bash_invocation": invocation_command,
    "task_config_files": {
        task: str(Path(root).resolve() / task / "config.json") for task in tasks
    },
    "task_preset_files": {
        task: str(Path(root).resolve() / "task_presets" / f"{task}.txt") for task in tasks
    },
    # This file is finalized after the task configs are written.  Do not use
    # the shell defaults above as the effective evaluation parameters.
    "effective_parameters": None,
}
with open(Path(root) / "eval_config.json", "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, ensure_ascii=True)
PY

# absolute_width
run_eval_task() {
    local task="$1"
    local task_index="${2:-0}"
    local worker_slot="${3:-${task_index}}"
    local display_number=$((EVAL_DISPLAY_BASE + worker_slot))
    local gpu_id="${SERIAL_GPU_ID}"
    local log_file="${EVAL_LOG_DIR}/${task}.log"
    local preset_file="${EVAL_ROOT}/task_presets/${task}.txt"

    get_task_preset "${task}"
    get_task_collision_args "${task}"
    get_task_planner_max_time_ms
    mkdir -p "${EVAL_ROOT}/task_presets"
    printf '%q ' "${TASK_PRESET_ARGS[@]}" "${TASK_COLLISION_ARGS[@]}" > "${preset_file}"
    printf '\n' >> "${preset_file}"

    local eval_mode="serial"
    if (( SERIAL_WORKER_COUNT > 1 )); then eval_mode="parallel"; fi
    echo "[eval-start] task=${task} mode=${eval_mode} display=:${display_number} gpu=${gpu_id} planner_max_time_ms=${TASK_PLANNER_MAX_TIME_MS} preset_args=${#TASK_PRESET_ARGS[@]} log=${log_file}"
    (
        set -o pipefail
        export DISPLAY=":${display_number}"
        export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
        export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
        export QT_PLUGIN_PATH="${QT_PLUGIN_PATH}"
        export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION}"
        export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME}"
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export EVAL_WORKER_SLOT="${worker_slot}" EVAL_WORKERS="${SERIAL_WORKER_COUNT}"
        export EVAL_DISPLAY_BASE="${EVAL_DISPLAY_BASE}" EVAL_GPU_ID="${gpu_id}"
        export EVAL_RLBENCH_HEADLESS="${EVAL_RLBENCH_HEADLESS}"
        export EVAL_LAUNCH_COMMAND_FILE="${EVAL_ROOT}/command.txt"
        export RLBENCH_PLANNER_MAX_TIME_MS="${TASK_PLANNER_MAX_TIME_MS}"
        export COPPELIASIM_ROOT="${COPPELIASIM_ROOT}"
        export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
        export PATH="$(dirname "${PYTHON}"):${PATH}"
        export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH}"
        if [[ "${eval_verbose_terminal}" == "1" ]]; then
            "${PYTHON}" "${EVAL_SCRIPT}" \
                --task "$task" \
                --run-dir "${EVAL_ROOT}/${task}" \
                "${EVAL_FIXED_ARGS[@]}" \
                "${TASK_COLLISION_ARGS[@]}" \
                "${TASK_PRESET_ARGS[@]}" \
                "${EVAL_ARGS[@]}" 2>&1 | tee "${log_file}"
            eval_status="${PIPESTATUS[0]}"
        else
            "${PYTHON}" "${EVAL_SCRIPT}" \
                --task "$task" \
                --run-dir "${EVAL_ROOT}/${task}" \
                "${EVAL_FIXED_ARGS[@]}" \
                "${TASK_COLLISION_ARGS[@]}" \
                "${TASK_PRESET_ARGS[@]}" \
                "${EVAL_ARGS[@]}" 2>&1 | tee "${log_file}" | \
                awk '/^\[eval-trajectory\]/{printf "\r%s", $0; fflush(); next} /^\[task-summary\]|^\[done\]/{printf "\n%s\n", $0; fflush(); next}'
            eval_status="${PIPESTATUS[0]}"
        fi
        return "${eval_status}"
    )
}

FAILED=0
COMPLETED_TASKS=0
PREVIOUS_TASK_RESULTS=()

print_eval_progress() {
    local current_tasks="$1"
    local previous="none"
    if (( ${#PREVIOUS_TASK_RESULTS[@]} > 0 )); then
        local separator=""
        previous=""
        for result in "${PREVIOUS_TASK_RESULTS[@]}"; do
            previous+="${separator}${result}"
            separator=";"
        done
    fi
    echo "[eval-progress] completed_tasks=${COMPLETED_TASKS}/${#TASKS[@]} current_task=${current_tasks} previous_task_results=${previous}"
}

record_task_result() {
    local task="$1"
    local status="$2"
    local log_file="${EVAL_LOG_DIR}/${task}.log"
    local summary_line=""
    if [[ -f "${log_file}" ]]; then
        summary_line="$(rg '^\[task-summary\]' "${log_file}" | tail -1 || true)"
    fi
    if [[ -n "${summary_line}" ]]; then
        echo "[eval-task-done] task=${task} status=${status} ${summary_line#\[task-summary\] }"
        PREVIOUS_TASK_RESULTS+=("${task}:${summary_line#\[task-summary\] task=${task} }")
    else
        echo "[eval-task-done] task=${task} status=${status} summary=unavailable"
        PREVIOUS_TASK_RESULTS+=("${task}:status=${status}")
    fi
    COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
}

declare -a TASK_PIDS=()
if (( SERIAL_WORKER_COUNT > 1 )); then
    # Keep model loading and evaluation inside the requested concurrency
    # bound. Launching every task up front defeats --eval-workers because all
    # policies allocate GPU memory before any wait occurs.
    declare -A PID_TO_TASK=()
    declare -A PID_TO_SLOT=()
    declare -A SLOT_TO_PID=()
    next_task_index=0
    active_tasks=0
    while (( next_task_index < ${#TASKS[@]} || active_tasks > 0 )); do
        while (( active_tasks < SERIAL_WORKER_COUNT && next_task_index < ${#TASKS[@]} )); do
            task="${TASKS[$next_task_index]}"
            worker_slot=0
            while [[ -n "${SLOT_TO_PID[$worker_slot]+x}" ]]; do
                worker_slot=$((worker_slot + 1))
            done
            echo "[eval-dispatch] task=${task} task_index=${next_task_index} mode=parallel"
            run_eval_task "${task}" "${next_task_index}" "${worker_slot}" &
            task_pid=$!
            TASK_PIDS[${next_task_index}]="${task_pid}"
            PID_TO_TASK[${task_pid}]="${task}"
            PID_TO_SLOT[${task_pid}]="${worker_slot}"
            SLOT_TO_PID[${worker_slot}]="${task_pid}"
            next_task_index=$((next_task_index + 1))
            active_tasks=$((active_tasks + 1))
        done

        finished_pid=""
        wait -n -p finished_pid "${!PID_TO_TASK[@]}"
        status=$?
        task="${PID_TO_TASK[$finished_pid]}"
        if [[ "${status}" == "0" ]]; then
            record_task_result "${task}" "0"
        else
            record_task_result "${task}" "failed"
            FAILED=$((FAILED + 1))
        fi
        unset 'PID_TO_TASK[$finished_pid]'
        worker_slot="${PID_TO_SLOT[$finished_pid]}"
        unset 'PID_TO_SLOT[$finished_pid]'
        unset 'SLOT_TO_PID[$worker_slot]'
        active_tasks=$((active_tasks - 1))
        print_eval_progress "${active_tasks}_parallel_tasks"
    done
else
    for task_index in "${!TASKS[@]}"; do
        task="${TASKS[$task_index]}"
        echo "[eval-dispatch] task=${task} task_index=${task_index} mode=serial"
        # Serial mode has one X display; task_index must not become a display offset.
        run_eval_task "${task}" "${task_index}" "0"
        status=$?
        if [[ "${status}" == "0" ]]; then
            record_task_result "${task}" "0"
        else
            record_task_result "${task}" "failed"
            FAILED=$((FAILED + 1))
        fi
    done
fi

print_eval_progress "none"

"${PYTHON}" - "${EVAL_ROOT}" "${FAILED}" "${SERIAL_WORKER_COUNT}" "${REQUESTED_EVAL_WORKERS}" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root, failed, workers, requested_workers, *tasks = sys.argv[1:]
task_summaries = {}
success_log_lines = [
    "RLBench evaluation success log",
    f"run_root={Path(root).resolve()}",
]
for task in tasks:
    summary_path = Path(root) / task / "summary.json"
    if summary_path.is_file():
        with open(summary_path, "r", encoding="utf-8") as handle:
            task_summaries[task] = json.load(handle)
    else:
        task_summaries[task] = {"task": task, "status": "evaluation_failed"}

    item = task_summaries[task]
    results = item.get("results", [])
    successful_indices = [
        int(result["episode_index"])
        for result in results
        if result.get("success") is True and "episode_index" in result
    ]
    successes = int(item.get("successes", len(successful_indices)))
    episodes = int(item.get("episodes", len(results)))
    rate = successes / episodes if episodes else 0.0
    success_log_lines.extend([
        "",
        f"task={task}",
        f"successes={successes}",
        f"episodes={episodes}",
        f"success_rate={rate:.3f}",
        "successful_episode_indices=" + ",".join(str(index) for index in successful_indices),
        "successful_trajectory_ids=" + ",".join(f"episode_{index:03d}" for index in successful_indices),
    ])

total_episodes = sum(int(item.get("episodes", 0)) for item in task_summaries.values())
total_successes = sum(int(item.get("successes", 0)) for item in task_summaries.values())

# The task-level config is produced by the Python evaluator after argparse has
# applied command-line overrides.  Use it as the single source of truth so
# eval_config.json cannot report shell defaults instead of Bash arguments.
effective_parameters = None
if tasks:
    effective_path = Path(root) / tasks[0] / "config.json"
    if effective_path.is_file():
        with open(effective_path, "r", encoding="utf-8") as handle:
            effective_parameters = json.load(handle)

# Keep the root-level config useful on its own.  Its effective values must
# come from the parsed task config, where command-line arguments already won
# over the shell defaults.
eval_config_path = Path(root) / "eval_config.json"
if eval_config_path.is_file() and effective_parameters is not None:
    with open(eval_config_path, "r", encoding="utf-8") as handle:
        root_config = json.load(handle)
    root_config["effective_parameters"] = effective_parameters
    for key in (
        "policy_path",
        "episodes",
        "variation",
        "max_model_calls",
        "max_policy_action_steps",
        "controller_error_retries",
        "controller_error_retry_timeout_seconds",
        "controller_error_mode",
        "controller_continue_max_errors",
        "action_index",
        "exec_action_steps",
        "execution_mode",
        "arm_action_mode",
        "clip_within_workspace",
        "mover_max_tries",
        "mover_position_tolerance",
        "mover_rotation_tolerance",
        "mover_gripper_position_tolerance",
        "mover_gripper_rotation_tolerance",
        "reinfer_on_control_error",
        "gripper_after_reach",
        "pointact_pyrep_compat",
        "pointact_pyrep_compatibility",
        "collision_checking",
        "num_points",
        "gripper_points",
        "gripper_template",
        "gripper_delta_threshold",
        "gripper_delta_open_threshold",
        "gripper_delta_close_threshold",
        "gripper_delta_alignment",
        "gripper_mode",
        "gripper_open_threshold",
        "gripper_lock_after_close",
        "franka_ik_max_iterations",
        "franka_ik_tolerance",
        "franka_ik_damping",
        "device",
        "image_size",
        "video_fps",
        "video_width",
        "video_height",
        "save_video",
        "failure_artifacts_only",
        "save_action_records",
        "save_action_chunks",
        "save_action_visualizations",
        "save_determinism_diagnostics",
    ):
        if key in effective_parameters:
            root_config[key] = effective_parameters[key]
    root_config["checkpoint_path"] = str(Path(effective_parameters["policy_path"]).parent)
    root_config["checkpoint_name"] = Path(root_config["checkpoint_path"]).name
    with open(eval_config_path, "w", encoding="utf-8") as handle:
        json.dump(root_config, handle, indent=2, ensure_ascii=True)
task_output_dirs = {}
for task in tasks:
    item = task_summaries[task]
    episodes = int(item.get("episodes", 0))
    successes = int(item.get("successes", 0))
    rate_percent = round(100 * successes / episodes) if episodes else 0
    source = Path(root) / task
    destination = Path(root) / f"{task}_{rate_percent}%"
    if source.is_dir():
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite task output directory: {destination}")
        source.rename(destination)
        task_output_dirs[task] = str(destination.resolve())
        task_config_path = destination / "config.json"
        if task_config_path.is_file():
            with open(task_config_path, "r", encoding="utf-8") as handle:
                task_config = json.load(handle)
            task_config["run_dir"] = task_output_dirs[task]
            with open(task_config_path, "w", encoding="utf-8") as handle:
                json.dump(task_config, handle, indent=2, ensure_ascii=True)

# The run root stays stable, while completed task directories are named after
# their observed success rate for easier browsing.
if eval_config_path.is_file() and task_output_dirs:
    with open(eval_config_path, "r", encoding="utf-8") as handle:
        root_config = json.load(handle)
    root_config["task_config_files"] = {
        task: str(Path(path) / "config.json")
        for task, path in task_output_dirs.items()
    }
    root_config["task_preset_files"] = {
        task: str(Path(root).resolve() / "task_presets" / f"{task}.txt")
        for task in task_output_dirs
    }
    root_config["task_output_dirs"] = task_output_dirs
    if tasks and isinstance(root_config.get("effective_parameters"), dict):
        root_config["effective_parameters"]["run_dir"] = task_output_dirs.get(
            tasks[0], root_config["effective_parameters"].get("run_dir")
        )
    with open(eval_config_path, "w", encoding="utf-8") as handle:
        json.dump(root_config, handle, indent=2, ensure_ascii=True)

summary = {
    "run_root": str(Path(root).resolve()),
    "tasks": tasks,
    "workers": int(workers),
    "requested_workers": int(requested_workers),
    "parallel_evaluation": int(workers) > 1,
    "success_log": str(Path(root).resolve() / "successlog"),
    "worker_failures": int(failed),
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "overall_success_rate": total_successes / max(total_episodes, 1),
    "task_summaries": task_summaries,
    "task_output_dirs": task_output_dirs,
    "effective_parameters": effective_parameters,
}
with open(Path(root) / "summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=True)

with open(Path(root) / "successlog", "w", encoding="utf-8") as handle:
    handle.write("\n".join([
        *success_log_lines[:2],
        "",
        f"total_successes={total_successes}",
        f"total_episodes={total_episodes}",
        f"overall_success_rate={total_successes / total_episodes if total_episodes else 0.0:.3f}",
        *success_log_lines[2:],
        "",
    ]))
PY

summary_mode="serial"
if (( SERIAL_WORKER_COUNT > 1 )); then summary_mode="parallel"; fi
echo "[eval-summary] tasks=${#TASKS[@]} mode=${summary_mode} failed=${FAILED} run_root=${EVAL_ROOT} logs=${EVAL_LOG_DIR} successlog=${EVAL_ROOT}/successlog"
if (( FAILED > 0 )); then
    exit 1
fi
