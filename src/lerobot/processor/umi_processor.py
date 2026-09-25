#!/usr/bin/env python

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

from __future__ import annotations

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("umi")
class UMIProcessor(ProcessorStep):
    """
    Processor to convert world-coordinate actions, states, and point clouds to egocentric coordinates
    relative to the end-effector pose at the start of each chunk.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, transition):
        """
        Apply UMI transformation to the transition.

        Args:
            transition: EnvTransition containing observation data.

        Returns:
            Transformed transition with egocentric coordinates.
        """
        from .core import TransitionKey

        # Create a copy of the transition
        new_transition = transition.copy()

        # Get observation from transition
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            return new_transition
        observation = observation.copy()
        new_transition[TransitionKey.OBSERVATION] = observation

        # Extract data - observation keys have 'observation.' prefix stripped
        if 'observation.state' in observation:
            state = observation['observation.state']
        else:
            # No state data to process
            return new_transition
        if not torch.is_tensor(state):
            raise TypeError(
                "UMI observation.state must be a torch.Tensor, "
                f"got {type(state).__name__}."
            )
        if state.shape[-1] < 9:
            raise ValueError(
                "UMI observation.state must contain at least a 9D pose, "
                f"got shape {state.shape}."
            )
        state_had_sequence_dim = state.ndim == 3
        if state.ndim == 2:
            state_sequence = state[:, None, :]
        elif state.ndim == 3:
            state_sequence = state
        else:
            raise ValueError(
                "UMI observation.state must have shape (B, D) or (B, S, D), "
                f"got {state.shape}."
            )

        # Get action from transition
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not torch.is_tensor(action):
                raise TypeError(
                    f"UMI action must be a torch.Tensor, got {type(action).__name__}."
                )
            if action.shape[-1] < 9:
                raise ValueError(
                    f"UMI action must contain at least a 9D pose, got shape {action.shape}."
                )
            action_had_sequence_dim = action.ndim == 3
            if action.ndim == 2:
                action_sequence = action[:, None, :]
            elif action.ndim == 3:
                action_sequence = action
            else:
                raise ValueError(
                    "UMI action must have shape (B, D) or (B, S, D), "
                    f"got {action.shape}."
                )
        else:
            action_sequence = None
            action_had_sequence_dim = False

        # The egocentric origin must be the actually observed current EEF pose.
        # Using action[0] as the origin was only accidentally valid while action
        # and observation.state were duplicates. It would erase the displacement
        # of a commanded absolute target.
        origin_pose9 = state_sequence[:, 0, :9]
        action_umi = None
        can_transform_together = (
            action_sequence is not None
            and action_sequence.shape[0] == state_sequence.shape[0]
            and action_sequence.device == state_sequence.device
            and action_sequence.dtype == state_sequence.dtype
        )
        if can_transform_together:
            state_steps = state_sequence.shape[1]
            combined_pose9 = torch.cat(
                [state_sequence[..., :9], action_sequence[..., :9]],
                dim=1,
            )
            combined_umi = self.from_world_to_umi_tra_pose9_tensor(
                combined_pose9,
                origin_pose9_eff_to_world=origin_pose9,
            )
            state_umi = combined_umi[:, :state_steps]
            action_umi = combined_umi[:, state_steps:]
        else:
            state_umi = self.from_world_to_umi_tra_pose9_tensor(
                state_sequence[..., :9],
                origin_pose9_eff_to_world=origin_pose9,
            )
        transformed_state = state_sequence.clone()
        transformed_state[..., :9] = state_umi
        if not state_had_sequence_dim:
            transformed_state = transformed_state[:, 0]
        observation['observation.state'] = transformed_state

        # Update action if it exists (assuming action[..., :9] is the pose part)
        if action_sequence is not None:
            if action_umi is None:
                action_umi = self.from_world_to_umi_tra_pose9_tensor(
                    action_sequence[..., :9],
                    origin_pose9_eff_to_world=origin_pose9,
                )
            transformed_action = action_sequence.clone()
            transformed_action[..., :9] = action_umi
            if not action_had_sequence_dim:
                transformed_action = transformed_action[:, 0]
            new_transition[TransitionKey.ACTION] = transformed_action

        # self.vis_umi_data(obs_pose9_data_eff_2_eff0[0],point_cloud.squeeze(1)[0])

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """UMI processor doesn't change feature shapes or types, just coordinate systems."""
        return features

    def from_world_to_umi_tra_pose9(
        self,
        obs_pose9_eff_to_world,
        *,
        origin_pose9_eff_to_world=None,
    ):
        """NumPy-compatible wrapper around the device-preserving Torch transform."""

        poses = np.asarray(obs_pose9_eff_to_world)
        if poses.ndim != 3 or poses.shape[-1] != 9:
            raise ValueError(f"Expected pose sequence shape (B, S, 9), got {poses.shape}.")
        origin_tensor = None
        if origin_pose9_eff_to_world is not None:
            origin_tensor = torch.as_tensor(np.asarray(origin_pose9_eff_to_world))
        result = self.from_world_to_umi_tra_pose9_tensor(
            torch.as_tensor(poses),
            origin_pose9_eff_to_world=origin_tensor,
        )
        return result.cpu().numpy().astype(poses.dtype, copy=False)

    def from_world_to_umi_tra_pose9_tensor(
        self,
        obs_pose9_eff_to_world: Tensor,
        *,
        origin_pose9_eff_to_world: Tensor | None = None,
    ) -> Tensor:
        """Express a pose sequence in the current EEF frame without leaving Torch.

        The computation stays on the input device, avoiding the synchronous
        CUDA->CPU->CUDA copies that the former NumPy implementation introduced
        for every training batch. Float16/bfloat16 inputs are evaluated in
        float32 for stable rotation orthogonalization and cast back afterwards.
        """

        poses = obs_pose9_eff_to_world
        if not torch.is_tensor(poses):
            raise TypeError(f"Expected pose sequence tensor, got {type(poses).__name__}.")
        if poses.ndim != 3 or poses.shape[-1] != 9:
            raise ValueError(f"Expected pose sequence shape (B, S, 9), got {poses.shape}.")
        if not poses.is_floating_point():
            raise TypeError(f"UMI pose tensors must be floating point, got {poses.dtype}.")

        batch_size = poses.shape[0]
        output_dtype = poses.dtype
        compute_dtype = (
            torch.float32 if poses.dtype in (torch.float16, torch.bfloat16) else poses.dtype
        )
        poses_compute = poses.to(dtype=compute_dtype)

        if origin_pose9_eff_to_world is None:
            origin_poses = poses_compute[:, 0]
        else:
            if not torch.is_tensor(origin_pose9_eff_to_world):
                raise TypeError(
                    "UMI origin must be a torch.Tensor, "
                    f"got {type(origin_pose9_eff_to_world).__name__}."
                )
            origin_poses = origin_pose9_eff_to_world
            if origin_poses.ndim == 3 and origin_poses.shape[1] == 1:
                origin_poses = origin_poses[:, 0]
            if origin_poses.shape != (batch_size, 9):
                raise ValueError(
                    "UMI origin must have shape (B, 9), "
                    f"got {origin_poses.shape} for batch size {batch_size}."
                )
            origin_poses = origin_poses.to(device=poses.device, dtype=compute_dtype)

        translation = poses_compute[..., :3]
        rotation = self.rot6d_to_matrix(poses_compute[..., 3:9])
        origin_translation = origin_poses[..., :3]
        origin_rotation_inv = self.rot6d_to_matrix(origin_poses[..., 3:9]).transpose(-1, -2)
        relative_translation = (
            origin_rotation_inv[:, None]
            @ (translation - origin_translation[:, None]).unsqueeze(-1)
        ).squeeze(-1)
        relative_rotation = origin_rotation_inv[:, None] @ rotation
        result = torch.cat(
            [
                relative_translation,
                relative_rotation[..., :, 0],
                relative_rotation[..., :, 1],
            ],
            dim=-1,
        )
        return result.to(dtype=output_dtype)

    def create_frame(self, position, rot_matrix, scale=0.03):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale,
            origin=[0, 0, 0]
        )
        frame.rotate(rot_matrix, center=np.zeros(3))
        frame.translate(position)
        return frame
    def vis_umi_data(self,action,pointcloud):
        ##########UMI
        # ================= Pred =================

        geometries =[]
        origin_frame = self.create_frame(np.array([0,0,0]), np.eye(3), scale=0.03)
        geometries.append(origin_frame)
        for per_pred_action in action: ####GT
            per_pred_action = per_pred_action
            pred_xyz = per_pred_action[:3]
            pred_rot6d = per_pred_action[3:9]
            pred_rotmat = self.rot6d_to_matrix(torch.tensor(pred_rot6d)).cpu().numpy()
            frame = self.create_frame(pred_xyz, pred_rotmat, scale=0.03)
            geometries.append(frame)

        # ================= Scene Point Cloud =================
        cloud = pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] / 255)
        geometries.append(pcd)

        o3d.visualization.draw_geometries(geometries)
        ##########UMI

    def from_world_to_umi_pointcloud(self, obs_pose9_eff_to_world, pointcloud_world):
        # pointcloud_world shape: (batch_size, seq_len, N, 6)
        batch_size, seq_len, N = pointcloud_world.shape[:3]

        # Convert pose9 to homogeneous matrices - same as in trajectory function
        T_world_list = []
        for i in range(batch_size):
            for j in range(seq_len):
                pose9 = obs_pose9_eff_to_world[i, j]
                T_world = self.pose9_to_homo(torch.tensor(pose9).unsqueeze(0)).squeeze(0).cpu().numpy()
                T_world_list.append(T_world)

        T_world = np.array(T_world_list).reshape(batch_size, seq_len, 4, 4)  # (batch_size, seq_len, 4, 4)

        # Compute inverse transformations
        T_inv_fast = self.fast_inverse_homogeneous(T_world)  # (batch_size, seq_len, 4, 4)

        # Expand for point cloud: (batch_size, seq_len, 1, 4, 4) -> (batch_size, seq_len, N, 4, 4)
        T_inv_fast_expand = np.repeat(T_inv_fast[:, :, np.newaxis, :, :], N, axis=2)  # (batch_size, seq_len, N, 4, 4)

        # Convert point cloud to homogeneous coordinates
        # pointcloud_world[..., :3] shape: (batch_size, seq_len, N, 3)
        # P_world_H shape: (batch_size, seq_len, N, 4)
        P_world_H = np.concatenate([
            pointcloud_world[..., :3],  # xyz
            np.ones((batch_size, seq_len, N, 1))  # homogeneous coordinate
        ], axis=-1)

        # Apply transformation
        # Reshape for matrix multiplication
        P_world_H_reshaped = P_world_H.reshape(-1, 4, 1)  # (batch_size*seq_len*N, 4, 1)
        T_inv_reshaped = T_inv_fast_expand.reshape(-1, 4, 4)  # (batch_size*seq_len*N, 4, 4)

        P_eff_H = (T_inv_reshaped @ P_world_H_reshaped).squeeze(-1)  # (batch_size*seq_len*N, 4)

        # Extract xyz coordinates and concatenate with rgb
        P_eff_xyz = P_eff_H[..., :3].reshape(batch_size, seq_len, N, 3)  # (batch_size, seq_len, N, 3)
        P_eff_rgb = pointcloud_world[..., 3:]  # (batch_size, seq_len, N, 3)

        P_eff = np.concatenate([P_eff_xyz, P_eff_rgb], axis=-1)  # (batch_size, seq_len, N, 6)

        return P_eff.astype(np.float32)

    def pose9_to_homo(self, pose9: torch.Tensor) -> torch.Tensor:
        t = pose9[..., 0:3]
        R = self.rot6d_to_matrix(pose9[..., 3:9])
        H = torch.zeros(*pose9.shape[:-1], 4, 4, device=pose9.device, dtype=pose9.dtype)
        H[..., 3, 3] = 1.0
        H[..., :3, :3] = R
        H[..., :3, 3] = t
        return H

    def rot6d_to_matrix(self, d6: torch.Tensor) -> torch.Tensor:
        a1 = d6[..., 0:3]
        a2 = d6[..., 3:6]
        b1 = F.normalize(a1, dim=-1, eps=1e-6)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1, eps=1e-6)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # columns

    def fast_inverse_homogeneous(self, T):
        """
        Input: T (..., 4, 4) with arbitrary leading dimensions.
        Output: T_inv (..., 4, 4).
        """
        # Preserve the original leading dimensions.
        original_shape = T.shape
        batch_shape = original_shape[:-2]  # All dimensions except the final 4x4.

        # Flatten to (*batch_shape, 4, 4).
        T_reshaped = T.reshape(-1, 4, 4)

        # 1. Extract rotation R and translation t.
        R = T_reshaped[:, :3, :3]  # (N, 3, 3)
        t = T_reshaped[:, :3, 3:]  # (N, 3, 1)

        # 2. Transpose R to obtain its inverse.
        R_inv = np.transpose(R, (0, 2, 1))  # (N, 3, 3)

        # 3. Compute the inverse translation: -R^T * t.
        t_inv = - (R_inv @ t)  # (N, 3, 1)

        # 4. Assemble the inverse transform.
        T_inv_reshaped = np.tile(np.eye(4), (T_reshaped.shape[0], 1, 1))  # (N, 4, 4)
        T_inv_reshaped[:, :3, :3] = R_inv
        T_inv_reshaped[:, :3, 3:] = t_inv

        # Restore the original shape.
        T_inv = T_inv_reshaped.reshape(original_shape)

        return T_inv

    def from_H_to_trajectory(self, H):
        """Convert a homogeneous transform to trajectory data."""
        position = H[:3, 3]
        rotation_matrix = H[:3, :3]
        euler_zyx = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
        trajectory = np.hstack((position, euler_zyx))
        return trajectory

    def traj6_to_pose9(self, traj6: np.ndarray) -> np.ndarray:
        if len(traj6.shape) == 1:
            t = traj6[:3].astype(np.float32)
            euler_zyx = traj6[3:].astype(np.float32)
            Rm = R.from_euler('zyx', euler_zyx, degrees=False).as_matrix().astype(np.float32)
            rot6d = np.hstack((Rm[:, 0], Rm[:, 1]))
            return np.concatenate([t, rot6d], axis=0)

        # Handle multi-dimensional case
        original_shape = traj6.shape
        if len(original_shape) == 3:  # (batch_size, seq_len, 6)
            batch_size, seq_len = original_shape[:2]
            t = traj6[..., :3].reshape(-1, 3).astype(np.float32)  # (batch_size*seq_len, 3)
            euler_zyx = traj6[..., 3:].reshape(-1, 3).astype(np.float32)  # (batch_size*seq_len, 3)

            # Compute rotation matrices in a batch.
            rotations = R.from_euler('zyx', euler_zyx, degrees=False)
            Rm = rotations.as_matrix().astype(np.float32)  # (batch_size*seq_len, 3, 3)
            # Extract the first two rotation columns as (batch_size*seq_len, 6).
            rot6d = np.concatenate([Rm[:, :, 0], Rm[:, :, 1]], axis=1)  # (batch_size*seq_len, 6)
            # Concatenate translation and rotation-6D.
            pose9_flat = np.concatenate([t, rot6d], axis=1)  # (batch_size*seq_len, 9)
            # Restore the original shape.
            pose9 = pose9_flat.reshape(batch_size, seq_len, 9)
            return pose9

        # Fallback for other shapes
        t = traj6[:, :3].astype(np.float32)
        euler_zyx = traj6[:, 3:].astype(np.float32)
        # Compute rotation matrices in a batch.
        rotations = R.from_euler('zyx', euler_zyx, degrees=False)
        Rm = rotations.as_matrix().astype(np.float32)  # (N, 3, 3)
        # Extract the first two rotation columns as (N, 6).
        rot6d = np.concatenate([Rm[:, :, 0], Rm[:, :, 1]], axis=1)  # (N, 6)
        # Concatenate translation and rotation-6D.
        pose9 = np.concatenate([t, rot6d], axis=1)  # (N, 9)

        return pose9
