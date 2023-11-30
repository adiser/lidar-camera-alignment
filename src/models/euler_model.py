import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import Parameter

from utils import LabelConfig


class EulerModel(nn.Module):
    def __init__(self, init_extrinsics):

        super().__init__()

        self.init_extrinsics = init_extrinsics

        # 1. Get the rotation
        rotation = Rotation.from_matrix(self.init_extrinsics[:, :3])
        # Scalar last

        yaw, pitch, roll = rotation.as_euler('ZYX', degrees=False)

        self.rot_param = Parameter(torch.tensor([yaw, pitch, roll]).float())

        # # 2. Get the translation
        tx, ty, tz = self.init_extrinsics[:, 3]
        self.translation_param = Parameter(torch.tensor([tx, ty, tz]).float())

    def construct_extrinsics_matrix(self) -> np.ndarray:
        extrinsics = np.zeros((3, 4), dtype=np.float32)
        R = np.array(self.construct_rotation_matrix().detach().numpy())
        extrinsics[:, :3] = R
        extrinsics[:, 3] = np.array(self.translation_param.detach().cpu())
        return extrinsics

    def construct_rotation_matrix(self):
        R = euler_angles_to_matrix(self.rot_param, convention='ZYX')
        return R

    def forward(self,
                lidar_data,
                projection_matrix,
                image_height,
                image_width,
                label_data,
                add_depth=True,
                depth_scaling_factor: float = 1.):

        R = self.construct_rotation_matrix()

        Tr = torch.hstack([R, self.translation_param[:, None]])

        # Apply the extrinsics
        lidar_points_hom = torch.hstack([lidar_data, torch.ones(len(lidar_data), 1)])
        lidar_points_camera_coords = lidar_points_hom @ Tr.transpose(1, 0)

        # Convert to homography coordinates
        depth_data = lidar_points_camera_coords[:, 2] * depth_scaling_factor

        lidar_points_camera_coords_hom = torch.hstack([lidar_points_camera_coords,
                                                       torch.ones((len(lidar_points_camera_coords), 1))])

        # Apply the projection matrix.
        pts_2d_image_coords_hom = lidar_points_camera_coords_hom @ projection_matrix.transpose(1, 0)
        pts_2d_image_coords_cart = pts_2d_image_coords_hom[:, :2]
        pts_2d_image_coords_cart = pts_2d_image_coords_cart / pts_2d_image_coords_hom[:, 2, None]

        # Only take everything that's within the image
        fov_inds = (pts_2d_image_coords_cart[:, 0] < image_width - 1) & (pts_2d_image_coords_cart[:, 0] >= 0) & \
                   (pts_2d_image_coords_cart[:, 1] < image_height - 1) & (pts_2d_image_coords_cart[:, 1] >= 0)

        fov_inds = fov_inds & (lidar_data[:, 0] > 2)
        pts_2d_image_coords_cart = pts_2d_image_coords_cart[fov_inds, :]
        label_data = label_data[fov_inds]
        depth_data = depth_data[fov_inds]

        points_dict = dict()
        for rel_label in LabelConfig.labels():
            if add_depth:
                position = pts_2d_image_coords_cart[label_data == rel_label]
                depth = depth_data[label_data == rel_label]
                points_dict[rel_label] = torch.hstack([position, depth[:, None]])
            else:
                points_dict[rel_label] = pts_2d_image_coords_cart[label_data == rel_label]

        return points_dict

    # def forward(self,
    #             lidar_data,
    #             projection_matrix,
    #             image_height,
    #             image_width,
    #             label_data,
    #             add_depth=True,
    #             depth_scaling_factor: float = 1.):
    #
    #     R = self.construct_rotation_matrix()
    #
    #     Tr = torch.hstack([R, self.translation_param[:, None]])
    #
    #     # Apply the extrinsics
    #
    #     lidar_points_hom = torch.cat([lidar_data, torch.ones([lidar_data.shape[0], lidar_data.shape[1], 1])], dim=-1)
    #     lidar_points_camera_coords = lidar_points_hom @ Tr.transpose(1, 0)
    #
    #     # Convert to homography coordinates
    #     # depth_data = lidar_points_camera_coords[:, 2] * depth_scaling_factor
    #     depth_data = lidar_points_camera_coords[:, :, 2]
    #
    #     lidar_points_camera_coords_hom = torch.cat([lidar_points_camera_coords,
    #                                                 torch.ones([lidar_data.shape[0], lidar_data.shape[1], 1])], dim=-1)
    #
    #     # Apply the projection matrix.
    #     pts_2d_image_coords_hom = lidar_points_camera_coords_hom @ projection_matrix.transpose(1, 0)
    #     pts_2d_image_coords_cart = pts_2d_image_coords_hom[:, :, :2]
    #     pts_2d_image_coords_cart = pts_2d_image_coords_cart / pts_2d_image_coords_hom[:, :, 2][:, :, None]
    #
    #     # Only take everything that's within the image
    #     cond_1 = pts_2d_image_coords_cart[:, :, 0] < (image_width - 1)
    #     cond_2 = pts_2d_image_coords_cart[:, :, 0] >= 0
    #     cond_3 = pts_2d_image_coords_cart[:, :, 1] < (image_height - 1)
    #     cond_4 = pts_2d_image_coords_cart[:, :, 1] >= 0
    #     cond_5 = lidar_data[:, :, 0] > 2
    #
    #     filter_mask = cond_1 & cond_2 & cond_3 & cond_4 & cond_5
    #
    #     # fov_inds = (pts_2d_image_coords_cart[:, 0] < image_width - 1) & (pts_2d_image_coords_cart[:, 0] >= 0) & \
    #     #            (pts_2d_image_coords_cart[:, 1] < image_height - 1) & (pts_2d_image_coords_cart[:, 1] >= 0)
    #     #
    #     # fov_inds = fov_inds & (lidar_data[:, 0] > 2)
    #     # pts_2d_image_coords_cart = pts_2d_image_coords_cart[filter_mask[:, :, None].repeat(1, 1, 2)]
    #     # label_data = label_data[filter_mask]
    #     # depth_data = depth_data[filter_mask]
    #
    #     batch_size = pts_2d_image_coords_cart.shape[0]
    #
    #     points_dict = {k: [] for k in LabelConfig.labels()}
    #     for b in range(batch_size):
    #         sample_pos = pts_2d_image_coords_cart[b, filter_mask[b]]
    #         sample_depth = depth_data[b, filter_mask[b]]
    #         sample_label = label_data[b, filter_mask[b]]
    #
    #         for rel_label in LabelConfig.labels():
    #             # Apply filter mask
    #             position = sample_pos[sample_label == rel_label]
    #             depth = sample_depth[sample_label == rel_label]
    #             position_with_depth = torch.hstack([position, depth[:, None]])
    #             points_dict[rel_label].append(position_with_depth)
    #
    #     for rel_label in LabelConfig.labels():
    #         points_dict[rel_label] = collate_points(points_dict[rel_label])
    #
    #     return points_dict
