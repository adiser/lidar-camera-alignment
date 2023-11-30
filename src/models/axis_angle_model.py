import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import Parameter

from utils import LabelConfig


class AxisAngleModel(nn.Module):
    def __init__(self, init_extrinsics: np.ndarray):

        super().__init__()

        self.init_extrinsics = init_extrinsics

        rotation = Rotation.from_matrix(self.init_extrinsics[:, :3])

        # TODO: Be wary of this.
        axis_angle_tensor = matrix_to_axis_angle(torch.from_numpy(self.init_extrinsics[:, :3]))
        axis_angle_tensor = axis_angle_tensor.clone().detach().float()
        self.rot_param = Parameter(axis_angle_tensor)

        tx, ty, tz = self.init_extrinsics[:, 3]
        self.translation_param = Parameter(torch.tensor([tx, ty, tz]).float())

    def construct_rotation_matrix(self):
        R = axis_angle_to_matrix(self.rot_param)
        return R

    def construct_extrinsics_matrix(self) -> np.ndarray:
        extrinsics = np.zeros((3, 4), dtype=np.float32)
        R = np.array(self.construct_rotation_matrix().detach().numpy())
        extrinsics[:, :3] = R
        extrinsics[:, 3] = np.array(self.translation_param.detach().cpu())
        return extrinsics

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

