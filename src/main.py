from typing import NamedTuple, Dict

import cv2
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation
from torch import nn

from calibration import Kitti, get_initialized_input
from utils import LabelConfig, DOWNSCALING_FACTOR
from models.utils import get_model


class LidarSample(NamedTuple):
    lidar_points: np.ndarray
    lidar_labels: np.ndarray


def main_bundle_adjust():
    camera_id = 2

    kitti = Kitti()

    # Get the image properties.
    image_height = kitti.height
    image_width = kitti.width
    projection_matrix = torch.from_numpy(kitti.Ps[camera_id]).float()

    # Parse hyperparameters.
    rot_param_type = 'euler'
    lr = 1e-6
    rotation_degrees = dict(z=5, y=5, x=5)
    translation_meters = dict(z=0.0, y=-0.0, x=-0.0)
    num_iter = 1000
    image_labels_subsampling_factor = 5.
    depth_scaling_factor = 0.1
    num_samples = 16
    data_dump_dir = "../data/batch_test/"
    translation_upweighting = 0

    # Just get the initialized extrinsics from the projection of the first kitti sample
    _, _, _, init_extrinsics, _ = kitti.get_rotated_projection(
        idx=0, camera_id=camera_id, rotation_degrees=rotation_degrees,
        translation_meters=translation_meters
    )

    _, _, _, gt_extrinsics, _ = kitti.get_gt_projection(idx=0, camera_id=2)

    # Modeling components.
    calibrator_model = get_model(init_extrinsics=init_extrinsics, rot_param_type=rot_param_type)
    optimizer = torch.optim.SGD(calibrator_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    assert num_samples < len(kitti), "Cannot get more samples than the KITTI dataset."

    for iteration in range(num_iter):

        total_chamdist = 0
        samples = list(np.random.choice(list(range(0, 200)), num_samples))

        lidar_samples = [
            LidarSample(lidar_points=kitti.load_lidar_points(i),
                        lidar_labels=kitti.load_lidar_labels(i)) for i in samples
        ]

        gt_image_point_clouds = [
            kitti.get_image_semlabels_with_depth(
                idx=i, subsampling_factor=image_labels_subsampling_factor, camera_id=camera_id,
                depth_scaling_factor=depth_scaling_factor)[1] for i in samples
        ]

        gt_images = [
            kitti.load_image(idx=i, camera_id=camera_id) for i in samples
        ]

        # Choose image to plot.
        which_image_to_display = 0
        image = gt_images[which_image_to_display][:, :, ::-1].copy()

        for i, (lidar_sample, gt_image_point_cloud) in enumerate(zip(lidar_samples, gt_image_point_clouds)):

            # Forward pass of the calibrator model
            lidar_sample: LidarSample
            lidar_points = torch.from_numpy(lidar_sample.lidar_points)
            lidar_labels = torch.from_numpy(lidar_sample.lidar_labels.astype(np.int32))

            # Forward pass.
            pred_point_cloud: Dict[str, torch.Tensor] = calibrator_model(lidar_data=lidar_points,
                                                                         projection_matrix=projection_matrix,
                                                                         image_height=image_height,
                                                                         image_width=image_width,
                                                                         label_data=lidar_labels,
                                                                         add_depth=True,
                                                                         depth_scaling_factor=depth_scaling_factor)

            # Convert to Tensor.
            gt_image_point_cloud: Dict[str, np.ndarray]
            gt_image_point_cloud = {class_name: torch.from_numpy(arr).float()
                                    for class_name, arr in gt_image_point_cloud.items()}

            assert sorted(LabelConfig.labels()) == sorted(pred_point_cloud.keys())
            assert sorted(LabelConfig.labels()) == sorted(gt_image_point_cloud.keys())

            for class_name in LabelConfig.labels():

                pred_points = pred_point_cloud[class_name]
                gt_points = gt_image_point_cloud[class_name]

                # Draw the points on the first image.
                # if i == 0:
                #     pred_color = LabelConfig.color(class_name)[::-1]
                #     gt_color = LabelConfig.color(class_name)
                #     image[pred_points[:, 1].int(), pred_points[:, 0].int(), :] = np.array(pred_color)
                #     image[gt_points[:, 1].int(), gt_points[:, 0].int(), :] = np.array(gt_color)

                chamdist, _ = chamfer_distance(gt_points[None, :], pred_points[None, :], single_directional=False)
                # chamdist = chamdist / len(gt_points) * 4000
                total_chamdist = total_chamdist + chamdist


        # Dump the image
        # cv2.imwrite(f'{data_dump_dir}/iteration_{iteration}.png', image[:, :, ::-1])

        # Downscale the value for numerical stability and average them per sample
        total_chamdist = total_chamdist / DOWNSCALING_FACTOR / num_samples

        # Loss function calculation.
        loss = criterion(total_chamdist, torch.tensor(0).float())
        optimizer.zero_grad()
        loss.backward()

        # Upweight the translation update
        for name, param in calibrator_model.named_parameters():
            if name == 'translation_param':
                if param.grad is not None:
                    param.grad *= translation_upweighting

        optimizer.step()

        # Visualize it on the first sample:
        lidar_points = torch.from_numpy(kitti.load_lidar_points(0))
        lidar_labels = torch.from_numpy(kitti.load_lidar_labels(0).astype(int))

        gt_image_point_clouds_test = kitti.get_image_semlabels_with_depth(
            idx=0, subsampling_factor=image_labels_subsampling_factor, camera_id=camera_id,
            depth_scaling_factor=depth_scaling_factor)[1]
        gt_image = kitti.load_image(idx=0, camera_id=camera_id)
        gt_image_vis = gt_image.copy()[:, :, ::-1]

        pred_point_cloud: Dict[str, torch.Tensor] = calibrator_model(lidar_data=lidar_points,
                                                                     projection_matrix=projection_matrix,
                                                                     image_height=image_height,
                                                                     image_width=image_width,
                                                                     label_data=lidar_labels,
                                                                     add_depth=True,
                                                                     depth_scaling_factor=depth_scaling_factor)

        for class_name in LabelConfig.labels():
            pred_points = pred_point_cloud[class_name]
            gt_points = gt_image_point_clouds_test[class_name]

            # Draw the points on the first image.
            pred_color = LabelConfig.color(class_name)[::-1]
            gt_color = LabelConfig.color(class_name)
            gt_image_vis[pred_points[:, 1].int(), pred_points[:, 0].int(), :] = np.array(pred_color)
            gt_image_vis[gt_points[:, 1].astype(int), gt_points[:, 0].astype(int), :] = np.array(gt_color)

        cv2.imwrite(f'{data_dump_dir}/iteration_{iteration}.png', gt_image_vis[:, :, ::-1])

        print(f"Iteration {iteration}: {loss.item()}, {calibrator_model.rot_param.data} {calibrator_model.translation_param.data}")
        print(f'Error translation: {torch.from_numpy(gt_extrinsics[:, 3]) - calibrator_model.translation_param.data }')

        pred_rotation_matrix = calibrator_model.construct_extrinsics_matrix()[:, :3]
        pred_rotation = Rotation.from_matrix(pred_rotation_matrix)
        pred_rotation_euler = pred_rotation.as_euler('ZYX', degrees=True)

        gt_rotation = Rotation.from_matrix(gt_extrinsics[:, :3])
        gt_rotation_euler = gt_rotation.as_euler('ZYX', degrees=True)

        error_rotation = np.abs(pred_rotation_euler - gt_rotation_euler)

        print(f"\t Error rotation: z: {error_rotation[0]}, y: {error_rotation[1]}, z: {error_rotation[2]}")



if __name__ == "__main__":
    main_bundle_adjust()