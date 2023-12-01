from typing import Dict

import cv2
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation
from torch import nn

from calibration import Kitti
from utils import LabelConfig, DOWNSCALING_FACTOR, convergence_criteria_frame, convergence_criteria, EarlyStopper, \
    choose_samples, pred_and_visualize, LidarSample
from models.utils import get_model


def get_error(pred_extrinsics, gt_extrinsics):
    pred_rotation_matrix = pred_extrinsics[:, :3]

    delta_rotation_matrix = np.linalg.inv(pred_rotation_matrix) @ gt_extrinsics[:, :3]
    delta_rotation = Rotation.from_matrix(delta_rotation_matrix)
    delta_rotation_euler = delta_rotation.as_euler('ZYX', degrees=True)

    return delta_rotation_euler


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
    depth_scaling_factor = 1
    sample_range = (0, 1)
    num_samples = 1
    data_dump_dir = "../data/batch_test/"
    translation_upweighting = 0

    # Visualization
    sample_to_visualize = 0

    # Convergence stuff
    convergence_criteria_frame_id = 525
    patience = 50
    min_delta = 0.01
    convergence_threshold = 0.01

    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, convergence_threshold=convergence_threshold)

    # Just get the initialized extrinsics from the projection of the first kitti sample
    _, _, _, init_extrinsics, _ = kitti.get_rotated_projection(
        idx=0, camera_id=camera_id, rotation_degrees=rotation_degrees,
        translation_meters=translation_meters
    )
    _, _, _, gt_extrinsics, _ = kitti.get_gt_projection(idx=0, camera_id=2)

    # Choose a particular frame as convergence criteria
    key_lidar_data, key_lidar_labels, annotated_gt_points_dict = convergence_criteria_frame(
        kitti, convergence_criteria_frame_id, camera_id)

    # Modeling components.
    calibrator_model = get_model(init_extrinsics=init_extrinsics, rot_param_type=rot_param_type)
    optimizer = torch.optim.SGD(calibrator_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    assert num_samples < len(kitti), "Cannot get more samples than the KITTI dataset."

    for iteration in range(num_iter):

        total_chamdist = 0

        # Choose samples.
        lidar_samples, gt_image_point_clouds, gt_images = choose_samples(sample_range,
                                                                         num_samples,
                                                                         kitti,
                                                                         image_labels_subsampling_factor,
                                                                         camera_id,
                                                                         depth_scaling_factor)

        # Choose image to plot.

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

                chamdist, _ = chamfer_distance(gt_points[None, :], pred_points[None, :], single_directional=False)
                total_chamdist = total_chamdist + chamdist

        # Downscale the value for numerical stability and average them per sample
        total_chamdist = total_chamdist / DOWNSCALING_FACTOR / num_samples

        # Loss function calculation.
        loss = criterion(total_chamdist, torch.tensor(0).float())
        optimizer.zero_grad()
        loss.backward()

        # To perform translation update.
        for name, param in calibrator_model.named_parameters():
            if name == 'translation_param':
                if param.grad is not None:
                    param.grad *= translation_upweighting

        optimizer.step()

        pred_and_visualize(sample_to_visualize,
                           kitti,
                           calibrator_model,
                           image_labels_subsampling_factor,
                           camera_id,
                           depth_scaling_factor,
                           projection_matrix,
                           image_height,
                           image_width,
                           data_dump_dir,
                           iteration)

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

        print(
            f"Iteration {iteration}: {loss.item()}, {calibrator_model.rot_param.data} {calibrator_model.translation_param.data}")
        print(f'Error translation: {torch.from_numpy(gt_extrinsics[:, 3]) - calibrator_model.translation_param.data}')

        # Get the error
        pred_extrinsics = calibrator_model.construct_extrinsics_matrix()
        delta_rotation_euler = get_error(pred_extrinsics=pred_extrinsics, gt_extrinsics=gt_extrinsics)

        # Infer in convergence criteria frame.
        error_criteria, converged = convergence_criteria(calibrator_model=calibrator_model,
                                                         key_lidar_data=key_lidar_data,
                                                         key_lidar_labels=key_lidar_labels,
                                                         annotated_gt_points_dict=annotated_gt_points_dict,
                                                         projection_matrix=projection_matrix,
                                                         image_height=image_height,
                                                         image_width=image_width,
                                                         early_stopper=early_stopper)

        print(
            f"Error rotation: z: {delta_rotation_euler[0]}, y: {delta_rotation_euler[1]}, x: {delta_rotation_euler[2]}")
        print(f"Convergence criteria: {error_criteria}")

        if converged:
            assert converged in [1, 2]
            convergence_criteria_dict = {1: "Patience", 2: "Convergence threshold"}
            raise SystemExit(f"Converged with criteria {converged}: {convergence_criteria_dict[converged]}")


if __name__ == "__main__":
    main_bundle_adjust()
