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
import argparse
from config_parser import ConfigParser
from tqdm import tqdm
import wandb


def get_error(pred_extrinsics, gt_extrinsics):
    pred_rotation_matrix = pred_extrinsics[:, :3]

    delta_rotation_matrix = np.linalg.inv(pred_rotation_matrix) @ gt_extrinsics[:, :3]
    delta_rotation = Rotation.from_matrix(delta_rotation_matrix)
    delta_rotation_euler = delta_rotation.as_euler('ZYX', degrees=True)

    return delta_rotation_euler


def main_bundle_adjust(config):

    ## Initializing all the required parameters
    camera_id = config.camera_id
    sequence = config.sequence
    rot_param_type = config.rot_param_type
    lr = config.lr
    num_iter = config.num_iter
    image_labels_subsampling_factor = config.image_labels_subsampling_factor
    depth_scaling_factor = config.depth_scaling_factor
    sample_range = (config.sample_range["min"], config.sample_range["max"])
    num_samples = config.num_samples
    data_dump_dir = config.data_dump_dir
    translation_upweighting = config.translation_upweighting
    key_frame_id = config.key_frame_id
    patience = config.patience
    min_delta = config.min_delta
    convergence_threshold = config.convergence_threshold
    batch_sampling_type = config.batch_sampling_type
    use_depth = config.use_depth

    # Initialize logger
    

    # Visualization, Convergence (Using the same key_frame_id for both)
    sample_to_visualize = key_frame_id
    convergence_criteria_frame_id = key_frame_id


    kitti = Kitti(sequence=sequence)

    # Get the image properties.
    image_height = kitti.height
    image_width = kitti.width
    projection_matrix = torch.from_numpy(kitti.Ps[camera_id]).float()



    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, convergence_threshold=convergence_threshold)
    logger = None

    for j in range(len(config.noise_configs)):
        rotation_degrees = config.noise_configs[j]["rotation_degrees"]
        translation_meters = config.noise_configs[j]["translation"]
        if logger is not None:
            wandb.finish()

        logger = wandb.init(project="single_image_calibration", config=config.get_dict())

        run_name =str(config.sequence) + str(config.key_frame_id) +  "Rotation: " + str(rotation_degrees) + " Translation: " + str(translation_meters)
        logger.name = run_name
        # logger.group = str(config.sequence) + str(config.key_frame_id)

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

        for iteration in tqdm(range(num_iter)):

            total_chamdist = 0

            # Choose samples.
            lidar_samples, gt_image_point_clouds, gt_images = choose_samples(sample_range,
                                                                            num_samples,
                                                                            kitti,
                                                                            image_labels_subsampling_factor,
                                                                            camera_id,
                                                                            depth_scaling_factor,
                                                                            key_frame_id,
                                                                            sampling_method=batch_sampling_type)

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

                    chamdist, _ = chamfer_distance(pred_points[None, :], gt_points[None, :], single_directional=True)
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
            if iteration % 100 == 0:

                vis_image = pred_and_visualize(sample_to_visualize,
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
                logger.log({"visualization": wandb.Image(vis_image)})

            

            # print(
            #     f"Iteration {iteration}: {loss.item()}, {calibrator_model.rot_param.data} {calibrator_model.translation_param.data}")
            # print(f'Error translation: {torch.from_numpy(gt_extrinsics[:, 3]) - calibrator_model.translation_param.data}')
            logger.log({"loss": loss.item()})

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

            # print(
            #     f"Error rotation: z: {delta_rotation_euler[0]}, y: {delta_rotation_euler[1]}, x: {delta_rotation_euler[2]}")
            # print(f"Convergence criteria: {error_criteria}")

            logger.log({"error_rotation_z": delta_rotation_euler[0],
                        "error_rotation_y": delta_rotation_euler[1],
                        "error_rotation_x": delta_rotation_euler[2],
                        "error_translation_x": torch.from_numpy(gt_extrinsics[:, 3])[0] - calibrator_model.translation_param.data[0],
                        "error_translation_y": torch.from_numpy(gt_extrinsics[:, 3])[1] - calibrator_model.translation_param.data[1],
                        "error_translation_z": torch.from_numpy(gt_extrinsics[:, 3])[2] - calibrator_model.translation_param.data[2]})

            if converged:
                assert converged in [1, 2]
                convergence_criteria_dict = {1: "Patience", 2: "Convergence threshold"}
                # raise SystemExit(f"Converged with criteria {converged}: {convergence_criteria_dict[converged]}")
                wandb.finish()
                break


    

def parse_args():
    parser = argparse.ArgumentParser(description='Process configuration file path')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config_file_path = args.config
    config = ConfigParser(config_file_path)
    main_bundle_adjust(config)

    # Use the config_file_path to get the parameters
