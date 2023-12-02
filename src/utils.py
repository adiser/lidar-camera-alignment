from typing import Dict, NamedTuple

import cv2
import numpy as np
import torch
import yaml
from fvcore import nn
from pytorch3d.loss import chamfer_distance
from torch.nn import MSELoss


class LabelConfig:
    # REL_LABELS_COUNT = [
    #     (40, 34228),
    #     (70, 27123),
    #     (48, 26360),
    #     (50, 18268),
    #     (0, 6506),
    #     (44, 3268),
    #     (72, 2964),
    #     (52, 1471),
    #     (99, 1227),
    #     (71, 1192),
    # ]

    LABEL_MAP = {
        'car': 10,
        'road': 40,
        'sidewalk': 48,
        # 'building': 50,
        # 'vegetation': 70,
        # 'terrain': 72
    }

    data_source = '../data/Kitti/dataset/semantic-kitti.yaml'
    yaml_dict = yaml.safe_load(open(data_source))

    @classmethod
    def labels(cls):
        return cls.LABEL_MAP.values()

    @classmethod
    def label_map(cls):
        return cls.LABEL_MAP

    @classmethod
    def label_names(cls):
        return cls.LABEL_MAP.keys()

    @classmethod
    def color(cls, label: int):
        # BGR to RGB.
        return cls.yaml_dict['color_map'][label][::-1]


DOWNSCALING_FACTOR = 1000


def load_semkitti_config(data_source: str = '../data/semantic-kitti.yaml'):
    yaml_dict = yaml.safe_load(open(data_source))
    return yaml_dict


def convergence_criteria_frame(kitti, convergence_criteria_frame, camera_id):
    label_img, pts_2d_fov, pts_label_fov, tr, lidar_data = kitti.get_gt_projection(idx=convergence_criteria_frame,
                                                                                   camera_id=camera_id)
    annotated_gt_points_dict = dict()
    for label in LabelConfig.labels():
        pts_2d = torch.from_numpy(pts_2d_fov[pts_label_fov == label])
        pts_2d = torch.hstack([pts_2d, torch.zeros(len(pts_2d))[:, None]]).float()
        annotated_gt_points_dict[label] = pts_2d

    lidar_labels = kitti.load_lidar_labels(convergence_criteria_frame)
    return lidar_data, lidar_labels, annotated_gt_points_dict

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.001, convergence_threshold: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.convergence_threshold = convergence_threshold

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return 1

        if validation_loss < self.convergence_threshold:
            return 2

        return 0

def convergence_criteria(calibrator_model,
                         key_lidar_data,
                         key_lidar_labels,
                         annotated_gt_points_dict,
                         projection_matrix,
                         image_height,
                         image_width,
                         early_stopper: EarlyStopper):

    lidar_data = torch.from_numpy(key_lidar_data)
    lidar_labels = torch.from_numpy(key_lidar_labels.astype(int))

    pred_point_cloud: Dict[int, torch.Tensor] = calibrator_model(lidar_data=lidar_data,
                                                                 projection_matrix=projection_matrix,
                                                                 image_height=image_height,
                                                                 image_width=image_width,
                                                                 label_data=lidar_labels,
                                                                 add_depth=True,
                                                                 depth_scaling_factor=0)

    total_chamdist = 0
    for class_name in LabelConfig.labels():
        pred_points = pred_point_cloud[class_name]
        gt_points = annotated_gt_points_dict[class_name]

        chamdist, _ = chamfer_distance(gt_points[None, :], pred_points[None, :], single_directional=False)
        total_chamdist = total_chamdist + chamdist

    total_chamdist = total_chamdist / DOWNSCALING_FACTOR

    converged = early_stopper.early_stop(total_chamdist)

    return total_chamdist, converged

def choose_samples(sample_range,
                   num_samples,
                   kitti,
                   image_labels_subsampling_factor,
                   camera_id,
                   depth_scaling_factor,
                   use_gt_labels: bool = False):

    samples = list(np.random.choice(list(range(sample_range[0], sample_range[1])), num_samples))

    lidar_samples = [
        LidarSample(lidar_points=kitti.load_lidar_points(i),
                    lidar_labels=kitti.load_lidar_labels(i)) for i in samples
    ]

    gt_image_point_clouds = [
        kitti.get_image_semlabels_with_depth(
            idx=i, subsampling_factor=image_labels_subsampling_factor, camera_id=camera_id,
            depth_scaling_factor=depth_scaling_factor, return_label_gt=use_gt_labels)[1] for i in samples
    ]

    gt_images = [
        kitti.load_image(idx=i, camera_id=camera_id) for i in samples
    ]

    return lidar_samples, gt_image_point_clouds, gt_images

def pred_and_visualize(idx,
                       kitti,
                       calibrator_model,
                       image_labels_subsampling_factor,
                       camera_id,
                       depth_scaling_factor,
                       projection_matrix,
                       image_height,
                       image_width,
                       data_dump_dir,
                       iteration):

    # Visualize it on the first sample:
    lidar_points = torch.from_numpy(kitti.load_lidar_points(idx))
    lidar_labels = torch.from_numpy(kitti.load_lidar_labels(idx).astype(int))
    gt_image_point_clouds_test = kitti.get_image_semlabels_with_depth(
        idx=0, subsampling_factor=image_labels_subsampling_factor, camera_id=camera_id,
        depth_scaling_factor=depth_scaling_factor)[1]
    gt_image = kitti.load_image(idx=idx, camera_id=camera_id)
    gt_image_vis = gt_image.copy()[:, :, ::-1]

    pred_point_cloud: Dict[int, torch.Tensor] = calibrator_model(lidar_data=lidar_points,
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


class LidarSample(NamedTuple):
    lidar_points: np.ndarray
    lidar_labels: np.ndarray
