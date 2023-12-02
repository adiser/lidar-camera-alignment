import glob
import os
from dataclasses import dataclass
from typing import Optional, Any, Dict

import cv2
import numpy as np
import matplotlib
import torch
import yaml
from pytorch3d.loss import chamfer_distance
from scipy.spatial.transform import Rotation

from utils import LabelConfig

matplotlib.use('TkAgg')


def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


@dataclass()
class CalibInput:
    label_map: np.ndarray | torch.Tensor
    pts_2d_fov: np.ndarray | torch.Tensor
    label_fov: np.ndarray | torch.Tensor
    tr: np.ndarray | torch.Tensor

    def torch(self):
        label_map = torch.from_numpy(self.label_map)
        pts_2d_fov = torch.from_numpy(self.pts_2d_fov)
        label_fov = torch.from_numpy(self.label_fov)
        tr = torch.from_numpy(self.tr)
        return CalibInput(label_map=label_map, pts_2d_fov=pts_2d_fov, label_fov=label_fov, tr=tr)


class Kitti:
    def __init__(self,
                 sequence: str = '00',
                 data_root: str = '../data/Kitti/dataset'):
        self.sequence = sequence
        self.data_root = data_root

        self.yaml_file = os.path.join(self.data_root, 'semantic-kitti.yaml')
        self.sequence_folder = os.path.join(self.data_root, 'sequences', self.sequence)
        self.calib_file = os.path.join(self.data_root, 'sequences', self.sequence, 'calib.txt')
        self.calib_data = read_calib_file(self.calib_file)

        self.kitti_config = yaml.safe_load(open(self.yaml_file, 'r'))
        self.color_map = self.kitti_config['color_map']
        self.tr = self.calib_data['Tr'].reshape(3, 4)
        self.p0 = self.calib_data['P0'].reshape(3, 4)
        self.p1 = self.calib_data['P1'].reshape(3, 4)
        self.p2 = self.calib_data['P2'].reshape(3, 4)
        self.p3 = self.calib_data['P3'].reshape(3, 4)

        self.Ps = [self.p0, self.p1, self.p2, self.p3]

        self.height, self.width = self._get_image_shape()

    def _get_image_shape(self):
        img_file = os.path.join(self.sequence_folder, f'image_{2}', f'{str(0).zfill(6)}.png')
        img = cv2.imread(img_file)
        height, width, _ = img.shape
        return height, width

    def load_lidar_points(self, idx):
        bin_file = os.path.join(self.sequence_folder, 'velodyne', f'{str(idx).zfill(6)}.bin')
        # Just the xyz points.
        lidar = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))[:, :3]
        return lidar

    def load_image(self, idx, camera_id):
        img_file = os.path.join(self.sequence_folder, f'image_{camera_id}', f'{str(idx).zfill(6)}.png')
        img = cv2.imread(img_file)
        return img

    def project_lidar_to_image_coordinates(self, lidar_points, camera_id: int, tr: Optional[np.ndarray] = None):
        if tr is None:
            tr = self.tr

        assert camera_id in [0, 1, 2, 3]

        lidar_points_hom = np.hstack([lidar_points, np.ones((len(lidar_points), 1))])
        lidar_points_camera_coords = lidar_points_hom @ np.transpose(tr)
        P = self.Ps[camera_id]
        lidar_points_camera_hom = np.hstack([lidar_points_camera_coords, np.ones((len(lidar_points), 1))])
        pts_2d_image_coords = lidar_points_camera_hom @ np.transpose(P)
        pts_2d_image_coords[:, 0] /= pts_2d_image_coords[:, 2]
        pts_2d_image_coords[:, 1] /= pts_2d_image_coords[:, 2]
        return pts_2d_image_coords[:, :2]

    def load_lidar_labels(self, idx):
        label_file = os.path.join(self.sequence_folder, 'labels', f"{str(idx).zfill(6)}.label")
        label = np.fromfile(label_file, dtype=np.uint32)
        label = label & 0xFFFF
        return label

    def get_gt_projection(self, idx: int, camera_id: int = 2):
        return self.project_semlabels(idx, camera_id, None)

    def get_rotated_projection(self,
                               idx: int,
                               camera_id: int = 2,
                               rotation_degrees=None,
                               translation_meters=None):
        if rotation_degrees is None:
            rotation_degrees = dict(z=10, y=5, x=5)
        if translation_meters is None:
            translation_meters = dict(z=10, y=5, x=5)

        assert rotation_degrees is not None
        return self.project_semlabels(idx, camera_id, rotation_degrees, translation_meters=translation_meters)

    def get_depth(self, idx, camera_id: int = 2, min_depth=1.0, max_depth=50., depth_scaling_factor=20):
        depth_folder = os.path.join(self.sequence_folder, f'image_{camera_id}')
        depth_data_filename = os.path.join(depth_folder, f'{str(idx).zfill(6)}_depth.npy')
        depth_data = np.load(depth_data_filename)

        depth_data = torch.nn.functional.interpolate(
            torch.from_numpy(depth_data), (self.height, self.width), mode="bilinear", align_corners=False)
        depth_data = depth_data.detach().cpu().numpy()
        depth_data = depth_data.squeeze()
        depth_data = depth_data * depth_scaling_factor

        # Clamp em
        # depth_data[depth_data < min_depth] = min_depth
        # depth_data[depth_data > max_depth] = max_depth

        return depth_data

    def get_image_semlabels_with_depth(self,
                                       idx,
                                       subsampling_factor: float = 2.,
                                       camera_id: Optional[int] = 2,
                                       depth_scaling_factor: float = 5.,
                                       return_label_gt: bool = False):

        image_semlabel_dir = os.path.join(self.sequence_folder, f'image_labels_{camera_id}')
        label_map = LabelConfig.label_map()

        if return_label_gt:
            label_img, pts_2d_fov, pts_label_fov, tr, lidar_data = self.project_semlabels(
                idx, camera_id, rotation_degrees=None, translation_meters=None
            )

        depth_image = self.get_depth(idx, camera_id, depth_scaling_factor=depth_scaling_factor)

        gt_3d_positions_dict = dict()
        gt_labels_positions_dict = dict()
        for label, label_id in label_map.items():

            if not return_label_gt:
                label_img_file = os.path.join(image_semlabel_dir, label, f'{str(idx).zfill(6)}.png')
                if os.path.exists(label_img_file):
                    label_img = cv2.imread(label_img_file)
                else:
                    label_img = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)

                subsampled_height = int(round(label_img.shape[0] / subsampling_factor))
                subsampled_width = int(round(label_img.shape[1] / subsampling_factor))

                # The trick to subsample is to get them from the small image.
                label_img_subsampled = cv2.resize(label_img, (subsampled_width, subsampled_height))

                # Height width to width height
                mask_locations = np.argwhere(label_img_subsampled > 0)[:, :2][:, ::-1]

                depth_val = depth_image[
                    np.round(mask_locations[:, 1]).astype(int), np.round(mask_locations[:, 0]).astype(int)]
                # Revert it back to its original size.
                mask_locations = mask_locations * subsampling_factor

                gt_labels_positions_dict[label_id] = mask_locations
                gt_3d_positions_dict[label_id] = np.hstack([mask_locations, depth_val[:, None]])
            else:
                mask_locations = pts_2d_fov[pts_label_fov == label_id]
                depth_val = depth_image[
                    np.round(mask_locations[:, 1]).astype(int), np.round(mask_locations[:, 0]).astype(int)]
                gt_labels_positions_dict[label_id] = mask_locations
                gt_3d_positions_dict[label_id] = np.hstack([mask_locations, depth_val[:, None]])

        return gt_labels_positions_dict, gt_3d_positions_dict


    def get_image_semlabels(self, idx, subsampling_factor: float = 2., camera_id: Optional[int] = 2):

        image_semlabel_dir = os.path.join(self.sequence_folder, f'image_labels_{camera_id}')

        label_map = LabelConfig.label_map()

        gt_labels_positions_dict = dict()
        for label, label_id in label_map.items():
            label_img_file = os.path.join(image_semlabel_dir, label, f'{str(idx).zfill(6)}.png')
            label_img = cv2.imread(label_img_file)

            subsampled_height = int(round(label_img.shape[0] / subsampling_factor))
            subsampled_width = int(round(label_img.shape[1] / subsampling_factor))

            # The trick to subsample is to get them from the small image.
            label_img_subsampled = cv2.resize(label_img, (subsampled_width, subsampled_height))

            # Height width to width height
            mask_locations = np.argwhere(label_img_subsampled > 0)[:, :2][:, ::-1]

            # Revert it back to its original size.
            mask_locations = mask_locations * subsampling_factor

            gt_labels_positions_dict[label_id] = mask_locations

        return gt_labels_positions_dict

    def project_semlabels(self,
                          idx: int,
                          camera_id: int = 2,
                          rotation_degrees: Optional[Dict[str, Any]] = None,
                          translation_meters: Optional[Dict[str, Any]] = None):
        assert camera_id in [0, 1, 2, 3]

        lidar_data: np.ndarray = self.load_lidar_points(idx)
        label_data: np.ndarray = self.load_lidar_labels(idx)
        image_data: np.ndarray = self.load_image(idx, camera_id)

        height, width, _ = image_data.shape

        if rotation_degrees is not None:
            # ROLL      YAW       PITCH
            z_degrees, y_degrees, x_degrees = rotation_degrees['z'], rotation_degrees['y'], rotation_degrees['x']
            rotation = Rotation.from_euler('zyx', np.squeeze(np.array([z_degrees, y_degrees, x_degrees])), degrees=True)
            random_rotation_matrix = rotation.as_matrix()
        else:
            random_rotation_matrix = np.eye(3)

        if translation_meters is not None:
            z, y, x = translation_meters['z'], translation_meters['y'], translation_meters['x']
            random_translation = np.array([x, y, z])
        else:
            random_translation = np.zeros(3, dtype=float)

        # Don't modify the master variable.
        tr = self.tr.copy()

        # Apply rotation
        tr[:, :3] = random_rotation_matrix @ tr[:, :3]

        # Apply translation
        tr[:, 3] = tr[:, 3] + random_translation

        pts_2d = self.project_lidar_to_image_coordinates(lidar_data, camera_id, tr)
        fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
                   (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)

        # Filter Negative depth.
        fov_inds = fov_inds & (lidar_data[:, 0] > 2)
        pts_2d_fov = pts_2d[fov_inds, :]
        pts_label_fov = label_data[fov_inds]

        label_img = np.zeros((image_data.shape[0], image_data.shape[1]))
        window = 12

        for i in range(pts_2d_fov.shape[0]):
            pt_2d = pts_2d_fov[i]
            label = pts_label_fov[i]
            i, j = int(pt_2d[1]), int(pt_2d[0])
            start_i, end_i = max(i - window, 0), min(i + window, height - 1)
            start_j, end_j = max(j - window, 0), min(j + window, width - 1)

            # Road 40, Sidewalk 48, Cars 10
            if label in LabelConfig.labels():
                label_img[start_i: end_i, start_j: end_j] = label

        # plt.imshow(label_img)
        # plt.title('Label Image')
        # plt.show()

        return label_img, pts_2d_fov, pts_label_fov, tr, lidar_data

    def __len__(self):
        velodynes = glob.glob(os.path.join(self.sequence_folder, 'velodyne/*'))
        return len(velodynes)


class LabelMap:
    ROAD = 40
    SIDEWALK = 48
    CARS = 10

    pred_color_dict = {40: (255, 0, 0), 48: (0, 255, 0), 10: (0, 0, 255)}
    gt_color_dict = {40: (255, 255, 0), 48: (0, 255, 255), 10: (255, 0, 255)}

def convert_to_chamfer_input(pts_2d):
    # N x 2 to N x 3
    pts_2d_trailing_zeros = np.hstack([pts_2d, np.zeros([pts_2d.shape[0], 1])])
    return torch.from_numpy(pts_2d_trailing_zeros)


def get_initialized_input(kitti, idx, camera_id=2, num_iter: int = 1, rotation_degrees=None):
    if rotation_degrees is None:
        rotation_degrees = dict(z=15, y=5, x=5)

    img_label, pts_2d_fov, pts_label_fov, i, lidar_data = kitti.get_gt_projection(idx=idx, camera_id=camera_id)
    min_loss = np.inf

    for i in range(num_iter):
        print(f"Generating candidate at index {idx}, iteration {i}")

        img_label_noisy, pts_2d_fov_noisy, pts_label_fov_noisy, extrinsics_to_refine, _ = kitti.get_rotated_projection(
            idx=idx, camera_id=camera_id, rotation_degrees=rotation_degrees)

        total_chamdist = 0
        for label in [LabelMap.ROAD, LabelMap.SIDEWALK, LabelMap.CARS]:
            gt_pts_fov = pts_2d_fov[pts_label_fov == label]
            pred_pts_fov = pts_2d_fov_noisy[pts_label_fov_noisy == label]

            gt_pts_fov_torch = convert_to_chamfer_input(gt_pts_fov)[None, :].float()
            pred_pts_fov_torch = convert_to_chamfer_input(pred_pts_fov)[None, :].float()
            chamdist, _ = chamfer_distance(gt_pts_fov_torch, pred_pts_fov_torch)

            total_chamdist += chamdist

        if total_chamdist < min_loss:
            min_loss = total_chamdist
            init_pred_pts_2d_fov = pts_2d_fov_noisy
            init_pred_pts_label_fov = pts_label_fov_noisy
            init_extrinsics = extrinsics_to_refine

    np.save(f'../data/init_image_points.npy', init_pred_pts_2d_fov)
    np.save(f'../data/init_label_points.npy', init_pred_pts_label_fov)
    np.save(f'../data_old/init_extrinsics.npy', init_extrinsics)
    np.save(f'../data/init_lidar.npy', lidar_data)

    return init_pred_pts_2d_fov, init_pred_pts_label_fov, init_extrinsics, lidar_data


