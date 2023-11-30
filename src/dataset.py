from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from calibration import Kitti, get_initialized_input


def collate_points(pcd_list: List[torch.Tensor]):
    max_len = max([len(pcd) for pcd in pcd_list])
    dtype = pcd_list[0].dtype
    dim = pcd_list[0].shape[1]
    batch_size = len(pcd_list)
    pcd_arr = torch.zeros([batch_size, max_len, dim], dtype=dtype)
    for b, pcd in enumerate(pcd_list):
        pcd_numpy = np.resize(np.array(pcd.detach().cpu().numpy()), (max_len, dim))
        pcd_torch = torch.from_numpy(pcd_numpy)
        pcd_arr[b, :, :] = pcd_torch
    return pcd_arr


class KittiDataset(Dataset):
    def __init__(self,
                 kitti: Kitti = Kitti(),
                 camera_id: int = 2,
                 image_labels_subsampling_factor: float = 5.,
                 depth_scaling_factor = 1.):
        self.kitti = kitti
        self.camera_id = camera_id
        self.image_labels_subsampling_factor = image_labels_subsampling_factor
        self.depth_scaling_factor = depth_scaling_factor

    def get_init_extrinsics(self):
        _, _, _, tr, _ = self.kitti.get_rotated_projection(idx=0, camera_id=2)
        return tr

    @property
    def projection_matrix(self):
        projection_matrix = torch.from_numpy(self.kitti.Ps[self.camera_id]).float()
        return projection_matrix

    def collate_fn(self, batch_list):

        image_seg_label_data_list: List[Dict] = [b[2] for b in batch_list]
        classes = image_seg_label_data_list[0].keys()

        pcd_list = [b[0] for b in batch_list]
        pcd_arr = collate_points(pcd_list)

        lidar_seg_label_data_list = [b[1][:, None] for b in batch_list]
        label_arr = collate_points(lidar_seg_label_data_list).squeeze()

        image_seg_label_pos_dict =  dict()
        for class_name in classes:
            class_labels = [label_dict[class_name] for label_dict in image_seg_label_data_list]
            class_labels_arr = collate_points(class_labels)
            image_seg_label_pos_dict[class_name] = class_labels_arr

        return pcd_arr, label_arr, image_seg_label_pos_dict

    def __len__(self):
        return 1600  # Only got depth image up to this point 1600

    def __getitem__(self, idx):

        # Get the initialized LIDAR position
        lidar_data = self.kitti.load_lidar_points(idx)
        lidar_data = torch.from_numpy(lidar_data)

        # Get the labels.
        label_data = self.kitti.load_lidar_labels(idx=idx)
        label_data = torch.from_numpy(label_data.astype(np.int32))

        # Get the image labels as a target.
        _, gt_3d_points_position = self.kitti.get_image_semlabels_with_depth(
            idx=idx, subsampling_factor=self.image_labels_subsampling_factor, camera_id=self.camera_id,
            depth_scaling_factor=self.depth_scaling_factor
        )

        for key in gt_3d_points_position:
            gt_3d_points_position[key] = torch.from_numpy(gt_3d_points_position[key])

        return lidar_data, label_data, gt_3d_points_position
