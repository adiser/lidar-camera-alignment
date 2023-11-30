import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.loss import chamfer_distance
from torch import nn

from calibration import Kitti, get_initialized_input
from config import ExperimentConfig
from models.utils import get_model
from utils import LabelConfig, DOWNSCALING_FACTOR
import seaborn as sns


def main():
    camera_id = 2

    kitti = Kitti()
    image_height = kitti.height
    image_width = kitti.width
    projection_matrix = torch.from_numpy(kitti.Ps[camera_id]).float()

    experiment_config = ExperimentConfig(
        # Initial model.
        rot_param_type='euler',
        lr=1e-6,
        random_rotation=dict(z=5, y=2.5, x=2.5)
    )

    # Parse hyperparameters.
    rot_param_type = 'euler'
    lr = 1e-6
    rotation_degrees = dict(z=5, y=2.5, x=2.5)
    num_iter = 1000
    image_labels_subsampling_factor = 5.
    depth_scaling_factor = 1.
    use_depth = True

    for i in range(len(kitti)):
        # This will act as the image seglabels for now
        # if os.path.exists(f'../data/init_extrinsics_{i}.npy'):
        if False:
            init_pred_pts_2d_fov = np.load(f'../data/init_image_points_{i}.npy')
            init_pred_pts_label_fov = np.load(f'../data/init_label_points_{i}.npy')
            init_extrinsics = np.load(f'../data/init_extrinsics_{i}.npy')
            lidar_data = np.load(f'../data/init_lidar_{i}.npy')
        else:
            init_pred_pts_2d_fov, init_pred_pts_label_fov, init_extrinsics, lidar_data = get_initialized_input(
                kitti, idx=i, camera_id=camera_id, rotation_degrees=rotation_degrees)

        # label_img, pts_2d_fov_image, pts_label_fov_image, _, _ = kitti.get_gt_projection(idx=i, camera_id=camera_id)

        label_data = kitti.load_lidar_labels(idx=i)
        label_data = torch.from_numpy(label_data.astype(np.int32))

        # Initialize the model
        calibrator_model = get_model(init_extrinsics=init_extrinsics, rot_param_type=rot_param_type)
        calibrator_model.train()
        optimizer = torch.optim.SGD(calibrator_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        lidar_data = torch.from_numpy(lidar_data)
        is_changed_model = False

        # Get the labels without depth
        # gt_labels_dict = kitti.get_image_semlabels(idx=i, subsampling_factor=5., camera_id=camera_id)

        # Get the labels with depth
        gt_labels_dict, gt_3d_points_position = kitti.get_image_semlabels_with_depth(
            idx=i, subsampling_factor=image_labels_subsampling_factor, camera_id=camera_id,
            depth_scaling_factor=depth_scaling_factor
        )

        for iteration in range(num_iter):

            image = kitti.load_image(i, camera_id).copy()[:, :, ::-1]

            pred_points_dict = calibrator_model(lidar_data,
                                                projection_matrix,
                                                image_height,
                                                image_width,
                                                label_data,
                                                add_depth=use_depth,
                                                depth_scaling_factor=depth_scaling_factor)

            total_chamdist = 0

            # BGR to RGB
            for rel_label in LabelConfig.labels():

                if use_depth:
                    pred_points_position = pred_points_dict[rel_label]
                    gt_points_position = gt_3d_points_position[rel_label]

                    def analyze_depth_position():
                        depth_pred = pred_points_position[:, 2]
                        depth_gt = gt_points_position[:, 2]

                        fig, ax = plt.subplots()

                        sns.set_theme()  # <-- This actually changes the look of plots.
                        ax.hist([depth_pred.detach().cpu().numpy(), depth_gt], color=['r', 'b'], alpha=0.5)
                        # plt.hist([depth_pred.detach().cpu().numpy(), depth_gt], color=['r', 'b'], alpha=0.5)
                        plt.title("Gt Blue, Pred Red")
                        plt.show()

                    # analyze_depth_position()
                    gt_points_position = torch.from_numpy(gt_points_position).float()
                else:
                    pred_points_position = pred_points_dict[rel_label]
                    gt_points_position = gt_labels_dict[rel_label]
                    gt_points_position = torch.from_numpy(gt_points_position).float()

                    # Add trailing zeros.
                    pred_points_position = torch.hstack([pred_points_position, torch.zeros([pred_points_position.shape[0], 1])])
                    gt_points_position = torch.hstack([gt_points_position, torch.zeros([gt_points_position.shape[0], 1])])

                # TODO: Visualize whatever the fuck is happening.
                def draw_points():
                    pred_color = LabelConfig.color(rel_label)[::-1]
                    gt_color = LabelConfig.color(rel_label)

                    image[pred_points_position[:, 1].int(), pred_points_position[:, 0].int(), :] = np.array(pred_color)
                    image[gt_points_position[:, 1].int(), gt_points_position[:, 0].int(), :] = np.array(gt_color)

                draw_points()

                # Add batch size
                pred_points_position = pred_points_position[None, :]
                gt_points_position = gt_points_position[None, :]

                chamdist, _ = chamfer_distance(gt_points_position, pred_points_position, single_directional=False)
                total_chamdist = total_chamdist + chamdist

            # Scaling factor is configurable
            total_chamdist = total_chamdist / DOWNSCALING_FACTOR

            # Plotting.
            plt.imshow(image)
            cv2.imwrite(f'../data/index_{i}/iteration_{iteration}.png', image[:, :, ::-1])

            print("written")

            # Backprop
            loss = criterion(total_chamdist, torch.tensor(0).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Change to arbitrary parameterization if we come close enough
            if loss.item() < .2 and is_changed_model == False:
                # Get the extrinsics matrix at that (continued) time step.
                init_extrinsics = calibrator_model.construct_extrinsics_matrix()
                calibrator_model = get_model(init_extrinsics=init_extrinsics, rot_param_type='quaternion')
                calibrator_model.train()
                optimizer = torch.optim.SGD(calibrator_model.parameters(), lr=1e-6)
                is_changed_model = True
                
            # Logging.
            print(f"Iteration {iteration}: {loss.item()}, {calibrator_model.rot_param}")


if __name__ == "__main__":
    main()
