import os
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path
from PIL import Image
import math
import mvtec3d_util as mvt_util
import argparse
import render_utils_for_ground_truth
import glob
import shutil
import cv2
import time


def get_specific_resolution_pcd_rgb(organized_pc, rgb_image, width, height):
    resized_organized_pc = mvt_util.resize_organized_pc(organized_pc, target_height=height, target_width=width, tensor_out=False)
    resized_rgb_image = cv2.resize(rgb_image, (height, width))

    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(resized_organized_pc)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    o3d_pc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    none_zero_rgb = cv2.cvtColor(resized_rgb_image, cv2.COLOR_BGR2RGB)
    none_zero_rgb = none_zero_rgb.reshape(resized_rgb_image.shape[0] * resized_rgb_image.shape[1], resized_rgb_image.shape[2])
    none_zero_rgb = none_zero_rgb[nonzero_indices, :] / 255.

    return o3d_pc, resized_rgb_image, none_zero_rgb

# ALL DATA IS RESIZED TO 224
def preprocess_pc(multi_view_vis:render_utils_for_ground_truth.MultiViewRender, tiff_path, root_path, save_path):
    # READ FILES
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")

    gt_exists = os.path.isfile(gt_path)
    if not gt_exists:
        return

    gt_image = cv2.imread(gt_path)
    gt_image[gt_image>0] = 255
    # READ Point Cloud
    organized_pc = mvt_util.read_tiff_organized_pc(tiff_path)
    # calculate fpfh features and render multi-view images
    rgb_image = cv2.imread(rgb_path)

    _224_pcd, _224_rgb, _224_nonezero_rgb = get_specific_resolution_pcd_rgb(organized_pc, rgb_image, width=224, height=224)
    _ori_reso_pcd, _ori_reso_gt, _ori_nonezero_gt = get_specific_resolution_pcd_rgb(organized_pc, gt_image, gt_image.shape[0], gt_image.shape[1])
    _ori_nonezero_gt = _ori_nonezero_gt[:, 0]
    non_zero_number = len(_ori_nonezero_gt)
    origin_colors = [[0.5, 0.5, 0.5]] * non_zero_number
    origin_colors = np.array(origin_colors)
    gt_colors = origin_colors.copy()
    gt_colors[_ori_nonezero_gt > 0.5, :] = (1., 0, 0)  # RGB


    _ = multi_view_vis.calculate_fpfh_features(_ori_reso_pcd)
    gt_images, _ = multi_view_vis.multiview_render(_ori_reso_pcd, gt_colors, _224_pcd)
    ori_images, _ = multi_view_vis.multiview_render(_ori_reso_pcd, origin_colors, _224_pcd)

    # calculate the root and file paths

    tiff_spilt = os.path.split(tiff_path.replace(root_path, save_path))
    save_xyz_root = os.path.join(tiff_spilt[0], tiff_spilt[1][:-5])

    # MAKE DIRS
    os.makedirs(save_xyz_root, exist_ok=True)

    for idx, (gt_image, ori_image) in enumerate(zip(gt_images, ori_images)):

        # for visualization, check whether we get right results
        # viz_image = render_utils.draw_points3d_on_image(image.astype(np.uint8), point)
        # cv2.imshow('viz1', image.astype(np.uint8))
        # cv2.imshow('viz', viz_image)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(save_xyz_root, f"view_{idx:03d}_gt.png"), gt_image)
        cv2.imwrite(os.path.join(save_xyz_root, f"view_{idx:03d}_ori.png"), ori_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MVTec 3D-AD')
    parser.add_argument('--dataset_path', type=str,
                        default='../../datasets/mvtec_3d_anomaly_detection',
                        help='The root path of the MVTec 3D-AD. The preprocessing is done inplace (i.e. the preprocessed dataset overrides the existing one)')
    parser.add_argument('--color-option', type=str,
                        default='GT', choices=['GT'],
                        help='in [X,Y,Z,NORM,FPFH]')
    parser.add_argument('--category', type=str, default='dowel')
    parser.add_argument('--save-dir', type=str,
                        default='../../datasets/multi_view_gt_mvtec_3d_anomaly_detection',
                        help='The save path for the generated multi-view-mvtec3d dataset')

    # NOTE: You should run the preprocessing.py first

    args = parser.parse_args()
    #
    # mvtec3d_classes = [
    #     "bagel",
    #     "cable_gland",
    #     "carrot",
    #     "cookie",
    #     "dowel",
    #     "foam",
    #     "peach",
    #     "potato",
    #     "rope",
    #     "tire",
    # ]
    name = args.category
    #
    # mvtec3d_classes = [
    #     "bagel"
    # ]

    color_option_dict = {'GT': render_utils_for_ground_truth.MultiViewRender.COLOR_GT
                         }

    dataset_path = args.dataset_path
    color_option = color_option_dict[args.color_option]
    save_dir = args.save_dir

    split = ['test']

    processed_number = 0
    # parameter_path = os.path.join(dataset_path, name, 'calibration', 'camera_parameters.json')
    multi_view_vis = render_utils_for_ground_truth.MultiViewRender('', color=color_option)


    print(f'processing {name}...')

    time_begin = time.time()
    for s in split:
        img_path = os.path.join(dataset_path, name, s)

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(img_path)

        for defect_type in defect_types:

            tiff_paths = glob.glob(os.path.join(img_path, defect_type, 'xyz') + "/*.tiff")
            tiff_paths.sort()

            for tiff_path in tiff_paths:
                preprocess_pc(multi_view_vis, tiff_path, dataset_path, save_dir)

                processed_number += 1
                if processed_number % 50 == 0:
                    cur_time = time.time()
                    print(f"Processed {processed_number} tiff files... using {cur_time - time_begin:.2f} s...")


