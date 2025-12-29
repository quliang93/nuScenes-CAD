# nuScenes-CAD post_processing
# By: 
# Date: 2025/8/28

import os
import cv2
import json
import tqdm
import shutil
import struct
import numpy as np


def len_subfiles(path: str) -> int:
    """
    Return: 
            length of sub-directory.
    """
    return len(os.listdir(path))


def parse_label(path: str, border_points_num: int):
    """
    Parse label into a np.ndarray
    """
    f = open(path, 'rb')
    data = f.read()
    label = np.array(struct.unpack(str(border_points_num) + 'd', data))
    f.close()
    return label


def visualize_pred(seg_img, label):
    """
    Render CAD-label on segmentation image.
    """
    real_range = np.asarray([0, -3.141592653589793, -2, 20, 3.141592653589793, 0.5])
    grid_size = np.asarray([128, 384, 1])
    unit_size = (real_range[3:] - real_range[:3]) / grid_size

    h, w, c = seg_img.shape
    resolution = 40 / 200
    center = [w/2-0.5, h/2-0.5]
    phi_start = real_range[1] + unit_size[1] / 2

    phi_real = phi_start + unit_size[1] * np.arange(0, grid_size[1])
    x_real = label * np.cos(phi_real)
    y_real = label * np.sin(phi_real)
    x_pix = (x_real / resolution + center[0]).astype(int)
    y_pix = (h - 1 - (y_real / resolution + center[1])).astype(int)
    points = np.stack([x_pix, y_pix], 0).T
    res = cv2.fillPoly(seg_img, [points], color=(241, 105, 128))

    # for p in points:
    #     cv2.line(seg_img, (100, 100), (p[0], p[1]), (128, 90, 128))
    

    return res


if __name__ == "__main__":
    dataset_path = "/path/to/nuScenes-CAD-scene-split/"# "../nuScenes-CAD/"
    scene_dirs = os.listdir(dataset_path)
    print(f"Scene number: {len(scene_dirs) - 1}")
    
    camera_name = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

    CAD_LABEL_TOTAL_PATH = os.path.join(dataset_path, "CAD_LABEL")
    CAD_LABEL_DIR = os.path.join(CAD_LABEL_TOTAL_PATH, "label")
    CAD_RENDER_DIR = os.path.join(CAD_LABEL_TOTAL_PATH, "render")
    
    if not os.path.exists(CAD_LABEL_DIR):
        os.makedirs(CAD_LABEL_DIR)
    if not os.path.exists(CAD_RENDER_DIR):
        os.makedirs(CAD_RENDER_DIR)

    with open(os.path.join(CAD_LABEL_TOTAL_PATH, "scene_id2sample_id.json"), 'r', encoding='utf-8') as f:
        scene_id2sample_id = json.load(f)

    total_label_num = 0

    for scene in scene_dirs:
        if scene == "CAD_LABEL":
            # CAD_LABEL infomation prepare.
            continue
                
        scene_path = os.path.join(dataset_path, scene)
        scene_camera_path = os.path.join(scene_path, "camera")
        scene_extrinsics_path = os.path.join(scene_path, "extrinsics")
        scene_intrinsics_path = os.path.join(scene_path, "intrinsics")
        scene_label_path = os.path.join(scene_path, "label")
        scene_laser_path = os.path.join(scene_path, "laser")
        scene_lidar_path = os.path.join(scene_path, "lidar")
        scene_pcd_path = os.path.join(scene_path, "pcd")
        scene_pose_path = os.path.join(scene_path, "pose")
        scene_seg_img_path = os.path.join(scene_path, "seg_img")

        # Check data items except images
        if os.path.exists(scene_laser_path) and os.path.exists(scene_lidar_path):
            assert len_subfiles(scene_extrinsics_path) == len_subfiles(scene_intrinsics_path) \
                == len_subfiles(scene_label_path) == len_subfiles(scene_laser_path) == len_subfiles(scene_lidar_path) \
                    == len_subfiles(scene_pcd_path) == len_subfiles(scene_pose_path) == len_subfiles(scene_seg_img_path), print(f"current scene: {scene} error !")

        for cam in camera_name:
            cam_path = os.path.join(scene_camera_path, cam)
            assert len_subfiles(cam_path) == len_subfiles(scene_label_path), print(f"Camera: {cam} Error in Scene: {scene} !")

        total_label_num += len_subfiles(scene_label_path)

        # Delete the lidar directory !
        if os.path.exists(scene_lidar_path):
            shutil.rmtree(scene_lidar_path)
        
        # Delete the laser directory !
        if os.path.exists(scene_laser_path):
            shutil.rmtree(scene_laser_path)

        # Generate the rendered RIGHT-CAD LABEL image in each scene's render directory
        scene_render_path = os.path.join(scene_path, "render")
        if not os.path.exists(scene_render_path):
            os.makedirs(scene_render_path)

        # we loop all files in label
        for file in os.listdir(scene_label_path):
            print(file)
            # load label file
            label_file_path = os.path.join(scene_label_path, file)
            label = parse_label(label_file_path, 384)
                       
            # load seg_img file
            seg_path = os.path.join(scene_seg_img_path, file.split(".")[0]+".png") # seg_file
            seg = cv2.imread(seg_path)

            # Acquire rendered seg-image !
            res = visualize_pred(seg, label)

            # store rendered image in current scene's render directory !
            cv2.imwrite(os.path.join(scene_render_path, file.split(".")[0]+".png"), res)

            scene_id = scene + "_" + file.split(".")[0]
            
            # print(scene_id in scene_id2sample_id.keys())
            assert scene_id in scene_id2sample_id.keys(), f"Error in {scene}, file ID: {file.split('.')[0]} !"
            sample_id = scene_id2sample_id[scene_id]

            # copy label to CAD_LABEL/label
            shutil.copy(label_file_path, os.path.join(CAD_LABEL_DIR, sample_id+".data"))

            # imwrite rendered image to CAD_LABEL/render
            cv2.imwrite(os.path.join(CAD_RENDER_DIR, sample_id+".png"), res)

    print(f"Total labels collected: {total_label_num}")
    print(f"Total nuScenes-CAD valid data CAD items: {len(scene_id2sample_id.keys())}")
    print(f"Dataset Checked ! No Error !")