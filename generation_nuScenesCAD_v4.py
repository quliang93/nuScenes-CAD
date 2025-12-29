# Cross-Frame Instance Augmentation
# Date: 2025/7/16
# By: 

import os
import cv2
import json
import torch
import struct
import argparse
import shutil
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.sparse import coo_matrix
from nuscenes.map_expansion.map_api import NuScenesMap


def static_map_generation(nusc: NuScenes, test_scene: dict):
    """ A semantic global mapping for a nuscene-style scene
    Args:
        nusc (NuScenes): nuscence official object of NuScenes
        test_scene: the scene which needs semantic global mapping 
    Return: 
        map_points (np.ndarray): map points, (N, 3)
        map_labels (np.ndarray): map points labels (N, 1)
    """
    current_sample_token = test_scene['first_sample_token']
    nbr_samples = test_scene['nbr_samples']

    map = []
    map_labels = []
    for _ in range(nbr_samples):
        current_sample = nusc.get('sample', current_sample_token)
        # Get LIDAR_TOP, LIDAR-Seg, ego_pose, calibration
        lidar_sample = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
        lidar_seg_sample = nusc.get('lidarseg', lidar_sample['token'])
        ego_pose = nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        cs_record = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
        lidar_sample_path = os.path.join(nusc.dataroot, lidar_sample['filename'])
        lidar_seg_path = os.path.join(nusc.dataroot, lidar_seg_sample['filename'])

        points, labels = get_lidar_labels(lidar_sample_path, lidar_seg_path)
        ego_trans, ego_rot = get_ego_vehicle_pose(ego_pose)
        l2e_r, l2e_t = cs_record['rotation'], cs_record['translation']

        transformed_points, labels = transform_points_to_map_frame(points, labels, 
                                                                    ego_trans, ego_rot,
                                                                    l2e_t, l2e_r)
        map.append(transformed_points)
        map_labels.append(labels)

        current_sample_token = current_sample['next']
    
    map = np.concatenate(map)
    map_labels = np.concatenate(map_labels)
    
    return map, map_labels


def transform_points_to_target(src_points, src_labels, src_info, target_info):
    """ We transform source frame's instance's points & labels to target instance frame.
    Args:
        src_points  (np.ndarray): points in source frame (N, 3)
        src_labels  (np.ndarray): labels in source frame (N, 1)
        src_info    (np.ndarray): information of source frame
        target_info (np.ndarray): information of target frame
    
    For cross-frame instance augmentation, we need to put the points of the same instance
    in different frame into the target frame's instance's bounding box. Not just the coordinate transformation.       

    Return:
        transformed_points (np.ndarray) :
        src_labels (np.ndarray)         :
    """
    target_trans = np.array(target_info['target_translation'])
    target_rot = np.array(target_info['target_rotation'])

    target_rot_matrix = Quaternion(target_rot).rotation_matrix
    target_rot_inv = np.linalg.inv(target_rot_matrix)

    src_trans = np.array(src_info['translation'])
    src_rot = np.array(src_info['rotation'])
    src_rot_inv = np.linalg.inv(Quaternion(src_rot).rotation_matrix)

    # Do the transformation ...
    # points_shifted = src_points - (src_trans - target_trans)
    # transformed_points = target_rot_inv @ points_shifted.T

    points_shifted_local = (src_rot_inv @ (src_points - src_trans).T).T
    transformed_points = (target_rot_matrix @ points_shifted_local.T).T + target_trans
    
    return transformed_points, src_labels


def get_lidar_labels(lidar_path, lidar_seg_path, filter= True):
    """ Get Lidar & segmentation-labels
    Args
        lidar_path (string)     : lidar point cloud path (sample)
        lidar_seg_path (string) : lidar point cloud segmentation path
        filter (bool)           : whether to filter target class
    
    Return:
        points (np.ndarray) : (N, 3)
        labels (np.ndarray) : (N, 1)
    """
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    labels = np.fromfile(lidar_seg_path, dtype= np.uint8).reshape(-1, 1)

    if filter:
        unwanted_classes = {
            0,  # noise
            1,  # animal
            2, 3, 4, 5, 6, 7, 8,  # human.*
            9, 10, 11, 12, # moveable
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 31,  # vehicle.* & ego vehcle.
        }
        mask = ~np.isin(labels.flatten(), list(unwanted_classes))
        points = points[mask]
        labels = labels[mask]

    return points[:, :3], labels


def get_ego_vehicle_pose(ego_pose_sample: dict):
    """ Acquire ego vehicle's pose
    Args:
        ego_pose_sample (dict) : ego_pose got from sample
    Return:
        trans (np.ndarray): ego vehicle's pose position, (x,y,z)
        rot (np.ndarray)  : ego vehicle's pose quaternion (qx, qy, qz, qw)
    """
    trans = ego_pose_sample['translation']
    rot = ego_pose_sample['rotation']
    # print(f"trans: {trans}, rot: {rot}")
    return np.array(trans), np.array(rot)


def transform_points_to_map_frame(cur_points, # current frame lidar points
                                      cur_labels, # current frame lidar points labels
                                      cur_trans,   # current frame ego trans
                                      cur_rot,     # current frame ego rot
                                      cur_l2e_trans,
                                      cur_l2e_rot
                                      ):
    """ Transform points from LiDAR -> globl map frame.
    points(Global Map) <----(ego_pose)----- points(Ego Vehicle) <-----(Extrinsics)---- points(LiDAR)

    Args:
        cur_points (np.ndarray)    : points in LiDAR frame (N, 3),
        cur_points (np.ndarray)    : labels
        cur_trans (np.ndarray)     : ego vehicle's pos (3, ) x, y, z
        cur_rot (np.ndarray)       : ego vehicle's rot (4, )
        cur_l2e_trans (np.ndarray) : lidar -> ego vehicle's pos (3, )
        cur_l2e_rot (np.ndarray)   : lidar -> ego vehicle's rot (4, )

    Return:
        pc_map (np.ndarray)        : points in map frame (N, 3)
        cur_labels (np.ndarray)    : points labels (N, 1)
    """
    rot_e2g_cur = Quaternion(cur_rot).rotation_matrix
    rot_l2e_cur = Quaternion(cur_l2e_rot).rotation_matrix
    trans_e2g_cur = np.array(cur_trans)
    trans_l2e_cur = np.array(cur_l2e_trans)

    pc2_ego = (rot_l2e_cur @ cur_points.T).T + trans_l2e_cur
    pc_map = (rot_e2g_cur @ pc2_ego.T).T + trans_e2g_cur
    
    return pc_map, cur_labels


def segscene_g2e(current_scene: np.ndarray, current_scene_labels, ego_trans, ego_rot, range_min=[-20, -20, -1], range_max=[20, 20, 3]):
    """
    Transform current global dense-scene to current ego vehicle's frame.
    current_scene: (N, 3)
    current_scene_labels: (N, 1)
    ego_trans: (3, )
    ego_rot: (4, ) Quaternion
    range_min: x/y/z min bound in ego-vehicle's frame
    range_max: x/y/z max bound in ego-vehicle's frame
    """
    range_min = np.array(range_min)
    range_max = np.array(range_max)
    
    shifted_points = current_scene - ego_trans
    rot_inv = np.linalg.inv(Quaternion(ego_rot).rotation_matrix)
    transformed_points = (rot_inv @ shifted_points.T).T

    mask = (
        (transformed_points[:, 0] >= range_min[0]) & (transformed_points[:, 0] <= range_max[0]) &
        (transformed_points[:, 1] >= range_min[1]) & (transformed_points[:, 1] <= range_max[1]) &
        (transformed_points[:, 2] >= range_min[2]) & (transformed_points[:, 2] <= range_max[2])
    )
    transformed_points = transformed_points[mask]
    transformed_labels = current_scene_labels[mask]

    return transformed_points, transformed_labels


def segimg_acc(ego_scene: np.ndarray, ego_scene_labels: np.ndarray, 
           lidarseg_idx2name_mapping: dict, color_map: dict, 
           sample_idx: int, resolution: list = [0.2, 0.2]) -> tuple:
    """ Painting the local ego vehicle's BEV map
    Args
        ego_scene (np.ndarray)       : (N, 3), local points in ego vehicle's frame
        ego_scene_labels (np.ndarray): (N, 1), points' labels
        lidarseg_idx2name_mapping: cls_id -> name
        color_map: name -> color(r, g, b)
        sample_idx : sample idx in a given scene
        resolution : x/y resolution for map generation
    Return:
        None. Just store the BEV map in target directory.
    """
    
    ego_scene = np.asarray(ego_scene, dtype=np.float32)
    ego_scene_labels = np.asarray(ego_scene_labels, dtype=np.int32).flatten()
    
    dx, dy = resolution

    range_min = np.array([-20, -20], dtype=np.float32)
    range_max = np.array([20, 20], dtype=np.float32)
    grid_w, grid_h = np.array((range_max - range_min) / [dx, dy], dtype=np.int32)   
    
    grid_indices = np.floor((ego_scene[:, :2] - range_min) / [dx, dy]).astype(np.int32)
    valid_mask = (
        (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < grid_w) &
        (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < grid_h)
    )
    grid_indices = grid_indices[valid_mask]
    valid_labels = ego_scene_labels[valid_mask]
    
    priority_labels = set(range(2, 9)).union(set(range(14, 24)).union({31}))
    
    seg_indices = np.full((grid_h, grid_w), 32, dtype=np.int32)  # 默认虚拟类别 32（白色背景)
    for label in np.unique(valid_labels):
        mask = valid_labels == label
        if not np.any(mask):
            continue
        data = np.ones(np.sum(mask), dtype=np.int32)
        row = grid_indices[mask, 1]
        col = grid_indices[mask, 0]
        hist = coo_matrix((data, (row, col)), shape=(grid_h, grid_w)).toarray()
        
        if label in priority_labels:
            seg_indices[hist > 0] = label
        else:
            mask_non_priority = (seg_indices == 32) & (hist > 0)
            if 'counts' not in locals():
                counts = hist
                seg_indices[mask_non_priority] = label
            else:
                mask_update = (hist > counts) & mask_non_priority
                seg_indices[mask_update] = label
                counts[mask_update] = hist[mask_update]
    
    color_array = np.array([color_map[lidarseg_idx2name_mapping.get(i, 'noise')][::-1] for i in range(32)] + 
                           [(255, 255, 255)], dtype=np.uint8)  # 添加白色 (BGR) 作为类别 32
    
    seg_image = np.full((grid_h, grid_w, 3), (255, 255, 255), dtype=np.uint8)
    seg_map = np.full((grid_h, grid_w, 3), (255, 255, 255), dtype= np.uint8)

    for y in range(grid_h):
        seg_image[grid_h-1-y] = color_array[seg_indices[y]]
    
    # output_path = f'scene0/{str(sample_idx).zfill(6)}.png'
    # cv2.imwrite(output_path, seg_image)
    # cv2.imwrite(f"{str(sample_idx).zfill(4)}_map.png", seg_map)
    return seg_image
    

class NuScenesInstance:
    def __init__(self, nusc: NuScenes, instance_info: dict):
        self.nusc = nusc
        self.category_name = instance_info['category_name']
        sample=nusc.get('sample', instance_info['sample_token'])
        self.lidar_sample = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        self.lidar_seg_sample = nusc.get('lidarseg', self.lidar_sample['token'])
        self.ego_vehicle_pose = nusc.get('ego_pose', self.lidar_sample['ego_pose_token'])
        self.calib_record = nusc.get('calibrated_sensor', self.lidar_sample['calibrated_sensor_token'])

        size_factor = np.array([1.5, 1.2, 1.2])
        self.size = np.array(instance_info['size']) * size_factor # w, l, h
        self.center = np.array(instance_info['translation']) # map frame, instance's translation
        self.rotation = np.array(instance_info['rotation']) # map frame, instance's rotation quaternion

    def get_sample_points_labels(self, filter_cls=True):
        """
        We only keep points belong to the same category with the target instance
        Return (filter by class):
            points: (N, 3)
            labels: (N, 1)
        """
        name2idx = self.nusc.lidarseg_name2idx_mapping
        lidar_sample_path = os.path.join(self.nusc.dataroot, self.lidar_sample['filename'])
        lidar_seg_sample_path = os.path.join(self.nusc.dataroot, self.lidar_seg_sample['filename'])
        points = np.fromfile(lidar_sample_path, dtype= np.float32).reshape(-1, 5)
        labels = np.fromfile(lidar_seg_sample_path, dtype=np.uint8).reshape(-1, 1)
        
        target_cls_id = name2idx[self.category_name]
        
        if filter_cls:
            mask = (labels==target_cls_id).flatten()
            points = points[mask]
            labels = labels[mask]
        assert points.shape[0] == labels.shape[0]
        
        return points[:, :3], labels


    def get_ego_pose(self, ):
        """
        Return the instance correspoding sample's ego-vehicle pose (trans, rot)
        """
        trans = self.ego_vehicle_pose['translation']
        rot = self.ego_vehicle_pose['rotation']

        return np.array(trans), np.array(rot)
    

    def get_calib_info(self, ):
        """
        Return the calibration of current sample, (trans, rot)
        """
        l2e_rot, l2e_trans = self.calib_record['rotation'], self.calib_record['translation']
        return np.array(l2e_trans), np.array(l2e_rot)


    def lidarseg_ego2map(self, ):
        """
        Transform points & labels from lidar -> ego_vehicle -> global_map
        """
        # current frame : lidar
        points, labels = self.get_sample_points_labels()

        l2e_trans, l2e_rot = self.get_calib_info()
        e2m_trans, e2m_rot = self.get_ego_pose()

        l2e_rot_matrix = Quaternion(l2e_rot).rotation_matrix
        e2m_rot_matrix = Quaternion(e2m_rot).rotation_matrix

        # Transform points from lidar -> ego_vehicle frame
        pc_ego = (l2e_rot_matrix @ points.T).T + l2e_trans
        # Transform points from ego_vehicle -> map
        pc_map = (e2m_rot_matrix @ pc_ego.T).T + e2m_trans

        return pc_map, labels


    def instance_points_inside_bbox(self, ):
        """
        For current instance object, we only keep the points & labels that inside its current
        bbox, which can filter out the same category points but not belong to this instance.
        """
        # we transform all points which share the same category into map frame.
        scene_cls_points, scene_cls_labels = self.lidarseg_ego2map()
        
        # Do the filter...
        scene_cls_labels = scene_cls_labels.flatten() # (N, )
        # Transform all scene points into the instance's bbox frame for better filter those outside
        rot_matrix = Quaternion(self.rotation).rotation_matrix
        points_local = scene_cls_points - self.center
        points_local = points_local @ rot_matrix.T

        half_size = self.size / 2.0
        min_bounds = -half_size
        max_bounds = half_size
        inside_mask = np.all((points_local>=min_bounds) & (points_local <= max_bounds), axis= 1)

        filtered_points = scene_cls_points[inside_mask]
        filtered_labels = scene_cls_labels[inside_mask]

        return filtered_points, filtered_labels.reshape(-1, 1)


def colored_points_with_bbox(points, labels, idx2name, colormap, instance_info, vis_bbox=False):
    
    labels = labels.flatten()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((points.shape[0], 3))

    for i in range(points.shape[0]):
        label_id = int(labels[i])
        class_name = idx2name[label_id]
        colors[i] = np.array(colormap[class_name]) / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建可视化窗口并显示点云
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Colored Point Cloud")
    vis.add_geometry(pcd)
    
    if vis_bbox:
        center = np.array(instance_info['translation']) # x,y,z
        quaternion = np.array(instance_info['rotation']) # Quaternion
        # size = np.array(instance_info['size']) # length, width, height
        size = np.array([instance_info['size'][1], instance_info['size'][0], instance_info['size'][2]]) # nuscenes size : w,l,h
        rotation_matrix = Quaternion(quaternion).rotation_matrix
        
        # Create bounding box
        corners = np.array([
            [-size[0]/2, -size[1]/2, -size[2]/2],  # 0: 左前下
            [ size[0]/2, -size[1]/2, -size[2]/2],  # 1: 右前下
            [ size[0]/2,  size[1]/2, -size[2]/2],  # 2: 右后下
            [-size[0]/2,  size[1]/2, -size[2]/2],  # 3: 左后下
            [-size[0]/2, -size[1]/2,  size[2]/2],  # 4: 左前上
            [ size[0]/2, -size[1]/2,  size[2]/2],  # 5: 右前上
            [ size[0]/2,  size[1]/2,  size[2]/2],  # 6: 右后上
            [-size[0]/2,  size[1]/2,  size[2]/2],  # 7: 左后上
        ])

        corners = corners @ rotation_matrix.T + center
        
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Connecting edges
        ]
        # Create LineSet for bounding box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set bounding box color (red)
        colors = [[1, 0, 0] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # Add bounding box to visualizer
        vis.add_geometry(line_set)


    # 设置渲染选项
    render_option = vis.get_render_option()
    # render_option.point_size = 2.0  # 设置点的大小
    render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色

    # 运行可视化
    vis.run()
    vis.destroy_window()    


def colored_points(points, labels, idx2name, colormap):
    
    labels = labels.flatten()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((points.shape[0], 3))

    for i in range(points.shape[0]):
        label_id = int(labels[i])
        class_name = idx2name[label_id]
        colors[i] = np.array(colormap[class_name]) / 255.0

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建可视化窗口并显示点云
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Colored Point Cloud")
    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    # render_option.point_size = 2.0  # 设置点的大小
    render_option.background_color = np.array([0, 0, 0])  # 设置背景为黑色

    # 运行可视化
    vis.run()
    vis.destroy_window()    


def main(args):
    nusc = NuScenes(version=args.version, # "v1.0-trainval"
                dataroot= args.source, #"./data/trainval",
                verbose=True)

    nuCAD_dataroot = args.target # "./nuScenes-CAD-scenes-split/"
    CAD_LABEL_PATH = os.path.join(nuCAD_dataroot, "CAD_LABEL") 
    if not os.path.exists(nuCAD_dataroot):
        os.makedirs(nuCAD_dataroot)
        os.makedirs(CAD_LABEL_PATH)

    scene_within_id2sample_id = {}

    scenes = [scene for scene in nusc.scene] # All nuscenes scenes .
    # Check all train & val scenes official name
    all_scenes_name = [s['name'] for s in nusc.scene]
    print(f"We'll deal with all scenes: {all_scenes_name}")

    color_map = nusc.colormap
    # print(f"color_map: {color_map}")
    lidarseg_idx2name_mapping = nusc.lidarseg_idx2name_mapping
    
    print(f"There are total {len(scenes)} scenes !")

    for scene_id, test_scene in enumerate(scenes):
        test_scene_name = test_scene['name'] # official scene name
        print(f"We are dealing with {test_scene_name} Scenarios in NuScenes ...")

        nbr_samples = test_scene['nbr_samples']

        # We acquire a semantic global map for a given scene !
        map, map_labels = static_map_generation(nusc, test_scene)

        # we pick up the first sample
        first_sample_token = test_scene['first_sample_token']
        current_sample_token = first_sample_token

        # 
        camera_name = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        scene_path = os.path.join(nuCAD_dataroot, test_scene_name) 
        scene_lidar_path = os.path.join(scene_path, "lidar")
        scene_camera_path = os.path.join(scene_path, "camera")
        scene_pose_path = os.path.join(scene_path, "pose")
        scene_seg_path = os.path.join(scene_path, "seg_img")
        scene_extrinsics_path = os.path.join(scene_path, "extrinsics")
        scene_intrinsics_path = os.path.join(scene_path, "intrinsics")

        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
            os.makedirs(scene_lidar_path)
            os.makedirs(scene_camera_path)
            os.makedirs(scene_pose_path)
            os.makedirs(scene_seg_path)
            os.makedirs(scene_extrinsics_path)
            os.makedirs(scene_intrinsics_path)
            for camera in camera_name:
                os.makedirs(os.path.join(scene_camera_path, camera))

        for sample_idx in tqdm(range(nbr_samples)):
            # print(f"We are dealing with scene: {scene_id}, sample id: {str(sample_idx).zfill(6)}, sample token: {current_sample_token}")
            current_sample = nusc.get('sample', current_sample_token)
            lidar_sample = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_sample['ego_pose_token']) # ego_pose_token
            ego_trans, ego_rot = get_ego_vehicle_pose(ego_pose)
            all_anns = current_sample['anns']

            current_scene = map
            current_scene_labels = map_labels   

            for target_id, target_anno_token in enumerate(all_anns):
                target_anno_metadata = nusc.get('sample_annotation', target_anno_token)
                target_instance = nusc.get('instance', target_anno_metadata['instance_token'])
                target_instance_first_token = target_instance['first_annotation_token']
                target_instance_last_token = target_instance['last_annotation_token']

                # 
                current_sample_anno_token = target_instance_first_token
                idx= 0
                scene_instance_points = [] # to preseeve cross-frame augmentation points of each instance.
                scene_instance_labels = [] # to preseeve cross-frame augmentation labels of each instance.

                target_sample_anno = nusc.get('sample_annotation', target_anno_token)
                target_frame_info = {
                    "target_sample_token":target_sample_anno['sample_token'],
                    "target_translation": target_sample_anno['translation'],
                    "target_size": target_sample_anno['size'],
                    "target_rotation": target_sample_anno['rotation'],
                    "target_category_name": target_sample_anno['category_name'],
                    'target_num_lidar_pts': target_sample_anno['num_lidar_pts']
                }

                while True:
                    current_sample_anno = nusc.get('sample_annotation', current_sample_anno_token)
                    assert current_sample_anno['instance_token']==target_instance['token']

                    instance_info={
                        'sample_token': current_sample_anno['sample_token'],
                        'translation': current_sample_anno['translation'],
                        'size': current_sample_anno['size'],
                        'rotation': current_sample_anno['rotation'],
                        'category_name': current_sample_anno['category_name'],
                        'num_lidar_pts': current_sample_anno['num_lidar_pts']
                    }

                    # Our processing ...
                    # get the instance's corresponding lidarseg (points-labels) in global frame.
                    instance_object = NuScenesInstance(nusc=nusc, instance_info=instance_info)
                    # scene points
                    # instance_points, instance_labels = instance_object.lidarseg_ego2map()
                    instance_points, instance_labels = instance_object.instance_points_inside_bbox()
                    # transform cross-frame instance points & labels to target instance
                    totarget_points, totarget_labels = transform_points_to_target(instance_points, instance_labels,
                                                                                instance_info, target_frame_info)

                    scene_instance_points.append(totarget_points)
                    scene_instance_labels.append(totarget_labels)

                    if current_sample_anno_token == target_instance_last_token:
                        break

                    current_sample_anno_token = current_sample_anno['next']
                    idx += 1

                scene_instance_points = np.concatenate(scene_instance_points)
                scene_instance_labels = np.concatenate(scene_instance_labels)
            
                current_scene = np.vstack((current_scene, scene_instance_points))
                current_scene_labels = np.vstack((current_scene_labels, scene_instance_labels))

            p, l = segscene_g2e(current_scene, current_scene_labels, ego_trans, ego_rot)
            # save a seg-img
            seg_img = segimg_acc(p, l, lidarseg_idx2name_mapping, color_map, sample_idx)
            
            #************  Data transport to target directory ************#
            # camera-> CAM_BACK to CAM_FRONT, xxxxxx.jpg
            # lidar -> xxxxxx.pcd.bin
            # pose  -> xxxxxx.json, "ego_pose": [x, y, z, qw, qx, qy, qz] original nuscenes-style.
            # extrinsics: xxxxxx.json
            #       "LIDAR_TOP": [x, y, z, qw, qx, qy, qz]
            #       "CAM_FRONT": [x, y, z, qw, qx, qy, qz]
            #       ...
            #       "CAM_BACK": [x, y, z, qw, qx, qy, qz]
            #
            # intrinsics: xxxxxx.json
            #       "CAM_FRONT": list (3x3) -> "CAM_BACK"
            #
            # seg_img: xxxxxx.png

            sensor_extrinsics = {} # to store sensor to ego vehicle extrinsics data (E.g. lidar + 6 x cameras)
            camera_intrinsics = {} # all camera intrinsics data.

            # LiDAR
            src_lidar_sample_path = os.path.join(nusc.dataroot, lidar_sample['filename'])
            dst_lidar_sample_path = os.path.join(scene_lidar_path, f"{str(sample_idx).zfill(6)}.pcd.bin")
            shutil.copy(src_lidar_sample_path, dst_lidar_sample_path)

            # Images
            for camera in camera_name:
                camera_sample = nusc.get('sample_data', current_sample['data'][camera])
                camera_calib = nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
                # store camera's sensor extrinsics to ego vehicle
                sensor_extrinsics[camera] = camera_calib['translation'] + camera_calib['rotation']
                # store camera's sensor intrinsics matrix (3x3) list.
                camera_intrinsics[camera] = camera_calib['camera_intrinsic']
                src_camera_sample_path = os.path.join(nusc.dataroot, camera_sample['filename'])
                dst_camera_sample_apth = os.path.join(scene_camera_path, camera + f"/{str(sample_idx).zfill(6)}.jpg")
                shutil.copy(src_camera_sample_path, dst_camera_sample_apth)

            # Ego vehicle's pose information, [x, y, z, qw, qx, qy, qz]
            ego_pose_dict = {"ego_vehicle": ego_pose['translation'] + ego_pose['rotation']}
            
            with open(os.path.join(scene_pose_path, f"{str(sample_idx).zfill(6)}.json"), "w", encoding="utf-8") as f:
                json.dump(ego_pose_dict, f)

            # All sensors' extrinsics
            # LiDAR -> Ego vehicle's Extrinsics (very important for validation of the CAD GT labels)
            # LIDAR_TOP: [x, y, z, qw, qx, qy, qz]
            calibration_lidar = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            sensor_extrinsics['LIDAR_TOP'] = calibration_lidar['translation'] + calibration_lidar['rotation']
            with open(os.path.join(scene_extrinsics_path, f"{str(sample_idx).zfill(6)}.json"), "w", encoding="utf-8") as f:
                json.dump(sensor_extrinsics, f)

            # All cameras' intrinsics
            with open(os.path.join(scene_intrinsics_path, f"{str(sample_idx).zfill(6)}.json"), "w", encoding="utf-8") as f:
                json.dump(camera_intrinsics, f)

            # BEV semantic image, we locate this label file according to each sample in each scene !
            cv2.imwrite(os.path.join(scene_seg_path, f"{str(sample_idx).zfill(6)}.png"), seg_img)

            # # BEV semantic image, we locate this label file based on sample_id !!!
            scene_within_id2sample_id[test_scene_name + "_"+str(sample_idx).zfill(6)] = current_sample['token']

            # current_sample_token point to the next sample !
            current_sample_token = current_sample['next']

        # break # break scene ...
 
    with open(os.path.join(CAD_LABEL_PATH, "scene_id2sample_id.json"), 'w', encoding='utf-8') as ff:
        json.dump(scene_within_id2sample_id, ff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scene Splitting of nuScenes-CAD')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='NuScenes dataset version, defatul is v1.0-trainval')
    parser.add_argument('--source', type=str, default='./data/trainval', help='Path to the nuScenes dataset')
    parser.add_argument('--target', type=str, default='./nuScenes-CAD-scene-split', help='Directory where the nuScenes-CAD will be saved')
    args = parser.parse_args()
    main(args)