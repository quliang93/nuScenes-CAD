# Building nuScenes-CAD dataset

nuScenes is a widely-adopted, large-scale open-source dataset specifically designed for autonomous driving research. It provides multi-view camera images, LiDAR point clouds, and comprehensive annotations, making it an excellent benchmark for surround-view perception tasks such as Bird's-Eye-View (BEV) semantic segmentation and semantic occupancy prediction. Although the original nuScenes dataset does not include direct annotations for **Circular Accessible Depth (CAD)**, its rich metadata—including 3D bounding boxes, semantic point clouds, sensor calibrations, and ego-vehicle poses—enables the automated generation of high-quality ground-truth labels for this task. To this end, we introduce **nuScenes-CAD**, a derived dataset built directly on top of the original nuScenes dataset. In this section, we first describe the annotation generation pipeline for nuScenes-CAD dataset, followed by details on its structure and usage instructions.

## Contents
1. [Circular Accessible Depth](#circular-accessible-depth)
2. [Automated Generation Pipeline](#automated-generation-pipeline)   
2.1 [Temporal Semantic Point Cloud Fusion](#temporal-semantic-point-cloud-fusion)   
2.2 [Cross-Frame Instance Enhancement](#cross-frame-instance-enhancement)   
2.3 [Ray-based Accessible Depth Calculation](#ray-based-accessible-depth-calculation)   
3. [How to generate nuScenes-CAD dataset?](#how-to-use)   
3.1 [Scene Splitting](#scene-splitting)   
3.2 [Label Generation](#label-generation)   
3.3 [Postprocessing](#postprocessing)   
4. [How to use](#how-to-use)
4. [Reference](#reference)

## Circular Accessible Depth
CAD represents traversable space as a set of maximum accessible depths in all radial directions centered on the ego vehicle (see the purple area in the figure below). Unlike pixel-wise BEV semantic segmentation maps based on Cartesian coordinates, CAD adopts a polar coordinate representation that more intuitively and efficiently encodes the distance to the traversable area boundary.

<figure>
  <img src="assets/img1.jpg">
</figure>

## Automated Generation Pipeline
The nuScenes dataset organizes data by scenes, with each scene lasting approximately 20 seconds. Within each scene, keyframes (samples) and their associated multi-view images and point clouds are annotated at 2 Hz (yielding ~40 annotated samples per scene). Our automated labeling pipeline leverages these rich annotations to generate CAD labels for key samples in each scene. Each CAD label consists of $L=384$ accessible depths radiating from the ego vehicle in all directions around it. The automated generation process comprises three key steps: (1) Temporal Semantic Point Cloud Fusion, (2) Cross-Frame Instance Enhancement, and (3) Ray-based Accessible Depth Calculation, as shown in the figure below.


<figure>
  <img src="assets/img2.png">
</figure>

### Temporal Semantic Point Cloud Fusion
We first construct a dense point cloud map for each scene using the LiDAR segmentation annotations. For a scene with $N$ samples, let $P_i \in \mathbb{R}^4$ denote the semantic point cloud of the i-th sample, where $P_i = \{(x_j, y_j, z_j, cls_j) \mid j=1,\dots,M\}$. Each point includes position $(x,y,z)$ and semantic class $cls$ from [nuScenes LiDARSeg](https://www.nuscenes.org/nuscenes#download).
Since $P_i$ is in LiDAR coordinates, we transform it to the global map frame using the associated sensor calibration $Tf^{lidar \to ego}$ and ego pose $Tf^{ego \to map}$:  


$$P_i^{map} = Tf^{ego \to map} \cdot Tf^{lidar \to ego} \cdot P_i^{lidar}$$


To avoid trailing artifacts from moving objects during temporal fusion, we filter out dynamic classes (e.g., pedestrians, vehicles) based on semantics, retaining only static points:   


$$map = \textbf{Static}(P_1^{map}, \dots, P_N^{map})$$



### Cross-Frame Instance Enhancement
The semantic point cloud map provides static semantic elements (e.g., vegetation, curbs, buildings) for describing scene traversability. For dynamic obstacles (such as pedestrians, cyclists, vehicles, etc.), using only the current frame point cloud results in sparse object representations, while directly performing temporal fusion within a scene leads to trailing artifacts. Therefore, we introduce Cross-Frame Instance Enhancement (CFIE) to increase the point cloud density of objects in the current sample.

**Core idea**: By utilizing nuScenes' instance-level tracking within the same scene, we align the point clouds of the same instance across different frames to the 3D bounding box of the target frame, thereby significantly increasing the point cloud density of dynamic objects in the target frame.


Assume that the dynamic instance to be enhanced in the i-th sample is $obj_k$. Before enhancement, the point cloud within its instance 3D bounding box is $P_i^{obj_k} \in P_i$, consisting of all points inside the 3D box that match the instance’s class, with a total count of $N_k$. The pose of this instance box is $pose_k$. Thanks to the instance-level tracking provided by nuScenes, we can obtain the complete tracklet of $obj_k$ within the scene:


$$Seq^{obj_k}=\{ (P^{obj_k}_1, pose_1), \dots, (P^{obj_k}_T, pose_T) \}$$


We transform the point clouds describing the same instance from different timestamps into the corresponding 3D bounding box of the instance in the i-th frame. This enhances the point cloud density of the instance in the current frame, resulting in the enhanced point cloud representation of $obj_k$ in the i-th frame, with a total point count of:



$$\underbrace{N_k}_{\text{current sample}} + \underbrace{N_1 + N_2 + \dots + N_T}_{\text{tracked sequence}}$$



### Ray-based Accessible Depth Calculation
Following instance enhancement, we obtain a dense semantic point cloud in the ego-vehicle coordinate system for each sample. We crop a ±20m region around the ego vehicle and project it onto a 200×200 BEV grid (0.2m/pixel resolution). L=384 rays are cast outward from the ego center. Each ray stops at the first non-traversable obstacle pixel (e.g., vehicles, pedestrians, traffic cones, vegetation) and records the distance as the accessible depth. If no obstacle is hit within 20m, the depth is set to $D_{max}$ = 20m.


## How to generate nuScenes-CAD dataset?
**Preparation**: Before starting, please download the standard and complete nuScenes dataset along with the corresponding nuScenes-lidarseg annotations in advance. Clone/download this repository and set up the required environment.   
The entire automated annotation pipeline consists of three main steps: (1) Scene splitting (2) Label generation (3) Post-processing. Steps (1) and (3) are implemented in Python, while step (2) is implemented in C++.
The tools required for this pipeline include:   
- nuscenes-devkit==1.1.11   
- PCL (Point Cloud Library)   
- OpenCV library

### Scene Splitting
```Bash
python generation_nuScenesCAD_v4.py \
 --version v1.0-trainval \
 --source /path/to/nuScenes \
 --target /path/to/nuScenes-CAD-scene-split
```

### Label Generation
After performing scene splitting, we obtain the nuScenes-CAD-scene-split dataset, which follows the official nuScenes scene definitions (a total of 850 scenes in the trainval set, with 700 used for training and 150 for validation). Subsequently, we compile the automated label generation from source in the **raycasting** directory. This executable iterates through the **seg_img** folder of each scene in the nuScenes-CAD-scene-split directory and generates the corresponding labels based on the BEV segmentation images.

```Bash
cd /this-repo/raycasting/
mkdir build && cd build
cmake ..
make

./nuScenesCAD_gen_label
```
**Note**: The above process uses the default path to nuScenes-CAD-scene-split. If you wish to modify it, please edit line 225 in the file raycasting/src/nuScenesCAD_gen_label.cpp.

### Postprocessing
Finally, post-processing is applied to nuScenes-CAD-scene-split to:

- Remove redundant directories
- Convert point clouds to standard .pcd format (compatible with PCL or PyntCloud)   
- Create a CAD_LABEL folder containing per-sample CAD labels (named by sample token) and visualization canvases for GT/prediction results
- Generate an index for associating the t-th frame of each scene with its standard sample token.

```Bash
python nuScenes_CAD_postprocessing.py
```

## How to use
Here we recommend two main usage approaches:

1. Official style (compatible with the nuScenes official toolchain)   
2. Deployment style (suitable for real robot platform deployment and self-collected data)

After running the complete pipeline above, a **CAD_LABEL** folder will be generated under the nuScenes-CAD-scene-split directory. This folder contains all label files required for CAD tasks and is fully aligned with the corresponding **sample token** names in the nuScenes dataset.

Approach 1: Official style (recommended for research and development)

- Users can directly use tools provided by the official nuScenes (such as nuscenes-devkit) to load and visualize these labels.   
- When building a custom Dataset, simply use the sample’s token name to index the corresponding CAD label file.
- Available at: [CAD_LABEL.zip](https://drive.google.com/file/d/1W64T4RdQUGZPSHlWi3tkUPqgZdRydJvT/view?usp=sharing)

Approach 2: Deployment style (recommended for real robot deployment and self-collected data)

- When deploying on actual robot platforms, self-collected data is usually organized in a structure similar to nuScenes-CAD-scene-split, namely: (1) folders divided by scene (2) within each scene, point clouds, intrinsics, extrinsics, images, and other time-series data are saved in chronological order

- This structure better suits real-time collection and deployment needs, so we recommend organizing your self-collected dataset in the same format as nuScenes-CAD-scene-split.

## Reference
This repo was inspired by several excellent open-source works. We would like to express our sincere thanks and respect to the following repositories and their authors:

- [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

- [OpenOcc Dataset](https://github.com/OpenDriveLab/OccNet?tab=readme-ov-file#openocc-dataset)

- [CADLabeler](https://github.com/BruceXSK/CADLabeler)