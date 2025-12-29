# Building nuScenes-CAD dataset

nuScenes is a widely-adopted, large-scale open-source dataset specifically designed for autonomous driving research. It provides multi-view camera images, LiDAR point clouds, and comprehensive annotations, making it an excellent benchmark for surround-view perception tasks such as Bird's-Eye-View (BEV) semantic segmentation and semantic occupancy prediction. Although the original nuScenes dataset does not include direct annotations for **Circular Accessible Depth (CAD)**, its rich metadata—including 3D bounding boxes, semantic point clouds, sensor calibrations, and ego-vehicle poses—enables the automated generation of high-quality ground-truth labels for this task. To this end, we introduce **nuScenes-CAD**, a derived dataset built directly on top of the original nuScenes dataset. In this section, we first describe the annotation generation pipeline for nuScenes-CAD dataset, followed by details on its structure and usage instructions.

## Contents
1. [Circular Accessible Depth](#circular-accessible-depth)
2. [Automated Generation Pipeline](#automated-generation-pipeline)   
2.1 [Temporal Semantic Point Cloud Fusion](#temporal-semantic-point-cloud-fusion)   
2.2 [Cross-Frame Instance Enhancement](#cross-frame-instance-enhancement)   
2.3 [Ray-based Accessible Depth Calculation](#ray-based-accessible-depth-calculation)   
3. [How to use](#how-to-use)
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
Following instance enhancement, we obtain a dense semantic point cloud in the ego-vehicle coordinate system for each sample. We crop a ±20m region around the ego vehicle and project it onto a 200×200 BEV grid (0.2m/pixel resolution). L=384 rays are cast outward from the ego center. Each ray stops at the first non-traversable obstacle pixel (e.g., vehicles, pedestrians, traffic cones, vegetation) and records the distance as the accessible depth. If no obstacle is hit within 20m, the depth is set to D_max = 20m.


## How to use


## Reference