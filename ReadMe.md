# Building nuScenes-CAD dataset

nuScenes is a widely-adopted, large-scale open-source dataset specifically designed for autonomous driving research. It provides multi-view camera images, LiDAR point clouds, and comprehensive annotations, making it an excellent benchmark for surround-view perception tasks such as Bird's-Eye-View (BEV) semantic segmentation and semantic occupancy prediction. Although the original nuScenes dataset does not include direct annotations for **Circular Accessible Depth (CAD)**, its rich metadata—including 3D bounding boxes, semantic point clouds, sensor calibrations, and ego-vehicle poses—enables the automated generation of high-quality ground-truth labels for this tasks. To this end, we introduce **nuScenes-CAD**, a derived dataset built directly on top of the original nuScenes data. In this section, we first describe the annotation generation pipeline for nuScenes-CAD, followed by details on its file structure and usage instructions.

## Contents
1. [Circular Accessible Depth](#circular-accessible-depth)
2. [Automated Generation Pipeline](#automated-generation-pipeline)   
2.1 [Temporal Semantic Point Cloud Fusion](#temporal-semantic-point-cloud-fusion)   
2.2 [Cross-Frame Instance Enhancement](#cross-frame-instance-enhancement)   
2.3 [Ray-based Accessible Depth Calculation](#ray-based-accessible-depth-calculation)   
3. [How to use](#how-to-use)
4. [Reference](#reference)

## Circular Accessible Depth
CAD represents traversable space as a set of maximum accessible depths in all radial directions centered on the ego vehicle. Unlike pixel-wise BEV semantic segmentation maps based on Cartesian coordinates, CAD adopts a polar coordinate representation that more intuitively and efficiently encodes the distance to the traversable area boundary.

## Automated Generation Pipeline


### Temporal Semantic Point Cloud Fusion

### Cross-Frame Instance Enhancement

### Ray-based Accessible Depth Calculation

## How to use


## Reference