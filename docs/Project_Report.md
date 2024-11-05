<h1 style="text-align: center"> R7020E Final Project </h1>

<p style="text-align: center"> Group 6</p>
<p style="text-align: center"> Jose Juan Pena Gomez (jospen-3)</p>
<p style="text-align: center"> Sushanta Mohapatra (susmoh-3)</p>
<p style="text-align: center"> 31 October 2024</p>

# Index

1. Introduction
2. Object Detection
3. Decoy Filtering
4. Position Estimation
5. Conclusion


# Introduction

The aim of this project is to develop a system that can detect objects in a given image, classify if it's a decoy
object and then estimate the relative distance of the object given the depth. The system is divided into three main
components: 

- Object Detection
- Decoy Filtering
- Position Estimation


# Object Detection

Models were trained on the Roboflow [FireExtinguiser](https://universe.roboflow.com/fire-extinguisher/) dataset. The dataset consists of images of fire extinguishers and other objects. The model was trained using the YOLOv11 architecture. The model was trained for 100 epochs on a single NVIDIA RTX 2080 Ti GPU. 

Models trained were:
- Faster R-CNN
- YOLOv11

## Faster R-CNN


<p style="text-align: center"> Figure 1: Fire Extinguisher Detection with Faster R-CNN</p>

## YOLOv11


Some of the results are shown below:

![](pretrained_models/yolo_v11_extinguiser/val_batch2_pred.jpg)
<p style="text-align: center"> Figure 2: Fire Extinguisher Detection with YOLOv11</p>



# Decoy Filtering

The decoy filtering is done using two methods:
- Feature Extraction using ResNet18 and K-Means Clustering
- Local Binary Pattern histogram for Texture Analysis

## Feature Extraction using ResNet18 and K-Means Clustering


<p style="text-align: center"> Figure 3: Feature Extraction using ResNet18</p>


## Local Binary Pattern histogram for Texture Analysis

The Local Binary Pattern (LBP) is a texture descriptor that is used to classify textures based on the patterns formed by the intensity values of the pixels in an image. 
The LBP histogram is used to classify the object as a decoy or not.

<p style="text-align: center"> Figure 4: LBP Histogram for Texture Analysis</p>


# Position Estimation

The position estimation is done using the depth information from the image. The depth information is obtained using the depth map generated from the stereo camera. 
The depth map is used to estimate the relative position in 3D $(X,Y,Z)$ based in the 2D position $(x,y)$ of the object in the image and the depth $D(x,y)$.

$$
X = \frac{(x - c_x) * Z}{f_x}
Y = \frac{(y - c_y) * Z}{f_y}
Z = D(x,y)
$$

where:
- $(x,y)$ is the 2D position of the object in the image
- $(c_x, c_y)$ is the principal point of the camera
- $(f_x, f_y)$ is the focal length of the camera
- $D(x,y)$ is the depth of the object at position $(x,y)$
- $(X,Y,Z)$ is the 3D position of the object
- $Z$ is the depth of the object
- $X$ is the relative position in the x-axis
- $Y$ is the relative position in the y-axis

# Conclusion

