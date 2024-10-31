<h1 style="text-align: center"> R7020E Final Project </h1>

<p style="text-align: center"> Group 6</p>
<p style="text-align: center"> Jose Juan Pena Gomez (jospen-3)</p>
<p style="text-align: center"> Sushanta Mohapatra (susmoh-3)</p>
<p style="text-align: center"> 31 October 2024</p>

# Index

1. Introduction
2. Object Detection
3. Object Classification
4. Depth Estimation


# Introduction

The aim of this project is to develop a system that can detect objects in a given image, classify if it's a decoy
object and then estimate the relative distance of the object given the depth. The system is divided into three main
components: 

- Object Detection
- Object Classification 
- Depth Estimation.


# Object Detection

For the object detection we trained YOLOv11 on the [FireExtinguiser](https://universe.roboflow.com/fire-extinguisher/) dataset.


Some of the results are shown below:

![](pretrained_models/yolo_v11_extinguiser/val_batch2_pred.jpg)

<p style="text-align: center"> Figure 1: Fire Extinguisher Detection</p>

