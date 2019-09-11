# Weeds Detection Using Tensorflow
## Table of Content
[Introduction](#Introduction)
## Introduction  
This software is build using tensorflow framework to detect the most common weeds in the rural environment. Due overhead of GPU memory (6G RTX 2060) on my local desktop, I only trained faster_rcnn_inception_v2_coco from [tensorflow object detection models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) with three type of weeds (dandelion, oxalis,buckhorn plantain). you can definitely train the model to detect more weeds on your computer with much better GPU.   The google search engine also built in software allow users to check out solution to kill that specified weed in real time. Here is a quick demo:<br>

![demo video](src/video/demo.gif)
## Required Python Libraries:

* Tensorflow >=1.4.0
* PyQt5
* opencv-python3

## Usage :

```bash
    python3 user_gui.py
```
