# Weeds Detection Using Tensorflow
## Table of Content
* [Introduction](#Introduction)
* [Environment Requirements](#Environment-Requirements)
* [Prepare and Label Image Data](#Prepare-and-Label-image-data)
* [Generate tf record](#Generate-tf-record)
## Introduction  
This software is build using tensorflow framework to detect the most common weeds in the rural environment. Due overhead of GPU memory (6G RTX 2060) on my local desktop, I only trained faster_rcnn_inception_v2_coco from [tensorflow object detection models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) with three type of weeds (dandelion, oxalis,buckhorn plantain). you can definitely train the model to detect more weeds on your computer with much better GPU.   The google search engine also built in software allow users to check out solution to kill that specified weed in real time. Here is a quick demo:<br>

![demo video](src/video/demo.gif)
## Environment Requirements:
* Python >= 3.5
* Tensorflow >=1.4.0
* PyQt5
* opencv-python3





## Prepare and Label image data:
### Download images 
All weed images are located in corresponding directory within [images folder](/images). In case that you want to try out different type of weeds, you can download more images via [google image download](https://pypi.org/project/google_images_download/) (make sure that images are placed in directory under their name within images folder !).

### Label image data
 Download labelImg from [(tzutalin/labelImg)](https://github.com/tzutalin/labelImg), then follow through all procedures and install on your computer. Once all installations are finished, go to labelImg folder and type this command in your terminal:<br>
 ```bash
 $python3 labelImg.py
 ```
After entering above command,  A GUI window will pop up like this:<br>

![alt text](src/image/screen1.png)

Click the Open Dir and load all images into window. Once this is done, the bottom right corner will list all uploaded images<br>
![alt text](src/image/screen2.png)

Click Create\nRectBox (<b>Keep PascalVOC format !!!</b>) and start to label image.<br>

![demo video](src/video/demo2.gif)

After the image is labeled, and press Ctr+S to save all labeled data into xml file. after it is saved, Click Next images and repeat same process<br>




## Generate tf record:
 Split all labeled images along with their xml files(20% for training and 80% for testing) and place them into [test](images/test) and [train](images/train) in images folder. And convert all *.xml to csv files by running python scripts from your terminal:

```bash
$python3 xml_to_csv.py  
```
Next, generate tf record file for test and train data just by run python script from your terminal:
```bash
$python3 generate_tfrecord.py\
        --csv_input=data/test_labels.csv\
        --output_path=data/test.record\
        --image_dir=images/test/

$python3 generate_tfrecord.py\ 
         --csv_input=data/train_labels.csv\
         --output_path=data/train.record\ 
         --image_dir=images/train/
```
After running above scripts, the tfrecod file for both train and test should be located in [data](data/) folder.
## Usage :

```bash
$python3 user_gui.py
```
