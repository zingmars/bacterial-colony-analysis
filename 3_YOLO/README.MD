Training YOLOv6/8 for bacteria analysis
===

this folder has the scripts and 

## 1. Dataset preparation

NOTE: These tools use Windows paths, which probably won't work on Linux. Some work is needed to make the whole thing system agnostic!

First you'll need a dataset. This includes pictures and files that have bounding boxes marked for each bacteria. Bounding boxes can be created using the [bounding_box_draw tool](../1_Preparation/bounding_box_draw.py).

Then you'll need to split the dataset in training, validation and testing datasets (the latter is optional). For this you can use [yolo_data_splitter tool](../1_Preparation/yolo_data_splitter.py). This will take a folder that has all of the data in 1 folder and split them into 3 directories randomly.

If you need to split the data further, you can use the [yolo_dataset_splitter](../1_Preparation/yolo_dataset_splitter.py) tool which will split the dataset and re-compute the class list for the split dataset.

## 2. Running

### YOLOv6:

1. Clone Yolov6 implementation: https://github.com/meituan/YOLOv6
2. Copy dataset over to the folder
3. Run the script in the train.bat file

random_affine bug:

When running YOLOv6 I had it crash on random_affine function. Replace in yolov6/data/data_augment.py to height, width = new_shape
    try:
        height, width = new_shape
    except:
        height = new_shape
        width = new_shape

Note that this will only work if your input has 1:1 aspect ratio. Otherwise your model will return invalid results!

YOLOv8
1. Install YOLOv8
2. Copy dataset over to the folder
3. Run the script in the train.bat file

3. Memory Usage
773 images, 7 classes
YOLOv6s = 70GB
YOLOv6s = 24GiB VRAM (epoch 7)

## Logging

YOLOv6 and YOLOv8 training scripts
===

## Preparing data

You'll need:

* Images of your data
* Bounding boxes for your data

For the former you can see [parent README](../README.MD). For the latter you can try the AGAR dataset, or you can make your own using the [bounding box creator](../1_Preparation/bounding_box_creator.py). To use it you'll first need a list of classes, and save them as a json file that follows the following structure - `{"names": [ "class1, "class2" ]}`. Then put all of your images in a folder and pass both the json file and the image folder to the bounding box creator and start marking.

You can then use the [dataset prepare tool](../1_Preparation/prepare_dataset.py) to create a training dataset, and then split it into training/validation/testing datasets using [data splitter](../1_Preparation/yolo_data_splitter.py).

## Training

Pick a model! The data format for both YOLOv6 and YOLOv8 is compatible, meaning you can use it for either (or both) models. See instructions in either [yolov6](./yolov6/README.MD) or [yolov8](./yolov8/README.MD) directories.

## Getting a result

Each model has a train*.bat file, that you can use to run the model on actual data.

## Using the result

To get a result you'll need to use infer modes for the model you chose. The directory for each of the models has a script that allows you to do this. Afterwards you can take the original picture and the generated labels file and view it using [yolo_box_draw](./yolo_box_draw.py) which will also filter out overlapping boxes that might appear!
