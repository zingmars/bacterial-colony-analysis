Step 1: Data normalization
===

This folder contains the source code for image preparation process.

* Requires Python version 3.7.9
* OpenCV2 4.7 or newer (pip install opencv-python)
* Argparse library (pip install argparse)

You can also use the [requirements.txt](./requirements.txt) file to install these libraries!

## Files

### Bounding box management

* bounding_box_creator.py - Loads files in a folder and allows the user to draw bounding boxes around objects. The output is a json file containing the coordinates of the bounding boxes. Accepts a file with a list of classes for marking! (json file - `{"names": [ "class1, "class2" ]}`). This is pretty much required if you want to create a dataset for a YOLOv6 or YOLOv8 model.
* bounding_box_draw.py - A tool to visualize bounding boxes
* bounding_box_resize.py - A tool to resize images along with their bounding boxes

### Dataset preparation

* prepare_dataset.py - Prepares dataset for SSD and YOLO networks. Please note that SSD preparation only accepts single-class images. For YOLO networks it will create a valid dataset with all of the files marked for training. To further split it, use `yolo_data_splitter.py`

### Image preprocessing

* process_circledetect.py - Runs circle detection algorithm to cut the plate out of the image. Requires a lot of manual fine-tuning. Does not work well automatically.
* process_edgedetect.py - Runs edge detection algorithm on the image
* process_resize.py - Simple image resizer
* process_subtract.py - Simple image subtractor
* process_threshold.py - Runs thresholding algorithms on the image

### Networks

#### VGG

* vgg_file_rename.py - Renames data for the VGG network

#### YOLO

* yolo_box_draw.py - Draws an image with boxes on it using YOLO-compatible coordinates
* yolo_data_splitter.py - Splits data from the training fodler into training, validation and testing datasets. Use this after `prepare_dataset.py`.
* yolo_dataset_splitter.py  - Splits YOLO dataset into two, in case you need to make your dataset smaller. Will regenerate the data yaml file for you.