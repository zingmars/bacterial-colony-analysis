Baktēriju koloniju analīze izmantojot neironu tīklus | Bacterial colony analysis using neural networks
===

Šis repozitorijs satur vajadzīgos rīkus un pirmkodu programmai, kas veic baktēriju koloniju analīzi. Pielikums RTU DITF studenta Ingmāra Daniela Meļķa (161RDB280) maģistra darbam. 

###### Visi komentāri, un dokumentācija no šī punkta būs angļu valodā

This repository contains the scripts and source code for an application, that performs bacterial colony analysis. Developed as part of masters thesis by Rigas Technical University's student Ingmars Daniels Melkis.

This repository contains various tools for preparing datasets for SSD and YOLO networks. It also includes an implementation of a SSD network based on VGG-16, although in its current state it's not usable.

# Requirements

* All scripts were developed and tested for Windows. They might or might not work on Linux or MacOS.
* For VGG15 and SSD: Python 3.9.7. Newer versions _might_ or _might not_ work.
* For YOLOv6 and v8: Python 3.11 or newer.
* Main requirements for scripts in a given folder are listed in the corresponding `requirements.txt` file
* SSD Uses OpenCV and Tensorflow 2.10 (latest version with Windows GPU support). OpenCV w/ CUDA support is supported in some cases (though that might involve compiling it manually, and no instructions are provided in this repository).

# Suggested datasets

1. Your own!
2. [AGAR](https://arxiv.org/abs/2108.01234) - Currently the best available dataset, although you might want to filter out images that don't have any bacteria merked in them. This can be done using the [Dataset preparation script](./1_Preparation/prepare_dataset.py)!
2. [ColTapp](https://figshare.com/articles/media/Data_ColTapp/12951152/1) - Grayscale pictures
3. [CoCoNut](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6221277/#sec015title) - Very tiny dataset with HeLa and V79 in a flask and on a dish
4. [SP-SDS](https://data.mendeley.com/datasets/f8ykr3bt7y) - Also a very tiny dataset with a timeline of some bacteria growing on a dish

# Setting up

First you'll need to set up Python with all of the required libraries. Make sure that Tensorflow (SSD) and/or Torch (YOLO) are using your GPU. You can test this using the following scripts - [TF](./miscellaneous/test_tensorflow.py) and [Torch](./miscellaneous/test_torch.py).

Next you'll need to prepare your dataset. [Preparation folder](./1_Preparation/) has all of the scripts you might need for that.

# Running and evaluating

Refer to the README in the folder for the model you want to run (probably YOLO) - [SSD](./2_SSD/) and [YOLO](./3_YOLO/).

# Further work

1. Clean up the code for the SSD model. Make it actually work.
2. Check multi-class performance. This module was made assuming that you'll only have one class (one type of bacteria) per dish.
3. Try using RESNET or different base architecture with either SSD or R-CNN.
4. Create an automated data preparation pipeline. Right now the whole thing is as manual as it gets.

# Contents

1. `1_Normalization` - This folder contains the scripts for normalizing images for neural network
2. `2_SSD` - This folder contains the code for a single-shot detector based on a VGG16 neural network that predicts bounding boxes and counts them. Based on [PAZ](https://github.com/oarriaga/paz).
3. `3_YOLO` - This folder contains the code for YOLOv6 and YOLOv8 based models. In their case you can generally use their implementation as-is.
4. `miscellaneous` - Miscellanous scripts and test code.