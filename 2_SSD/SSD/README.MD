Single-shot MultiBox detector
===

SSD code modified from [here](https://github.com/oarriaga/paz). This code is adapted to disable many of the pre and postprocessing that might ruin the image thus ruining the dataset in my case. Ideally this post-processing should be re-enabled though.

Practically this network does not work as-is. Yes it will train, and yes you can infer using it, but at least for bacterialal recognition this network won't work in its' current state. More work is needed to make it work.

## Setting up

* Requires Python version 3.7.9
* OpenCV2 4.7 or newer (pip install opencv-python)
* Argparse library (pip install argparse)
* Requires TensorFlow w/ Keras. Please note that on Windows the last version that supports GPU natively is 2.10.2!

You can find the [requirements here](../requirements.txt)!

## Setting up

1. It's recommended to train your own base layer (in this case - VGG-16). For more information refer to VGG16's folder [here](../VGG16/)!
2. You can then transfer the base using [transfer_base.py](./transfer_base.py). You can also try training it from ground up, but in my experience the network (usually) won't converge properly.

## Preparing data

Just like the VGG-16 network, this implementation assumes that the data is in Keras' folder structure, namely 

```
root
  -- class
     -- 1.jpg
     -- 1.json
     -- 2.jpg
     -- 2.json
  -- class2
     -- 1.jpg
     -- 2.json
     -- 2.jpg
     -- 2.json
```

For input you need 

* Pictures in 512x512 size. For resizing you can use the [resizing tool](../../1_Preparation/process_resize.py)!
* Json files with bounding boxes. An example of the json file is as follows - 

```
{"colonies_number": 2, "labels": [{"height": 29, "width": 31, "x": 73, "y": 180}, {"height": 22, "width": 23, "x": 88, "y": 314}]}
```

These can be created by [the bounding box creator](../../1_Preparation/bounding_box_creator.py)!

## Running

Run [main.py --help](./main.py). By default, if your data is in a folder `dataset` and your transferred model is called `ssd.h5` you should be able to train the model by just running `main.py`. For inference and validation use the `--mode` argument (train,count,validate)!

## Output

After training just run the program in count mode. Note that currently the output does not maintain the original file name, so all files will be written as `output.jpg` (image) and `output.csv` (bounding boxes). This should, of course, be fixed, but right now making the model work is a higher priority.
