YOLOv6 training
===

See the [parent directory](../README.MD) for instructions on how to prepare the dataset

This folder contains the commands and scripts for training a YOLOv6 model

## Files

* fix_affinetransform.patch - Fixes a bug in YOLOv6 that causes affine transformation to crash the training process in some rare instances. This will only work if your pictures have 1:1 aspect ratio! Apply it inside YOLOv6 folder using `git apply fix_affinetransform.patch`.

Fixes the following issue during training:

```
  File "C:\Users\User\Projects\YOLOv6\yolov6\data\data_augment.py", line 81, in random_affine
    height, width = new_shape
    ^^^^^^^^^^^^^
TypeError: cannot unpack non-iterable int object
```

* finetune_config.py - An example for fine-tuning. You'll need to change the value of `pretrained` parameter to the location of your trained weights

* train_v6.bat - Example command to train YOLOv6. Assumes that data is in `{current_dir}/dataset/` and that the name of the data file is `data.yaml`. Uses yolov6s model. You can adjust image_size depending on your needs (though in my experience 1024x1024 did rather poorly) and batch_size depending on how much VRAM your computer has (batch size of 8, with 2048x2048 size needed 24GiB of VRAM in my case)

* infer_v6.bat - Example command to run predictions on each of the files in a given directory. You can then view the results using [yolo box draw tool](../yolo_box_draw.py)!

* validate_v6.bat - Example command to run validation to generate results graph

* finetune_v6.bat - Example command to further train (fine tune) your model

NOTE - the bat files should be run in the root of YOLOv6 repository (or you should just adjust the paths).

## Set-up

0. Prepare your dataset
1. Clone [YOLOv6](https://github.com/meituan/YOLOv6)
  * Apply the fix in the `fix_affinetransforms.patch`!
2. Copy the scripts in this folder to the cloned folder
3. Run the necessary scripts that you need. Start with train_v6.bat.

## Results

With 773 pictures, and 3.5h of training time (100 epochs) I got the following results on the AGAR dataset: `mAP@0.5: 0.5482137018679738 | mAP@0.50:0.95: 0.30236057819784035`.

## Graphing

YOLOv6 outputs all loss values using TensorBoard. To get graphs you'll need to install tensorflow (`pip install tensorflow`), and then run it inside YOLOv6 folder (`tensorboard --logdir runs/` -- assuming you have the python scripts folder in your PATH). It will prompt you with an URL you can open to see the graphs. Please note that it takes a while for the graphs to actually render as TensorBoard needs to parse the files.