YOLOv8 training scripts
===

See the [parent directory](../README.MD) for instructions on how to prepare the dataset

This folder contains the commands and scripts for training a YOLOv8 model

## Files

* train_v8.bat - Example command to train YOLOv8. Assumes that data is in `{current_dir}/dataset/` and that the name of the data file is `data.yaml`. Uses yolov6s model. You can adjust image_size depending on your needs (though in my experience 1024x1024 did rather poorly) and batch_size depending on how much VRAM your computer has (batch size of 4, with 2048x2048 size needed 24GiB of VRAM in my case)

* infer_v8.bat - Example command to run predictions on each of the files in a given directory. You can then view the results using [yolo box draw tool](../yolo_box_draw.py)!

* validate_v6.bat - Example command to run validation to generate results graph

* finetune_v6.bat - Example command to further train (fine tune) your model

## Set-up

0. Prepare your dataset
1. Install YOLOv8 (`pip install ultralytics`) OR clone [YOLOv8](https://github.com/ultralytics/ultralytics)
  * Copy the scripts in this folder to the cloned folder 
3. Run the necessary scripts that you need. Start with train_v8.bat.

## Results

With 773 pictures, and 2.5h of training time (100 epochs) I got the following results on the AGAR dataset: `mAP@0.5: 0.65269 | mAP@0.50:0.95: 0.3568`.

## Graphing

Unlike YOLOv6 which graphs data using TensorBoard, YOLOv6 will simply generate a file called `results.csv` in your training folder. This can then be graphed using MS Excel or similar software.