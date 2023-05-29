import logging
import argparse
import os
import json
import csv
import yaml
import shutil
import cv2

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="AGAR dataset preparer")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--model", default="YOLO", type=str, help="Model for which to prepare the data (SSD, YOLO")
args = argParser.parse_args()

logging.log(55, 'Script started.')

if args.model not in ["SSD", "YOLO"]:
    raise Exception(f"Unknown model type: {args.model}")

logging.info("Creating output directory...")
from os import listdir, mkdir, makedirs
from os.path import isfile, join, exists
output_folder = f"{args.image_folder}_output"
if not exists(output_folder):
    mkdir(output_folder)

logging.info("Generating file list...")
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]

logging.info("Processing...")

# Counter
class_map = {}

if args.model == "YOLO":
    # Data.yaml file
    data = {
        "train": "images\\train",
        "val": "images\\valid",
        "test": "images\\test",
        "nc": 0,
        "names": [],
        "is_colo": False
    }
    class_list = [] # For YOLO only

for file in fileList:
    if file.endswith(".jpg"):
        continue
    
    filename = os.path.splitext(file)
    json_file_path = f"{args.image_folder}\\{filename[0]}.json"
    image_file_path = f"{args.image_folder}\\{filename[0]}.jpg"
    
    if not os.path.exists(image_file_path):
        logging.error(f"Could not find image file for {file}! Skipping!")
        continue
    logging.info(f"Working on data file {file}...")

    json_file = open(json_file_path)
    annotation = json.load(json_file)
    json_file.close()

    # Check data
    if "classes" not in annotation:
        logging.error(f"File {file} does not have any classes defined! Skipping!")
        continue
    if "labels" not in annotation:
        logging.error(f"File {file} is malformed (no labels array)!")
        continue
    if len(annotation["labels"]) < 1:
        logging.error(f"File {file} has no labels! Skipping!")
        continue
    if annotation["colonies_number"] == -1:
        logging.error(f"File {file} does not have any colonies defined! Skipping!")
        continue
    annotation_count = len(annotation["classes"])
    if annotation_count == 0:
        logging.error(f"File {file} has 0 classes defined! Skipping!")
        continue

    if args.model == "SSD":
        if annotation_count > 1:
            logging.error(f"File {file} has more than 1 classes defined! Skipping!")
            continue

        # Agar dataset uses Trypticase Soy Agar plates
        gene_normalized = annotation["classes"][0]
        gene_normalized = gene_normalized.replace(".","")
        gene_normalized = gene_normalized.lower()
        if gene_normalized not in class_map:
            class_map[gene_normalized] = 0
        else:
            class_map[gene_normalized] += 1
        newNamePrefix = f"{output_folder}\\{gene_normalized}-tsa-{class_map[gene_normalized]}"
        logging.info(f"Copying files with prefix {newNamePrefix}...")
        shutil.copy(json_file_path, f"{newNamePrefix}.json")
        shutil.copy(image_file_path, f"{newNamePrefix}.jpg")
        #logging.info(f"Deleting original files...")
        #os.remove(json_file_path)
        #os.remove(image_file_path)
    elif args.model == "YOLO":
        # Put all images in the train folder. Just sort them out manually later
        baseImageFolder = f"{output_folder}\\images\\train"
        makedirs(baseImageFolder, exist_ok=True)
        baseLabelsFolder = f"{output_folder}\\labels\\train"
        makedirs(baseLabelsFolder, exist_ok=True)

        img  = cv2.imread(image_file_path)
        height, width, _ = img.shape

        # Generate bounding box file
        # Original format: x1, x2, height, width
        # Target format: class_id center_x center_y bbox_width bbox_height 
        yolo_boxes = []
        for box in annotation["labels"]:
            clazz = box["class"]
            x1 = box["x"]
            y1 = box["y"]
            x2 = box["x"] + box["width"]
            y2 = box["y"] + box["height"]
            w = box["width"]
            h = box["height"]

            centerX = int((x1 + x2) /2)
            centerY = int((y1 + y2) /2)
            bbox_width = int(w / 2)
            bbox_height = int(h / 2)

            # Normalize coordinates
            centerX = centerX / width
            if (centerX > 1.0): centerX = 1.0
            centerY = centerY / height
            if (centerY > 1.0): centerY = 1.0
            bbox_height = bbox_height / height
            if (bbox_height > 1.0): bbox_height = 1.0
            bbox_width = bbox_width / width
            if (bbox_width > 1.0): bbox_width = 1.0

            if clazz not in class_map:
                class_map[clazz] = 0
            else:
                class_map[clazz] += 1

            if clazz not in class_list:
                class_list.append(clazz)
            idx = class_list.index(clazz)
            
            yolo_boxes.append([idx, centerX, centerY, bbox_width, bbox_height])

        shutil.copy(image_file_path, f"{baseImageFolder}\\{filename[0]}.jpg")
        with open(f"{baseLabelsFolder}\\{filename[0]}.txt", 'w', newline='') as f:
            csvwriter = csv.writer(f, delimiter=" ")
            csvwriter.writerows(yolo_boxes)

if (args.model == "YOLO"):
    data["nc"] = len(class_list)
    data["names"] = class_list
    with open(f"{output_folder}\\data.yaml", 'w') as f:
        f.write(yaml.dump(data))
    logging.info(f"Saved a dataset with {data['nc']} classes!")

logging.log(56, "Script finished!")