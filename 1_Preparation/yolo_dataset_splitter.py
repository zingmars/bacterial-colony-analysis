import numpy as np
import logging
import argparse
from os import listdir, makedirs
from os.path import isfile, join, splitext
import random
import shutil
import yaml
import csv

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="YOLO dataset splitter - splits the original dataset into two parts!")
argParser.add_argument("--base-folder", default="dataset", type=str, help="Path to the directory that contains the dataset that should be split")
argParser.add_argument("--data-file", default="data.yaml", type=str, help="Name of the data yaml file")
argParser.add_argument("--split-folder", default="dataset_split", type=str, help="Path to the directory that the split images will be moved to")
argParser.add_argument("--train", default="train", type=str, help="Train folder")
argParser.add_argument("--validation", default="valid", type=str, help="Validation folder")
argParser.add_argument("--test", default="test", type=str, help="Test folder")
argParser.add_argument("--split", default="90/10", type=str, help="Split values, divided by a slash")
args = argParser.parse_args()

logging.log(55, 'Script started.')

def np_split(lst: list, split: list['int']):
    r1 = split[0]
    split_indices = [int(len(lst) * r1)]
    arr1, arr2 = np.split(lst, split_indices)
    return arr1, arr2

logging.info("Generating folder names...")
baseImagefolder = f"{args.base_folder}\\images"
baseLabelsfolder = f"{args.base_folder}\\labels"
targetImagefolder = f"{args.split_folder}\\images"
targetLabelsFoder = f"{args.split_folder}\\labels"

logging.info("Loading current class list...")
yamlFile = f"{args.base_folder}\\{args.data_file}"
with open(yamlFile, "r") as stream:
    dataset_info = yaml.safe_load(stream)
logging.info(f"Loaded a dataset file defininf {dataset_info['nc']} classes!")
oldClassList = dataset_info['names']
newClassList = []

logging.info("Creating all of the folders...")
makedirs(f"{targetImagefolder}\\{args.train}", exist_ok=True)
makedirs(f"{targetImagefolder}\\{args.validation}", exist_ok=True)
makedirs(f"{targetImagefolder}\\{args.test}", exist_ok=True)
makedirs(f"{targetLabelsFoder}\\{args.train}", exist_ok=True)
makedirs(f"{targetLabelsFoder}\\{args.validation}", exist_ok=True)
makedirs(f"{targetLabelsFoder}\\{args.test}", exist_ok=True)

logging.info("Generating split values...")
split = args.split.split("/")
if len(split) != 2:
    raise Exception("Split format should be: `size1/size2`!")
split = [int(x) for x in split]
if sum(split) != 100:
    raise Exception("Split does not add up to 100!")
logging.info(f"Split: {split[0]} / {split[1]}")
split = [x/100 for x in split]

def process_dataset(datasetType: str, train_set: bool):
    currentDir = f"{baseImagefolder}\\{datasetType}"
    fileList = [f for f in listdir(currentDir) if isfile(join(currentDir, f))]
    logging.info(f"Shuffling the {datasetType} dataset...")
    random.shuffle(fileList)
    logging.info(f"Splitting the {datasetType} dataset...")
    firstArr, secondArr = np_split(fileList, split)
    logging.info(f"Processing the {datasetType} dataset...")
    for file in secondArr:
        # 1. Locate labels
        logging.info(f"Processing {file}...")
        fullPath = f"{currentDir}\\{file}"
        splitFileName = splitext(file)
        fullLabelPath = f"{baseLabelsfolder}\\{datasetType}\\{splitFileName[0]}.txt"
        if not isfile(fullLabelPath):
            raise Exception(f"Could not find a matching label for {file}!")
        newLabelPath = f"{targetLabelsFoder}\\{datasetType}\\{splitFileName[0]}.txt"

        # 2. Read the labels
        logging.info(f"Processing {file} labels...")
        with open(fullLabelPath, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            newRows = []
            for row in reader:
                label = int(row[0])
                labelName = oldClassList[label]
                if labelName not in newClassList:
                    if not train_set:
                        logging.warning("Not copying current file: This is not the training dataset, and we encountered an unknown class!")
                        continue
                    newClassList.append(labelName)
                label_idx = newClassList.index(labelName)
                newRow = row.copy()
                newRow[0] = label_idx
                newRows.append(newRow)
            with open(newLabelPath, 'w', newline='') as outputfile:
                writer = csv.writer(outputfile, delimiter=' ')
                writer.writerows(newRows)
        
        # 3. Move the image and back up the thing
        logging.info(f"Backing up original file labels")
        shutil.move(fullPath, f"{targetImagefolder}\\{datasetType}\\{file}")
        backup_folder = f"{targetLabelsFoder}\\{datasetType}_backup"
        makedirs(backup_folder, exist_ok=True)
        shutil.move(fullLabelPath, f"{backup_folder}\\{splitFileName[0]}.txt")

logging.info("Processing the training dataset...")
process_dataset(args.train, True)

logging.info("Processing the validation dataset...")
process_dataset(args.validation, False)

logging.info("Processing the testing dataset...")
process_dataset(args.test, False)
      
# Saving the new data file
data_file_name = f"{args.split_folder}\\{args.data_file}"
dataset_info["nc"] = len(newClassList)
dataset_info["names"] = newClassList
with open(data_file_name, 'w') as f:
    f.write(yaml.dump(dataset_info))
    logging.info(f"Saved a dataset with {dataset_info['nc']} classes!")

logging.log(56, "Script finished!")
