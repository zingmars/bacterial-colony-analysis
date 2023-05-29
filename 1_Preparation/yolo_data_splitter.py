import numpy as np
import logging
import argparse
from os import listdir, makedirs
from os.path import isfile, join, splitext
import random
import shutil

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="YOLO dataset splitter. All data should be in the training folder!")
argParser.add_argument("--base-folder", default="dataset", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--source", default="train", type=str, help="Base folder from which the images will be taken (usually train). Changing this value is not supported, do it at your own risk!")
argParser.add_argument("--train", default="train", type=str, help="Train folder")
argParser.add_argument("--train-split", default="train", type=str, help="Train folder")
argParser.add_argument("--validation", default="valid", type=str, help="Validation folder")
argParser.add_argument("--test", default="test", type=str, help="Test folder")
argParser.add_argument("--split", default="70/20/10", type=str, help="Split values, divided by a slash")
args = argParser.parse_args()

logging.log(55, 'Script started.')

def np_split(lst: list, split: list['int']):
    r1, r2, r3 = split
    split_indices = [int(len(lst) * r1), int(len(lst) * (r1+r2))]
    arr1, arr2, arr3 = np.split(lst, split_indices)
    return arr1, arr2, arr3

logging.info("Generating folder names...")
baseImagefolder = f"{args.base_folder}\\images"
baseLabelsfolder = f"{args.base_folder}\\labels"
sourceImagefolder = f"{baseImagefolder}\\{args.source}"
sourceLabelsfolder = f"{baseLabelsfolder}\\{args.source}"

logging.info("Creating all of the folders...")
makedirs(f"{baseImagefolder}\\{args.train}", exist_ok=True)
makedirs(f"{baseImagefolder}\\{args.validation}", exist_ok=True)
makedirs(f"{baseImagefolder}\\{args.test}", exist_ok=True)
makedirs(f"{baseLabelsfolder}\\{args.train}", exist_ok=True)
makedirs(f"{baseLabelsfolder}\\{args.validation}", exist_ok=True)
makedirs(f"{baseLabelsfolder}\\{args.test}", exist_ok=True)

logging.info("Generating split values...")
split = args.split.split("/")
if len(split) != 3:
    raise Exception("Split format should be: `traincount/validationcount/testcount`!")
split = [int(x) for x in split]
if sum(split) != 100:
    raise Exception("Split does not add up to 100!")
logging.info(f"Split: {split[0]} / {split[1]} / {split[2]}")
split = [x/100 for x in split]

logging.info("Generating file list...")
fileList = [f for f in listdir(sourceImagefolder) if isfile(join(sourceImagefolder, f))]

logging.info("Shuffling the list...")
random.shuffle(fileList)

logging.info("Splitting the dataset...")
train, validate, test = np_split(fileList, split)

logging.info(f"Data was split into {len(train)} for training, {len(validate)} for validation and {len(test)} for testing!")

def process_move(fileList, dest):
    for file in fileList:
        logging.info(f"Processing {file}...")
        fullPath = f"{sourceImagefolder}\\{file}"
        splitFileName = splitext(file)
        fullLabelPath = f"{sourceLabelsfolder}\\{splitFileName[0]}.txt"
        if not isfile(fullLabelPath):
            raise Exception(f"Could not find a matching label for {file}!")
        shutil.move(fullPath, f"{baseImagefolder}\\{dest}\\{file}")
        shutil.move(fullLabelPath, f"{baseLabelsfolder}\\{dest}\\{splitFileName[0]}.txt")

logging.info("Moving data to validation folder...")
process_move(validate, args.validation)

logging.info("Moving data to test folder...")
process_move(test, args.test)

logging.log(56, "Script finished!")