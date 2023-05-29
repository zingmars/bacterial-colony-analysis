import logging
import argparse
import os
import shutil

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="File categorizer")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images from which circles should be extracted")
args = argParser.parse_args()

logging.log(55, 'Script started.')

logging.info("Creating output directory...")
from os import listdir, mkdir
from os.path import isfile, join, exists
file_list = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]
output_folder = f"{args.image_folder}_output"
if not exists(output_folder):
    mkdir(output_folder)

logging.info(f"Found {len(file_list)} files...")
# Counter
class_map = {}
skipped_files = 0

for file in file_list:
    if file.endswith(".json"):
        print(f"Skipping file {file}...")
        skipped_files += 1
        continue
    logging.info(f"Checking file {file}...")
    filename = os.path.splitext(file)
    split_name = filename[0].split("-")
    class_name = f"{split_name[0]}-{split_name[1]}"
    class_name = class_name.lower()
    
    if class_name not in class_map:
        class_map[class_name] = 0
    else:
        class_map[class_name] += 1
    if not exists(f"{output_folder}/{class_name}"):
        mkdir(f"{output_folder}/{class_name}")
    
    
    logging.info(f"Copying files with class {class_name}...")
    shutil.copy(f"{args.image_folder}/{file}", f"{output_folder}/{class_name}/{class_map[class_name]}.jpeg")

logging.info(class_map)
logging.log(56, "Script finished!")