import logging
import argparse
import os
import json

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Class map generator")
argParser.add_argument("--image-folder", default="dataset", type=str, help="Path to the directory that contains folder of images in Keras format")
argParser.add_argument("--output-file", default="classes.json", type=str, help="Name of the file that the class information will be saved to")
args = argParser.parse_args()

logging.log(55, 'Script started.')

folder_dict = {}
counter = 0

subfolders = [ f.name for f in os.scandir(args.image_folder) if f.is_dir() ]
for folder in subfolders:
    folder_dict[counter] = folder
    counter += 1
with open(args.output_file, "w") as outfile:
    outfile.write(json.dumps(folder_dict))

logging.log(56, "Script finished!")