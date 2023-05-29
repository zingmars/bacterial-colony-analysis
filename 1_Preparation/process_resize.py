import argparse
import logging
import cv2
from os import listdir, mkdir
from os.path import isfile, join, exists

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")
argParser = argparse.ArgumentParser(description="Picture resize utility")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images which should be resized")
argParser.add_argument("--output-folder", default="images_resized", type=str, help="Set this parameter to a name of a directory to export pictures")
argParser.add_argument("--width", default=512, type=int, help="Picture width")
argParser.add_argument("--height", default=512, type=int, help="Picture height")
args = argParser.parse_args()

logging.log(55, 'Script started.')

logging.info("Creating output directory...")
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]
output_folder = {args.output_folder}
if not output_folder:
    raise Exception("Please specify the output folder!")
if not exists(output_folder):
    mkdir(output_folder)

for file in fileList:
    logging.info(f"Resizing picture {file} to {args.width}x{args.height}...")
    image = cv2.imread(f"{args.image_folder}/{file}")
    image = cv2.resize(image, (args.width, args.height), interpolation = cv2.INTER_AREA)
    cv2.imwrite(f"{output_folder}/{file}", image)

logging.log(56, "Script finished!")