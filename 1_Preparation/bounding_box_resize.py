import argparse
import logging
import os
import json
import cv2

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")
argParser = argparse.ArgumentParser(description="Picture and bounding box resize utility (AGAR compatible)")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images which should be resized")
argParser.add_argument("--width", default=512, type=int, help="Picture width")
argParser.add_argument("--height", default=512, type=int, help="Picture height")
args = argParser.parse_args()

logging.log(55, 'Script started.')

logging.info("Creating output directory...")
from os import listdir, mkdir
from os.path import isfile, join, exists
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]
output_folder = f"{args.image_folder}_resized"
if not exists(output_folder):
    mkdir(output_folder)

for file in fileList:
    logging.info(f"Resizing bounding boxes in {file} to {args.width}x{args.height}...")
    if file.endswith(".jpg"):
        continue
    filename = os.path.splitext(file)

    # Load the picture to get its size
    image = cv2.imread(f"{args.image_folder}/{filename[0]}.jpg")
    height = image.shape[0]
    width = image.shape[1]

    json_file = open(f"{args.image_folder}/{filename[0]}.json")
    annotation = json.load(json_file)
    json_file.close()

    scaleX = args.width / width
    scaleY = args.height / height

    # Note: Slight distortion might happen if the aspect ratio changes
    for label in annotation["labels"]:
        label["x"] = int(label["x"] * scaleX)
        label["width"] = int(label["width"] * scaleX)
        label["y"] = int(label["y"] * scaleY)
        label["height"] = int(label["height"] * scaleY)

    # Save resized picture alongside the resized annotations
    image = cv2.resize(image, (args.width, args.height), interpolation = cv2.INTER_AREA)
    cv2.imwrite(f"{output_folder}/{filename[0]}.jpg", image)
    with open(f"{output_folder}/{filename[0]}.json", 'w') as file:
        file.write(json.dumps(annotation))

    # DEBUG
    '''
    for label in annotation["labels"]:
        cv2.rectangle(image, (label["x"], label["y"]), (label["x"] + label["width"], label["y"] + label["height"]), (0, 0, 255), 1)
    cv2.imshow("Detected Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

logging.log(56, "Script finished!")