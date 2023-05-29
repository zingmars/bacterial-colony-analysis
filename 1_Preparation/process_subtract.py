import cv2
import argparse
import logging
import os
from os import mkdir
from os.path import exists

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Image subtractor")
argParser.add_argument("--image1", default="images_output/2.jpg.threshold.jpg", type=str, help="Path to the image that will be used as a base for subtraction")
argParser.add_argument("--image2", default="images_output/1.jpg.threshold.jpg", type=str, help="Path to the image that will be subtracted from the base")
argParser.add_argument("--output-folder", default="images_output", type=str, help="Folder to which the result should be saved")
argParser.add_argument("--draw", default=True, type=bool, help="Should the image be drawn on screen?")
args = argParser.parse_args()

logging.log(55, 'Script started.')

if not args.output_folder:
    raise Exception("Please specify the output folder!")
if not exists(args.output_folder):
    mkdir(args.output_folder)

image1 = cv2.imread(f"{args.image1}")
image2 = cv2.imread(f"{args.image2}")
image3 = image1-image2
path = f"{args.output_folder}/{os.path.basename(args.image1)}.subtracted.jpg"
cv2.imwrite(path, image3)

if args.draw:
    cv2.imshow("Result", image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

subtracted = cv2.subtract(image1, image2)
path = f"{args.output_folder}/{os.path.basename(args.image1)}.subtracted_2.jpg"
cv2.imwrite(path, subtracted)
if args.draw:
    cv2.imshow("Result", subtracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

logging.log(56, "Script finished!")