import cv2
import logging
import argparse
from os import listdir, mkdir
from os.path import isfile, join, exists

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Threshold image generator")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--mode", default="adaptive", type=str, help="Which thresholding algorithm to run? adaptive or simple.")
argParser.add_argument("--draw", default=True, type=bool, help="Should the image be drawn on screen?")
argParser.add_argument("--export", default=None, type=str, help="Set this parameter to a name of a directory to export pictures")
args = argParser.parse_args()

if (args.mode not in ["adaptive", "simple"]):
    raise Exception("Please choose between adaptive and simple modes")

logging.log(55, 'Script started.')

logging.info("Creating output directory...")
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]
output_folder = args.export
if output_folder and not exists(output_folder):
    mkdir(output_folder)

for file in fileList:
    logging.info(f"Processing {file}...")
    frame = cv2.imread(f"{args.image_folder}/{file}", cv2.IMREAD_GRAYSCALE)
    frame = cv2.medianBlur(frame, 5)
    if (args.mode == "simple"):
        ret, thresh = cv2.threshold(frame, 1, 255, cv2.THRESH_TOZERO)
    elif (args.mode == "adaptive"):
        thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    if args.draw:
        cv2.imshow("Result", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if args.export:
        cv2.imwrite(f"{output_folder}/{file}", thresh)

logging.log(56, "Script finished!")