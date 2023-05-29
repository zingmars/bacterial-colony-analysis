import cv2
import logging
import argparse
import os
import csv
import yaml
from math import ceil, floor
from os import listdir
from os.path import isfile, join

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Dataset visualizer (YOLOv6 and YOLOv8 compatible)")
argParser.add_argument("--image-folder", default="dataset-yolo", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--config-file", default="data.yaml", type=str, help="Name of the dataset file in the image folder. Used to extract classes!")
argParser.add_argument("--type", default="train", type=str, help="Which dataset (i.e. train/valid/test) should be loaded?")
argParser.add_argument("--fit-to-screen", default=False, type=int, help="Set this to true if you want the image to fit your screen.")
args = argParser.parse_args()

logging.log(55, 'Script started.')

if args.fit_to_screen:
    import platform
    w = -1
    h = -1
    # Get main screen resolution
    if platform.system() == "Windows":
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    elif platform.system() == "Linux":
        logging.error("Linux is not currently supported. Will not scale the window!")
        args.fit_to_screen = False
    elif platform.system() == "Darwin":
        logging.error("Mac OS is not currently supported. Will not scale the window!")
        args.fit_to_screen = False
    elif platform.system() == "Java":
        logging.error("lmao what the hell bro")
        args.fit_to_screen = False

classList = []
yamlFile = f"{args.image_folder}\\{args.config_file}"
if os.isfile(yamlFile):
    with open(yamlFile, "r") as stream:
        dataset_info = yaml.safe_load(stream)
    logging.info(f"Loaded a dataset file defininf {dataset_info['nc']} classes!")
    classList = dataset_info['names']

baseImagefolder = f"{args.image_folder}/images/{args.type}"
baseLabelsfolder = f"{args.image_folder}/labels/{args.type}"

fileList = [f for f in listdir(baseImagefolder) if isfile(join(baseImagefolder, f))]

for file in fileList:
    filename = os.path.splitext(file)
    logging.info(f"Working on picture {file}!")
    img = cv2.imread(f"{baseImagefolder}/{file}")
    height, width, _ = img.shape

    annotation_file = f"{baseLabelsfolder}/{filename[0]}.txt" 

    with open(annotation_file) as csvfile:
        annotation = csv.reader(csvfile, delimiter=" ")
        # Format: class_id center_x center_y bbox_width bbox_height
        for line in annotation:
            clazz = int(line[0])
            center_x = int(float(line[1]) * width)
            center_y = int(float(line[2]) * height)
            bbox_width = int(float(line[3]) * width) 
            bbox_height = int(float(line[4]) * height)

            x1 = int(center_x - bbox_width/2)
            y1 = int(center_y - bbox_height/2)
            x2 = int(center_x + ceil(bbox_width/2))
            y2 = int(center_y + ceil(bbox_height/2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(img, classList[clazz] if clazz < len(classList) else clazz, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5, color=(0,0,255))

    if args.fit_to_screen:
        if h == -1 or w == -1:
            logging.error("Cannot continue: need to scale but don't know system resolution")
            quit(-1)
        hScale = h / img.shape[0]
        wScale = w / img.shape[1] 
        scale = hScale if hScale < wScale else wScale
        if scale == 0 or scale*100 > 100:
            scale = 100
            logging.info(f"Picture fits the screen. Not scaling it down!")
        else:
            scale = int(floor(scale*100))
            logging.info(f"Resizing pic from {img.shape[0]}x{img.shape[1]} to fit {w}x{h} using (scaled to {scale}%)")
            img = cv2.resize(img, (int(img.shape[0]*scale/100), int(img.shape[1]*scale/100)))
            logging.info(f"New picture size: {img.shape[0]}x{img.shape[1]}!")

    cv2.imshow("Detected Circles", img)
    keyInput = cv2.waitKey(0)
    if keyInput == 27: #Esc - Save and exit program
        logging.info("Quitting!")
        cv2.destroyAllWindows()
        logging.log(56, "Script finished!")
        quit()
    cv2.destroyAllWindows()

logging.log(56, "Script finished!")