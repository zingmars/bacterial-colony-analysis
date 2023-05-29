import cv2
import logging
import argparse
import os
import json
from math import floor

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Dataset mask visualizer (AGAR compatible)")
argParser.add_argument("--image-folder", default="move_img", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--fit-to-screen", default=False, type=int, help="Set this to true if you want the image to fit your screen")
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

from os import listdir
from os.path import isfile, join
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]

for file in fileList:
    if file.endswith(".json"):
        continue
    filename = os.path.splitext(file)
    logging.info(f"Working on picture {file}!")
    frame = cv2.imread(f"{args.image_folder}/{file}")

    json_file = open(f"{args.image_folder}/{filename[0]}.json")
    annotation = json.load(json_file)
    json_file.close()

    for label in annotation["labels"]:
        cv2.rectangle(frame, (label["x"], label["y"]), (label["x"] + label["width"], label["y"] + label["height"]), (0, 0, 255), 1)
        if ("class" in label):
                cv2.putText(frame, label["class"], (label["x"], label["y"]), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5, color=(0,255,0))

    if args.fit_to_screen:
        if h == -1 or w == -1:
            logging.error("Cannot continue: need to scale but don't know system resolution")
            quit(-1)
        hScale = h / frame.shape[0]
        wScale = w / frame.shape[1] 
        scale = hScale if hScale < wScale else wScale
        if scale == 0 or scale*100 > 100:
            scale = 100
            logging.info(f"Picture fits the screen. Not scaling it down!")
        else:
            scale = int(floor(scale*100))
            logging.info(f"Resizing pic from {frame.shape[0]}x{frame.shape[1]} to fit {w}x{h} using (scaled to {scale}%)")
            frame = cv2.resize(frame, (int(frame.shape[0]*scale/100), int(frame.shape[1]*scale/100)))
            logging.info(f"New picture size: {frame.shape[0]}x{frame.shape[1]}!")
    cv2.imshow("Detected Circles", frame)
    keyInput = cv2.waitKey(0)
    if keyInput == 27: #Esc - Save and exit program
        logging.info("Quitting!")
        cv2.destroyAllWindows()
        logging.log(56, "Script finished!")
        quit()
    cv2.destroyAllWindows()

logging.log(56, "Script finished!")