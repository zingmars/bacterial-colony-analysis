import cv2
import numpy as np
import logging
import argparse
from os import listdir, mkdir
from os.path import isfile, join, exists

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Edge detector")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--mode", default="canny", type=str, help="Which algorithm to use - sobel, canny, roberts")
argParser.add_argument("--draw", default=True, type=bool, help="Should the image be drawn on screen?")
argParser.add_argument("--export", default=None, type=str, help="Set this parameter to a name of a directory to export pictures")
args = argParser.parse_args()

if (args.mode not in ["sobel", "canny", "roberts"]):
    raise Exception("Please choose between sobel and canny algorithms")

logging.log(55, 'Script started.')

logging.info("Creating output directory...")
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]
output_folder = args.export
if output_folder and not exists(output_folder):
    mkdir(output_folder)

for file in fileList:
    logging.info(f"Working on picture {file}!")
    frame = cv2.imread(f"{args.image_folder}/{file}", cv2.IMREAD_GRAYSCALE)
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    if args.mode == "sobel":
        #sobelx = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        #sobely = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=frame, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        if args.draw:
            cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.export:
            cv2.imwrite(f"{output_folder}/{file}", sobelxy)
    elif args.mode == "canny":
        #edges = cv2.Canny(frame, 50, 150, apertureSize=5, L2gradient=True)
        #edges = cv2.Canny(frame, 5, 20, apertureSize=3, L2gradient=True)
        edges = cv2.Canny(frame, 10, 30, apertureSize=3, L2gradient=True)
        
        # Autodetect. Doesn't work for our use case
        # Source: https://towardsdatascience.com/easy-method-of-edge-detection-in-opencv-python-db26972deb2d
        '''
        sigma = 0.33
        md = np.median(frame)
        lower_value = int(max(0, (1.0-sigma) * md))
        upper_value = int(min(255, (1.0+sigma) * md))
        edges = cv2.Canny(frame, lower_value, upper_value)
        '''
        if args.draw:
            cv2.imshow('Canny', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.export:
            cv2.imwrite(f"{output_folder}/{file}", edges)
    elif args.mode == "roberts":
        # 2x2 Robert's cross (original)
        roberts_cross_v = np.array( [[1, 0 ], [0,-1 ]] )
        roberts_cross_h = np.array( [[ 0, 1 ], [ -1, 0 ]] )
        # 3x3 Robert's cross
        '''
        roberts_cross_v = np.array( [[ 0, 0, 0 ],
                                    [ 0, 1, 0 ],
                                    [ 0, 0,-1 ]] )

        roberts_cross_h = np.array( [[ 0, 0, 0 ],
                                    [ 0, 0, 1 ],
                                    [ 0,-1, 0 ]] )
        '''
        img_roberts_cross_v = cv2.filter2D(frame, -1, roberts_cross_v)
        img_roberts_cross_h = cv2.filter2D(frame, -1, roberts_cross_h)
        edges = cv2.addWeighted(np.square(img_roberts_cross_v), 0.5, np.square(img_roberts_cross_h), 1, 0)
        if args.draw:
            cv2.imshow('Robert\'s cross', edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.export:
            cv2.imwrite(f"{output_folder}/{file}", edges)

logging.log(56, "Script finished!")