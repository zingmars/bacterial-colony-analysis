import cv2
import numpy as np
import argparse
import logging
from os import listdir, mkdir
from os.path import isfile, join, exists

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Circle detector and extractor")
#argParser.add_argument("--image", default="1.jpg", type=str, help="Path to the image from which circles should be extracted")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--min-size", default="1", type=int, help="Minimum radius for something to be considered a circle using contour method. Use this to only extract huge circles.")
argParser.add_argument("--resize", default=False, type=bool, help="Resize picture? Useful for debugging")
argParser.add_argument("--resize-percentage", default=80, type=int, help="To what percentage the picture should be resized to")
argParser.add_argument("--method", default=2, type=int, help="Which methode to use? (0 - blob detector, 1 - contours, 2 - Huggs)")
argParser.add_argument("--min-radius", default=500, type=int, help="Min circle radius for Hough's method. Set it to a large value if you have big pictures, small value if not.")
argParser.add_argument("--max-radius", default=1000, type=int, help="Max circle radius for Hough's method. Set it to a large value if you have big pictures, small value if not.")
argParser.add_argument("--min-dist", default=10000, type=int, help="Minimum distance between circles. Set this to a large value to avoid having multiple circles in your picture.")
argParser.add_argument("--canny-threshold", default=25, type=int, help="First parameter for Hough's method. Higher threshold for circles. If the picture has good contrast, set this to higher value to make detection faster.")
argParser.add_argument("--votes-threshold", default=50, type=int, help="Second parameter for Hough's method. Accumulator threshold for circles. If the picture has good contrast, set this to higher value to make detection faster.")
argParser.add_argument("--cuda", default=False, type=bool, help="Whether CUDA should be used for Hough's method. Requires OpenCV with CUDA support to be installed. Faster, but also less precise.")
argParser.add_argument("--reuse", default=False, type=bool, help="Whether the results from the first picture should be re-used instead of calculating the circle every time. If you're using the pictures that are the same size, this will considerably speed up the process!")
argParser.add_argument("--draw", default=True, type=bool, help="Should the image be drawn on screen?")
argParser.add_argument("--export", default=None, type=str, help="Set this parameter to a name of a directory to export pictures")
args = argParser.parse_args()

logging.log(55, 'Script started.')

logging.info("Creating output directory...")
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]
output_folder = args.export
if output_folder and not exists(output_folder):
    mkdir(output_folder)

found = False
for file in fileList:
    logging.info(f"Reading image {file}...")
    imageWithColour = cv2.imread(f"{args.image_folder}/{file}")
    image = cv2.imread(f"{args.image_folder}/{file}", cv2.IMREAD_GRAYSCALE)
  
    # Resize
    if args.resize:
        logging.info(f"Resizing picture to {args.resize_percentage}% of the original...")
        width = int(image.shape[1] * args.resize_percentage / 100)
        height = int(image.shape[0] * args.resize_percentage / 100)
        # TODO: Use 1 original pic and then just convert to grayscale rather than running this twice
        image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        imageWithColour = cv2.resize(imageWithColour, (width, height), interpolation = cv2.INTER_AREA)

    logging.info("Finding contours...")
    if args.method == 0:
        # Couldn't get it to work...
        logging.info("Using blob detector")
        logging.warn("NOTE: The blob detector is highly unreliable")
        params = cv2.SimpleBlobDetector_Params()
        #params.filterByArea = True
        #params.minArea = 500
        #params.maxArea = 100000
        #params.filterByInertia = False
        #params.minInertiaRatio = 0.01
        #params.maxInertiaRatio = 1.0
        #params.filterByConvexity = True
        #params.minConvexity = 0.87
        #params.filterByCircularity = True
        #params.minCircularity = 0.1
        #params.maxCircularity = 1.2
        # Loop count
        params.minThreshold = 10
        #params.maxThreshold = 1000
        detector = cv2.SimpleBlobDetector_create(params)
        logging.info("Detecting blobs...")
        keypoints = detector.detect(image)
        imageWithColour = cv2.drawKeypoints(imageWithColour, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            r = int(keypoint.size / 2)
            cv2.circle(image, (x, y), r, (0, 255, 0, 2))
        if args.draw:
            cv2.imshow("Detected Circles", imageWithColour)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.export:
            cv2.imwrite(f"{output_folder}/{file}", imageWithColour)
    if args.method == 1:
        # Works with some datasets, not others
        logging.info("Using contour detection method")
        image = cv2.medianBlur(image,5)
        ret,thresh = cv2.threshold(image,127,255,0)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(image, contours, -1, (0,0,255), 3)
        # Make image coloured again
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            if (radius>args.min_size):
                cv2.circle(imageWithColour,center,radius,(0,0,255),2)
        if args.draw:
            cv2.imshow("Detected Circles", imageWithColour)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.export:
            cv2.imwrite(f"{output_folder}/{file}", imageWithColour)
    if args.method == 2:
        # Really slow with the wrong parameters, but mostly works
        logging.info("Using Hough Circles method")
        
        if not found:
            # CUDA is not recommended. It uses a slightly modified version of Hough Circles Detector that is not as precise as the original. I could not get it to work as well as the CPU version
            if args.cuda:
                blurredImage = cv2.GaussianBlur(image, (5, 5), 0) # Hides tiny details that we might not need to speed up circle detection
                gpuImage = cv2.cuda_GpuMat(blurredImage)
                hcd = cv2.cuda.createHoughCirclesDetector(1, args.min_dist, cannyThreshold=args.canny_threshold, votesThreshold=args.votes_threshold, minRadius=args.min_radius, maxRadius=args.max_radius, maxCircles = 1)
                cuda_circles = hcd.detect(gpuImage)
                circles = cuda_circles.download()
            else:
                blurredImage = cv2.GaussianBlur(image, (5, 5), 0) # Hides tiny details that we might not need to speed up circle detection
                circles = cv2.HoughCircles(blurredImage, cv2.HOUGH_GRADIENT, 1, args.min_dist, param1=args.canny_threshold, param2=args.votes_threshold, minRadius=args.min_radius, maxRadius=args.max_radius)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
            else:
                logging.error(f"Could not convert {file}: chosen threshold is not applicable for this image!")
                continue
            found = args.reuse

        logging.info("Extracting circle pictures...")
        (x, y, r) = circles[0]
        # Debug lines
        #cv2.line(image, (x-r, y), (x+r, y), (255, 0, 0), 2)
        #cv2.line(image, (x, y-r), (x, y+r), (255, 0, 0), 2)
        if args.draw:
            cv2.imshow("Detected Circles", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        imageWithColour = imageWithColour[y-r:y+r, x-r:x+r]
        len, width, _ = imageWithColour.shape
        if len == 0 or width == 0:
            logging.error("Could not create a picture: found circle would eliminate whole exis!")
            continue
        if args.export:
            cv2.imwrite(f"{output_folder}/{file}", imageWithColour)

logging.log(56, "Script finished!")