import cv2
import logging
import argparse
import os
import csv
import yaml
from shapely.geometry import Polygon
from math import ceil
from os import listdir, mkdir
from os.path import isfile, join, exists
import json
from math import floor

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="YOLO output visualizer")
argParser.add_argument("--image-folder", default="dataset-yolo", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--labels-folder", default=None, type=str, help="Path to the labels folder (leave as default if labels are in the same directory)")
argParser.add_argument("--actual-labels-folder", default=None, type=str, help="Path to the ground truth label folder (default disables the check)")
argParser.add_argument("--config-file", default="data.yaml", type=str, help="Name of the dataset file in the image folder. Used to extract classes!")
argParser.add_argument("--draw", default=False, type=bool, help="Should the image be drawn on screen?")
argParser.add_argument("--draw-original", default=True, type=bool, help="If actual labels folder is set, this will draw the ground truth on the image as well.")
argParser.add_argument("--save-folder", default="dataset-yolo/output", type=str, help="Directory to which the images should be saved. Unset if you don't want the images to be saved to disk!")
argParser.add_argument("--resize-percentage", default=None, type=int, help="Set this value to resize the image to a certain percentage")
argParser.add_argument("--fit-to-screen", default=False, type=int, help="Set this to true if you want the image to fit your screen. Not compatible with resize-precentage.")
argParser.add_argument("--save-statistics", default=False, type=bool, help="Should the script gather statistics?")
args = argParser.parse_args()

logging.log(55, 'Script started.')
if args.resize_percentage and args.fit_to_screen:
    raise Exception("Please select either resize percentage OR fit-to-screen!")

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

logging.info("Loading config file...")
classList = []
yamlFile = f"{args.image_folder}\\{args.config_file}"
if isfile(yamlFile):
    with open(yamlFile, "r") as stream:
        dataset_info = yaml.safe_load(stream)
    logging.info(f"Loaded a dataset file defininf {dataset_info['nc']} classes!")
    classList = dataset_info['names']

logging.info("Preparing folder & file lists; creating output folder if necessary...")
baseImagefolder = args.image_folder
baseLabelsfolder = args.image_folder if args.labels_folder is None else f"{args.labels_folder}"
baseActualLabelFolder = args.actual_labels_folder
baseSaveFolder = args.save_folder
if baseSaveFolder and not exists(baseSaveFolder):
    mkdir(baseSaveFolder)

def denormalize_data(line: list['str']):
    clazz = int(line[0])
    center_x = int(float(line[1]) * width)
    center_y = int(float(line[2]) * height)
    bbox_width = int(float(line[3]) * width) 
    bbox_height = int(float(line[4]) * height)
    if len(line) == 6:
        score = float(line[5])
    else:
        score = -1
    x1 = int(center_x - bbox_width/2)
    y1 = int(center_y - bbox_height/2)
    x2 = int(center_x + ceil(bbox_width/2))
    y2 = int(center_y + ceil(bbox_height/2))
    return [clazz, x1, y1, x2, y2, score]

def calculate_intersection(coordinateSet1: list, coordinateSet2: list):
    boxes = []
    for coordinateSet in [coordinateSet1, coordinateSet2]:
        boxCoordinates = ((coordinateSet[1], coordinateSet[2]), (coordinateSet[3], coordinateSet[2]), (coordinateSet[3], coordinateSet[4]), (coordinateSet[1], coordinateSet[4]), (coordinateSet[1], coordinateSet[2]))
        boxes.append(Polygon(boxCoordinates))
    if (boxes[0].intersects(boxes[1])):
        intersection = boxes[0].intersection(boxes[1]).area
        union =  boxes[0].union(boxes[1]).area
        if union != 0:
            return (intersection/union)*100
        else:
            return 0
    else:
        return 0

if args.save_statistics:
    stat_intersecting = []
    stat_totalGround = []
    stat_totalDetected = []
    stat_imageCount = 0
    stat_correct = []
    stat_classification = []

fileList = [f for f in listdir(baseImagefolder) if isfile(join(baseImagefolder, f))]
for file in fileList:
    if file == args.config_file:
        continue
    stat_imageCount += 1
    filename = os.path.splitext(file)
    logging.info(f"Working on picture {file}!")
    img = cv2.imread(f"{baseImagefolder}/{file}")
    height, width, _ = img.shape

    annotation_file = f"{baseLabelsfolder}/{filename[0]}.txt" 
    annotationClassDict = {}
    coordinates = []
    hasScores = True
    
    if not isfile(annotation_file):
        logging.error(f"Cannot open file {file}: No annotation file found!")
        continue

    with open(annotation_file) as csvfile:
        annotation = csv.reader(csvfile, delimiter=" ")
        # Format: class_id center_x center_y bbox_width bbox_height score
        counter = 0
        for line in annotation:
            deNormalizedData = denormalize_data(line)
            if len(classList) > deNormalizedData[0]:
                deNormalizedData[0] = classList[deNormalizedData[0]]
            if deNormalizedData[0] not in annotationClassDict:
                annotationClassDict[deNormalizedData[0]] = 1
            else:
                annotationClassDict[deNormalizedData[0]] += 1
            if deNormalizedData[5] == -1:
                hasScores = False
            coordinates.append(deNormalizedData)
            counter += 1
        logging.debug(f"Neural network detected {counter} boxes!")

    # Filter out overlapping boxes
    if not hasScores:
        logging.warning(f"Current file `{file}` does not have confidence scores in the labels file! If there are overlapping boxes, they will not be filtered!")
    else:
        intersectingBoxes = []
        # TODO: Optimize - don't make the same check multiple times!
        for coordinateSet1 in coordinates:
            for coordinateSet2 in coordinates:
                if coordinateSet1 != coordinateSet2:
                    intersection_percent = calculate_intersection(coordinateSet1, coordinateSet2)
                    if intersection_percent > 65:
                        logging.warning(f"Found intersecting bounding boxes ({intersection_percent}). Selecting the one with the highest score!")
                        if coordinateSet1[5] != -1 and coordinateSet2[5] != -1 and coordinateSet1[5] > coordinateSet2[5]:
                            intersectingBoxes.append(coordinateSet1)
                        else:
                            intersectingBoxes.append(coordinateSet2)
        logging.info(f"Found {len(intersectingBoxes)} intersecting boxes! Removing them")
        if len(intersectingBoxes) > 0:
            if args.save_statistics:
                stat_intersecting.append(len(intersectingBoxes) / 2)
            for intersectingBox in intersectingBoxes:
                if intersectingBox[0] in annotationClassDict:
                    annotationClassDict[intersectingBox[0]] -= 1
                    if annotationClassDict[intersectingBox[0]] == 0:
                        del annotationClassDict[intersectingBox[0]]
                    if intersectingBox in coordinates:
                        coordinates.remove(intersectingBox)

    logging.info(f"Detected {len(annotationClassDict)} bacteria classes. Full breakdown is as follows:")
    for key, value in annotationClassDict.items():
        logging.info(f"{key}: {value} bacteria!")

    # Draw the boxes
    if args.draw or baseSaveFolder is not None:
        for box in coordinates:
            cv2.rectangle(img, (box[1], box[2]), (box[3], box[4]), (0, 255, 0), 2)
            cv2.putText(img, str(box[0]), (box[1], box[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,255,0))
        cv2.putText(img, f"Found {len(coordinates)} bacteria", (0, height-50) ,cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=(0,255,0))
    stat_totalDetected.append(len(coordinates))
    
    if (args.draw or baseSaveFolder is not None) and baseActualLabelFolder is not None:
        with open(f"{baseActualLabelFolder}/{filename[0]}.txt") as csvfile:
            intersecting_count = 0
            counter = 0
            if args.save_statistics:
                correct = 0
                correct_classification = 0
            annotation = csv.reader(csvfile, delimiter=" ")
            for line in annotation:
                intersects = False
                deNormalizedData = denormalize_data(line)
                if len(classList) > deNormalizedData[0]:
                    deNormalizedData[0] = classList[deNormalizedData[0]]
                coordinateCounter = 0
                for box in coordinates:
                    coordinates[coordinateCounter].append(True)
                    intersection_percent = calculate_intersection(deNormalizedData, box)
                    if intersection_percent > 75:
                        box.append(True)
                        intersects = True
                        if args.save_statistics:
                            correct += 1
                        break
                    coordinateCounter += 1
                if not intersects:
                    if args.draw_original:
                        cv2.rectangle(img, (deNormalizedData[1], deNormalizedData[2]), (deNormalizedData[3], deNormalizedData[4]), (0, 0, 255), 2)
                        cv2.putText(img, deNormalizedData[0], (deNormalizedData[1], deNormalizedData[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,255))
                else:
                    intersecting_count += 1
                    if deNormalizedData[0] is not box[0]:
                        cv2.putText(img, deNormalizedData[0], (box[1], box[4]+13), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0,0,255))
                if intersects and deNormalizedData[0] is box[0] and args.save_statistics:
                    correct_classification += 1
                counter += 1
            cv2.putText(img, f"Ground truth has {counter} bacteria, of which {intersecting_count} intersect (IoU>75%)", (0, height-100) ,cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=2, color=(0,0,255))
            if args.save_statistics:
                stat_totalGround.append(counter)
                stat_correct.append(correct)
                stat_classification.append(correct_classification)

    # Export
    if args.draw:
        buffer = img.copy()
        if args.fit_to_screen:
            if h == -1 or w == -1:
                logging.error("Cannot continue: need to scale but don't know system resolution")
                quit(-1)
            hScale = h / buffer.shape[0]
            wScale = w / buffer.shape[1] 
            scale = hScale if hScale < wScale else wScale
            if scale == 0 or scale*100 > 100:
                scale = 100
                logging.info(f"Picture fits the screen. Not scaling it down!")
            else:
                scale = int(floor(scale*100))
                logging.info(f"Resizing pic from {buffer.shape[0]}x{buffer.shape[1]} to fit {w}x{h} using (scaled to {scale}%)")
                buffer = cv2.resize(buffer, (int(buffer.shape[0]*scale/100), int(buffer.shape[1]*scale/100)))
                logging.info(f"New picture size: {buffer.shape[0]}x{buffer.shape[1]}!")
        elif args.resize_percentage:
            buffer = cv2.resize(buffer, (args.width / int(args.resize_percentage), args.height / int(args.resize_percentage)), interpolation = cv2.INTER_AREA)
        cv2.imshow("Detected Circles", buffer)
        keyInput = cv2.waitKey(0)
        if keyInput == 27: #Esc - Save and exit program
            logging.info("Quitting!")
            cv2.destroyAllWindows()
            logging.log(56, "Script finished!")
            quit()
        cv2.destroyAllWindows()
    if baseSaveFolder is not None:
        cv2.imwrite(f"{baseSaveFolder}/{file}", img)
        output = {
            "class_count": len(coordinates),
            "classes": list(annotationClassDict.keys()),
            "labels": []
        }
        for box in coordinates:
            output["labels"].append({ "x": box[1], "y": box[2], "x2": box[3], "y2": box[4], "class": box[0], "score": box[5], "height": box[4]-box[2], "width": box[3]-box[1] })
            if baseActualLabelFolder:
                output["labels"][-1]["intersects"] = box[6] if len(box) > 6 else False
        with open(f"{baseSaveFolder}/{filename[0]}.json", 'w') as f:
            f.write(json.dumps(output))

if args.save_statistics:
    with open("statistics.json", 'w') as f:
        stats = {
                "total_files": stat_imageCount,
                "intersecting_boxes": stat_intersecting,
                "images_with_intersecting": len(stat_intersecting),
                "total_intersecting_boxes": sum(stat_intersecting),
                "boxes": stat_totalDetected,
                "total_boxes": sum(stat_totalDetected),
                "ground": stat_totalGround,
                "total_ground": sum(stat_totalGround),
                "correct": stat_correct,
                "total_correct": sum(stat_correct),
                "correct_classification": stat_classification,
                "total_correct_classification": sum(stat_classification)
                }
        f.write(json.dumps(stats))

logging.log(56, "Script finished!")