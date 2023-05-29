import cv2
import numpy as np
import logging
import argparse
import os
import json
from tkinter import *
from tkinter import messagebox
from math import floor, ceil

logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

argParser = argparse.ArgumentParser(description="Bounding box annotation tool (AGAR compatible)")
argParser.add_argument("--image-folder", default="images", type=str, help="Path to the directory that contains images from which circles should be extracted")
argParser.add_argument("--scale-to-screen", default=True, type=bool, help="(down)scale pictures to screen")
argParser.add_argument("--class-input", default=True, type=bool, help="Set this to False to disable class input!")
argParser.add_argument("--class-file", default="classes.json", type=str, help="List of classes! File should be a json file with all of the class names as a list in a variable called 'names'!")
args = argParser.parse_args()

logging.log(55, 'Script started.')

logging.info("Commands:")
logging.info("Esc: Save and quit")
logging.info("Enter: Confirm current box")
logging.info("s: save current data")
logging.info("n: save and move to the next picture")
logging.info("u: undo the last bounding box")
logging.info("z: Zoom in around the area where the current selection is")
logging.info("c: Close the zoom window")
logging.info("x: Cancel current selection")
logging.info("m: mark for deletion")
logging.info("Use mouse to draw. Click on point 1, then on point 2. Second point should be to the right and below the first point.")

from os import listdir, mkdir
from os.path import isfile, join, exists
fileList = [f for f in listdir(args.image_folder) if isfile(join(args.image_folder, f))]
output_folder = f"{args.image_folder}_output"
if not exists(output_folder):
    logging.info("Creating output directory...")
    mkdir(output_folder)

if args.class_input:
    logging.info("Preparing classes...")
    if os.path.isfile(args.class_file):
        json_file = open(args.class_file)
        annotation = json.load(json_file)
        json_file.close()
    else:
        logging.warning("Class file not loaded! While you can still manually set class IDs, you'll manually need to set their names later!")
        annotation = {"names": [ ]}
    labelString = ""
    counter = 0
    for label in annotation["names"]:
        labelString += f"{counter}: {label}\n"
        counter += 1
    labelInputHeight = len(annotation["names"]) * 25 + 50
else:
    annotation = {"names": [ ]}

if args.scale_to_screen:
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
        args.scale_to_screen = False
    elif platform.system() == "Darwin":
        logging.error("Mac OS is not currently supported. Will not scale the window!")
        args.scale_to_screen = False
    elif platform.system() == "Java":
        logging.error("lmao what the hell bro")
        args.scale_to_screen = False

# Allows inputting class labels when saving the file!
lastClass = 0
class InputForm():
    global lastClass
    result = 0
    def input_class_label(self):
        root = Tk()
        root.attributes("-topmost", True)
        root.title("Enter class label")
        root.geometry(f'200x{labelInputHeight}')
        root.resizable(False, False)

        Label(root, text="Input label:").grid(column=0, row=0)
        Label(root, text="Available labels:").grid(column=0, row=1)
        Label(root, text=labelString).grid(column=0, row=2)

        inp = Entry(root)
        inp.grid(column=1, row=0)
        inp.insert("0", str(lastClass))

        def return_variable(*args):
            global lastClass
            try:
                self.result = int(inp.get())
                lastClass = inp.get()
                #if (try_inp > len(annotation["names"])):
                    #raise Exception("ID not in class list")
            except:
                messagebox.ERROR("Input error", "Invalid input, please enter the id of the class label!")
                return
            root.destroy()
        root.bind('<Escape>', return_variable)
        root.bind('<Return>', return_variable)
        btn = Button(root, text="Save!", command=return_variable)
        btn.grid(column=1, row=1)
        root.focus_force()

        root.mainloop()

# Tracks coordinates from mouseCallback and sets coordinates depending on the state
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, jx, jy, drawing, confirmed
    if event == cv2.EVENT_LBUTTONDOWN and confirmed:
      drawing = True
      ix = x
      iy = y
    elif event == cv2.EVENT_MOUSEMOVE and confirmed and drawing:
        if x > ix and y > iy:
            jx = x
            jy = y
        else:
            jx = -1
            jy = -1
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if x > ix and y > iy:
            jx = x
            jy = y
            confirmed = False
        else: # Reset
            ix = -1
            jy = -1
            confirmed = True

def fix_rectangle_zoomed(event, x, y, flags, param):
    global ix, iy, jx, jy, confirmed
    # Maps local window coordinates to global coordinates, and lets us move the bounding box
    # The zoomed window is a 300x300 segment scaled into 600x600
    # Meaning the scaling factor is x2, and the halfpoint is 300x300
    if not confirmed and ix != -1 and iy != -1 and jx != -1 and jy != -1:
        # Absolute values mapped to source image
        # Calculate by how much is the zoom window scaled relative to the actual picture
        # Then calculate the relative coordinates from the zoom window the the actual image
        scaleW = (jx - ix + 300) / 600 
        scaleH = (jy - iy + 300) / 600
        ax = int(ceil(ix - 150 + (x*scaleW)))
        ay = int(ceil(iy - 150 + (y*scaleH)))
        #print(f"DEBUG: Scale X: {scaleW}, Scale Y: {scaleH}, newX: {ax}, newY: {ay}, event: {event}")
        # Relative to center
        #ix = int(ceil(ix - ((300 - x) / 2)))
        #iy = int(ceil(iy - ((300 - y) / 2)))
        #jx = int(floor(jx - ((300 - x) / 2)))
        #jy = int(floor(jy - ((300 - y) / 2)))

        # BUG: Will crash if the values go out of picture's bounds
        if event == cv2.EVENT_LBUTTONDOWN:
            if ax < jx and ay < jy:
                ix = ax
                iy = ay
        elif event == cv2.EVENT_RBUTTONDOWN:
            if ax > ix and ay > iy:
                jx = ax
                jy = ay           

def save_bounding_boxes(file_path, bounding_boxes):
    with open(file_path, 'w') as file:
        file.write(json.dumps(bounding_boxes))

def load_bounding_boxes(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

for file in fileList:
    # Rectangle coordinates and program state.
    # Since we can't pass by reference (thanks Python) we define them here
    # and then rely on the global keyword to set these values
    ix,iy = -1,-1
    jx,jy = -1,-1
    drawing = False # Whether we are currently drawing a rectangle or not
    confirmed = True # Whether the last rectangle was saved or not

    filename = os.path.splitext(file)
    if os.path.exists(f"{output_folder}/{filename[0]}.skip"):
        logging.info(f"Skipping file {args.image_folder}/{file}: File already processed!")
        continue
    logging.info(f"Loading file {args.image_folder}/{file}...")

    data_file_path = f"{output_folder}/{filename[0]}.json"
    if os.path.exists(data_file_path):
        logging.info("Loading existing bounding box list...")
        bounding_boxes = load_bounding_boxes(data_file_path)
    else:
        logging.info("Creating a new bounding box list...")
        bounding_boxes = { "classes": [], "labels": [], "count": 0}
        save_bounding_boxes(data_file_path, bounding_boxes)

    frame = cv2.imread(f"{args.image_folder}/{file}")

    scale = 100
    negScalePercentage = 1
    if args.scale_to_screen:
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
            negScalePercentage = 100-scale
            logging.info(f"Resizing pic from {frame.shape[0]}x{frame.shape[1]} to fit {w}x{h} using (scaled to {scale}%)")
            frame = cv2.resize(frame, (int(frame.shape[0]*scale/100), int(frame.shape[1]*scale/100)))
            logging.info(f"New picture size: {frame.shape[0]}x{frame.shape[1]}!")

    cv2.namedWindow(filename[0])
    cv2.setMouseCallback(filename[0], draw_rectangle)

    zoomed = False
    while True:
        # Copy to a framebuffer to avoid keeping garbage on screen
        frameBuffer = frame.copy()

        # Re-draw existing bounding boxes
        for box in bounding_boxes["labels"]:
            x1 = int(box["x"] * scale / 100)
            y1 = int(box["y"] * scale / 100)
            x2 = int((box["x"] + box["width"]) * scale / 100)
            y2 = int((box["y"] + box["height"]) * scale / 100)
            cv2.rectangle(frameBuffer, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if ("class" in box):
                cv2.putText(frameBuffer, box["class"], (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5, color=(0,255,0))
        # Draw current selection
        if ix != -1 and iy != -1:
            if jy != -1 and jy != -1:
                cv2.rectangle(frameBuffer, (ix, iy), (jx, jy), (0, 0, 255), 2)
            else:
                cv2.circle(frameBuffer, (ix, iy), 0, color=(0,0,255), thickness=5)

        cv2.imshow(filename[0], frameBuffer)

        if zoomed:
            if ix != -1 and iy != -1 and jx != -1 and jy != -1:
                x1 = ix - 150
                if x1 < 0:
                    x1 = 0
                y1 = iy - 150
                if y1 < 0:
                    y1 = 0
                x2 = jx + 150
                if x2 > frameBuffer.shape[0]:
                    x2 = frameBuffer.shape[0]
                y2 = jy + 150
                if y2 > frameBuffer.shape[1]:
                    y2 = frameBuffer.shape[1]
                #print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
                try:
                    zoomBuffer = frameBuffer[y1:y2, x1:x2]
                    zoomBuffer = cv2.resize(zoomBuffer, (600, 600))
                except:
                    zoomBuffer = np.zeros([600, 600, 3], dtype=np.uint8)
            else:
                zoomBuffer = np.zeros([600, 600, 3], dtype=np.uint8)
            cv2.imshow("Zoom", zoomBuffer)

        # Mouse input handled by mousecallback!

        keyInput = cv2.waitKey(100) # Limits loop to 10fps
        if keyInput == 27: #Esc - Save and exit program
            logging.info("Quitting!")
            cv2.destroyAllWindows()
            save_bounding_boxes(data_file_path, bounding_boxes)
            logging.log(56, "Script finished!")
            quit()
        elif keyInput == 13: #Enter - confirm coordinates
            x1 = ix if scale == 100 else int(ceil(ix + (ix / scale * negScalePercentage)))
            y1 = iy if scale == 100 else int(ceil(iy + (iy / scale * negScalePercentage)))
            x2 = jx-ix if scale == 100 else int(ceil((jx-ix) + ((jx-ix) / scale * negScalePercentage)))
            y2 = jy-iy if scale == 100 else int(ceil((jy-iy) + ((jy-iy) / scale * negScalePercentage)))
            
            if (args.class_input):
                inp_form = InputForm()
                inp_form.input_class_label()
                class_label = int(inp_form.result)
                if (class_label < len(annotation["names"])):
                    str_class_label = annotation["names"][class_label]
                else:
                    str_class_label = class_label
            else:
                str_class_label = 0

            bounding_boxes["labels"].append({"x": x1, "y": y1, "width": x2, "height": y2, "scale": scale, "negScale": negScalePercentage, "class": str_class_label})

            logging.info(f"Saved a new bounding box: {bounding_boxes['labels'][-1]}!")
            ix = -1
            iy = -1
            jx = -1
            jy = -1
            confirmed = True

            save_bounding_boxes(data_file_path, bounding_boxes)
        elif keyInput == ord("x"):
            ix = -1
            iy = -1
            jx = -1
            jy = -1
            confirmed = True
            #zoomed = False
            drawing = False
        elif keyInput == ord("s"):
            logging.info("Saving data to disk...")
            save_bounding_boxes(data_file_path, bounding_boxes)
        elif keyInput == ord("u"):
            if (len(bounding_boxes["labels"]) > 0):
                bounding_boxes["labels"].pop()
                logging.info("Removed last bounding box from screen!")
        elif keyInput == ord("n"):
            if len(bounding_boxes["labels"]) > 0:
                bounding_boxes["classes"] = [x["class"] for x in bounding_boxes["labels"]]
            bounding_boxes["count"] = len(bounding_boxes["labels"])
            bounding_boxes["scale"] = scale
            logging.info(f"Finshing with file {file}. File has {bounding_boxes['count']} bounding boxes!")
            save_bounding_boxes(data_file_path, bounding_boxes)
            with open(f"{output_folder}/{filename[0]}.skip", 'w') as file:
                file.write("")
            cv2.destroyAllWindows()
            break
        elif keyInput == ord("z"):
            zoomed = True
            cv2.namedWindow("Zoom")
            cv2.setMouseCallback("Zoom", fix_rectangle_zoomed)
        elif keyInput == ord("c"):
            zoomed = False
            cv2.destroyWindow("Zoom")
        elif keyInput == ord("m"):
            logging.info(f"Marking file {file} for deletion and moving to the next file!")
            with open(f"{output_folder}/{filename[0]}.skip", 'w') as file:
                file.write("")
            with open(f"{output_folder}/{filename[0]}.deleteme", 'w') as file:
                file.write("")
            cv2.destroyAllWindows()
            break

logging.info("No more images left!")
logging.log(56, "Script finished!")
