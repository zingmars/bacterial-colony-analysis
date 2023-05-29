import json
import os
import logging
import numpy as np
import random
from math import ceil

# Loads data. Assumes standard keras format, where each file is split into folders whose name is the class name
class DataLoader():
    def __init__(self, path=None, data = None, class_names = None, args_to_class = None):
        if path is None:
            logging.error("No image path provided for data loader!")
            raise Exception("Please provide image path for the data loader!")
        self.path = path
        self.data = data
        self.images_path = None
        self.arg_to_class = args_to_class
        self.class_names = class_names

    # Disable background - enable room warmer mode (it breaks the model)
    def load_data(self, class_names_from_json = False, split = 0.2, enable_background=True):
        subfolders = [ f.name for f in os.scandir(self.path) if f.is_dir() ]

        # Store class names
        if class_names_from_json:
            # Read in class names
            with open('classes.json', 'r') as f:
                self.arg_to_class = json.load(f) # Zip returns
            self.class_names = []
            for value in self.arg_to_class.values():
                self.class_names.append(value)
        else:
            # Alternatively, just extract folder names
            self.class_names = subfolders
            self.args_to_class = {}
            counter = 1 if enable_background else 0
            for folder in subfolders:
                self.args_to_class[counter] = folder
                counter += 1

        data = []
        # Extract picture data
        for folder in subfolders:
            path = f"{self.path}/{folder}"
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            self.class_to_arg = {value: key for key, value in self.args_to_class.items()}
            # Files are split as follows: image file.jpeg, image file.json. Both need to exist.
            if len(files) > 0:
                for file in files:
                    # Skip either json or jpg files and then use the other as base
                    if file.endswith(".jpg"):
                        continue
                    base_filename = os.path.splitext(file)
                    base_file_path = f"{path}/{base_filename[0]}"
                    if not os.path.exists(f"{base_file_path}.jpg"):
                        logging.error(f"Found a json file for {base_filename[0]}, but could not find the accompanying image!")
                        raise Exception("Could not find image file for json file!")
                    with open(f"{base_file_path}.json", 'r') as f:
                        annotation = json.load(f)
                    box_data = []
                    for label in annotation["labels"]:
                        box_data.append([label["x"], label["y"], label["x"] + label["width"], label["y"] + label["height"], self.class_to_arg[folder]])
                    #data.append({"image": f"{base_file_path}.jpg", "boxes": np.asarray(box_data), "class": folder})
                    data.append({"image": f"{base_file_path}.jpg", "boxes": np.asarray(box_data)})
        
        if enable_background:
            self.class_names.insert(0, "background")
            self.args_to_class[0] = 'background'
            self.class_to_arg['background'] = 0
        logging.info(f"Found {len(self.class_names)} classes!")

        # Split data into training and validation data
        random.shuffle(data)
        data_size = len(data) - 1
        validation_size = int(ceil(data_size * split))
        training_data = data[0:data_size-validation_size]
        validation_data = data[data_size-validation_size:]

        # Returns split training and validation data
        return (DataLoader(self.path, training_data, class_names=self.class_names, args_to_class=self.args_to_class), DataLoader(self.path, validation_data, class_names=self.class_names, args_to_class=self.args_to_class))
    
    def get_data(self):
        return self.data

    def get_args(self):
        return self.arg_to_class