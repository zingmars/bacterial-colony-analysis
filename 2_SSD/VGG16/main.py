import logging
from model import Model
import numpy as np
from math import floor
import argparse
import json
from calculate_memory_size import keras_model_memory_usage_in_bytes

parser = argparse.ArgumentParser(description="Neural network for bacterial analysis")
parser.add_argument("--mode", default="test", type=str, help="Mode in which the script is run (train/evaluate/test).")
parser.add_argument("--data-path", default="test_vgg", type=str, help="If the script is in train mode, this should be the path to the dataset. If the script is in evaluate mode, this should be the path to the file.")
parser.add_argument("--class-count", default=14, type=int, help="For training - the amount of classes the should be trained one")
parser.add_argument("--class-name-file", default="classes.json", type=str, help="For evaluating - name and id of the classes. Can be generated using generate-class-map.py!")
parser.add_argument("--train-epochs", default=500, type=int, help="For training - how many epochs (cycles) should the train run for")
parser.add_argument("--log-to-file", default=False, type=bool, help="Whether the script will log to file or console")
parser.add_argument("--log-name", default="log.txt", type=str, help="Name of the file to which the script will write logs to")
parser.add_argument("--train-log-name", default="history.csv", type=str, help="Name of the file to which TensorFlow will write training logs to")
parser.add_argument("--train-final-log-name", default="history_full.csv", type=str, help="Name of the file to which TensorFlow will write training logs to after training")
parser.add_argument("--weights-file", default="vgg16.h5", type=str, help="Path to the weights file")
args = parser.parse_args()

if args.log_to_file:
    logging.basicConfig(filename=args.log_name, level=logging.getLevelName('DEBUG'), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
else:
    logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!")
logging.addLevelName(56, "Goodbye!")

# Set up the model
load_dataset = True if args.mode == "train" else False
model = Model(args.data_path, load_dataset=load_dataset)
model.build_regional_proposal_network(train_classes=args.class_count, train_vgg=True)
model.compile_model()
model.load_model(args.weights_file)

#print(f"Model uses at least {keras_model_memory_usage_in_bytes(model.get_model(), batch_size = 14)} bytes")

if args.mode == "train":
    model.train_regional_proposal_network(epochs = args.train_epochs, save_log = True, log_name=args.train_log_name, final_log_name=args.train_final_log_name)
elif args.mode == "evaluate":
    prediction = model.predict_using_regional_proposal_network(args.data_path)
    f = open(args.class_name_file, 'r')
    classes = json.load(f)
    f.close()
    # Hard-coded classes for testing
    '''
    classes = { 
        '0': "bsubtilis-tsa", 
        '1': "calbicans-tsa", 
        '2': "ecoli-tbx", 
        '3': "ecoli-tsa", 
        '4': "enterobacteriaceae-vrbg", 
        '5': "hela-dmem", 
        '6': "listeriaspp-aloa", 
        '7': "mafam-pca", 
        '8': "paeruginosa-tsa", 
        '9': "saureus-braidparker", 
        '10': "saureus-tsa",
        '11': "staphylococcus aureus-sheepblod",
        '12': "streptococcusmutans-bhi",
        '13': "v79-dmem"}
    '''
    idx = np.argmax(prediction)
    print(f"Neural network reports that the image passed in is of class {classes[f'{idx}']} with {int(floor(prediction[0][f'{idx}']*100))}% certainty!")
elif args.mode == "test":
    # Load in images and check them
    from os import listdir
    from os.path import isfile, join, splitext
    results = []
    classDict = {}

    fileList = [f for f in listdir(args.data_path) if isfile(join(args.data_path, f))]
    for file in fileList:
        full_path = f"{args.data_path}/{file}"
        filename = splitext(file)
        split_filename = filename[0].split("-")
        class_name = split_filename[0]
        agar = split_filename[1] 
        # 2 is a random number, we can ignore it

        f = open(args.class_name_file, 'r')
        classes = json.load(f)
        f.close()

        prediction = model.predict_using_regional_proposal_network(full_path)
        idx = np.argmax(prediction)

        results.append({"filename": file, "results": prediction.tolist(), "class-id": f'{idx}', "class-name": classes[f'{idx}'], "actual-class": f"{class_name}-{agar}"})
        
        if classes[f'{idx}'] not in classDict:
            classDict[classes[f'{idx}']] = 1
        else:
            classDict[classes[f'{idx}']] = classDict[classes[f'{idx}']] + 1

    with open("results.json", "w") as outfile:
        outfile.write(json.dumps(results))
    with open("results_classdict.json", "w") as outfile:
        outfile.write(json.dumps(classDict))

logging.log(56, "Script finished!")