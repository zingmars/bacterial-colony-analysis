import logging
import argparse
import sys
import numpy as np
from keras.optimizers import SGD
from ssd512 import SSD512
from paz.multiboxloss import MultiBoxLoss
from paz.learningratescheduler import LearningRateScheduler
from paz.evaluatemap import evaluateMAP
from paz.sequencer import ProcessingSequence
from paz.detection import AugmentDetection, TRAIN, VAL
from keras.callbacks import ModelCheckpoint, CSVLogger
from detectsingleshot import DetectSingleShot
from dataloader import DataLoader
from calculate_memory_size import get_model_memory_usage

parser = argparse.ArgumentParser(description="Neural network for bacterial analysis")
parser.add_argument("--mode", default="train", type=str, help="Mode in which the script is run (train/evaluate).")
parser.add_argument("--class-count", default=15, type=int, help="For training - the amount of classes the should be trained (add +1 for the background)")
parser.add_argument("--batch-size", default=32, type=int, help="For training - the amount of classes the should be trained one")
parser.add_argument("--train-epochs", default=200, type=int, help="For training - how many times will the network see the dataset")
parser.add_argument("--learning-rate", default=0.0005, type=float, help="For training - how many times will the network see the dataset")
parser.add_argument("--weights-file", default="ssd.h5", type=str, help="Path to either SSD512 or VGG16 weights file")
parser.add_argument("--log-to-file", default=False, type=bool, help="Whether the script will log to file or console")
parser.add_argument("--log-name", default="log.txt", type=str, help="Name of the file to which the script will write logs to")
parser.add_argument("--train-log-name", default="history.csv", type=str, help="Name of the file to which TensorFlow will write training logs to")
parser.add_argument("--training-data-path", default="dataset", type=str, help="This should be the path to the training dataset.")
parser.add_argument("--image", default="test.jpg", type=str, help="For evaluating - name of the file run the model on.")
parser.add_argument("--trainable-base", default=False, type=bool, help="Whether training will also train the base (VGG) model")
args = parser.parse_args()

if args.log_to_file:
    logging.basicConfig(filename=args.log_name, level=logging.getLevelName('DEBUG'), filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
else:
    logging.basicConfig(level=logging.getLevelName('DEBUG'), format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.addLevelName(55, "Hello!") 
logging.addLevelName(56, "Goodbye!")

logging.log(55, 'Script started.')

if args.mode == "train":
    model = SSD512(args.class_count, ssd_weights=args.weights_file, trainable_base=args.trainable_base)
    loss = MultiBoxLoss()
    metrics = {'boxes': [loss.localization,
                        loss.positive_classification,
                        loss.negative_classification]}
    model.compile(optimizer=SGD(0.001, 0.9), loss=loss.compute_loss, metrics=metrics)

    logging.info(f"NOTE: The model will use {get_model_memory_usage(32, model)}GiB of VRAM!")

    # Load data
    (loader, eval_loader) = DataLoader(args.training_data_path).load_data() # default 80-20 split

    if (len(loader.class_names) != args.class_count):
        logging.error("Dataset has a different amount of classes than provided to the script.")
        logging.error(f"Found {len(loader.class_names)} classes, but the model was set up for {args.class_count} classes.")
        sys.exit(-1)
    
    # PAZ data processing pipeline
    augmentator = AugmentDetection(model.prior_boxes, TRAIN, num_classes=args.class_count)
    validation_augmentator = AugmentDetection(model.prior_boxes, VAL, num_classes=args.class_count)
    normalSequencer = ProcessingSequence(augmentator, args.batch_size, loader.get_data(), as_list=False)
    validationSequencer = ProcessingSequence(validation_augmentator, args.batch_size, eval_loader.get_data(), as_list=False)

    # Keras callbacks
    weights_backup = ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=5)
    logger = CSVLogger(args.train_log_name, separator=",", append=True)
    learning_rate_scheduler = LearningRateScheduler(args.learning_rate, 0.1, [50, 100, 150])
    model.fit(
        normalSequencer,
        epochs=args.train_epochs,
        verbose=1,
        callbacks=[weights_backup, logger, learning_rate_scheduler],
        validation_data=validationSequencer,
        use_multiprocessing=False,
        workers=1)
elif args.mode == "count":
    model = SSD512(args.class_count, ssd_weights=args.weights_file, trainable_base=False)
    loss = MultiBoxLoss()
    metrics = {'boxes': [loss.localization,
                        loss.positive_classification,
                        loss.negative_classification]}
    model.compile(SGD(0.001, 0.9), loss.compute_loss, metrics)
    (loader, eval_loader) = DataLoader(args.training_data_path).load_data()
    detector = DetectSingleShot(model, loader.class_names, 0.01, 0.45, print_count = True)
    result = evaluateMAP(
        detector,
        loader.get_data(),
        loader.get_args(),
        iou_thresh=0.5,
        use_07_metric=True)
elif args.mode == "evaluate":
    # Run count, but also do IoU calculations
    '''
        evaluate = EvaluateMAP(
        eval_loader,
        DetectSingleShot(model, loader.class_names, 0.01, 0.45, print_count=False),
        10,
        "dataset_ssd",
        0.5)
    '''
    model = SSD512(args.class_count, ssd_weights=args.weights_file, trainable_base=False) # Base was trained in the VGG16 model
    loss = MultiBoxLoss()
    metrics = {'boxes': [loss.localization,
                        loss.positive_classification,
                        loss.negative_classification]}
    model.compile(SGD(0.001, 0.9), loss.compute_loss, metrics)
    (loader, eval_loader) = DataLoader(args.training_data_path).load_data()
    detector = DetectSingleShot(model, loader.class_names, 0.01, 0.45, print_count = False)
    result = evaluateMAP(
        detector,
        loader.get_data(),
        loader.get_args(),
        iou_thresh=0.5,
        use_07_metric=True)
    result_str = "mAP: {:.4f}\n".format(result["map"])
    metrics = {'mAP': result["map"]}
    for arg, ap in enumerate(result["ap"]):
        if arg == 0 or np.isnan(ap):  # skip background
            continue
        metrics[loader.get_args()[arg]] = ap
        result_str += "{:<16}: {:.4f}\n".format(loader.get_args()[arg], ap)
    print(result_str)
else:
    logging.critical("Invalid action - %s", args.mode)
    sys.exit(12)

logging.log(56, "Script finished!")