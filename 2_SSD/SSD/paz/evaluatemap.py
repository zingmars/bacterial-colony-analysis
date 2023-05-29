import os
import numpy as np
from keras.callbacks import Callback
from .evaluationfunctions import compute_matches, calculate_average_precisions, calculate_relevance_metrics

# Source: https://github.com/oarriaga/paz/blob/master/paz/evaluation/detection.py
def evaluateMAP(detector, dataset, class_to_arg, iou_thresh=0.5,
                use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    Arguments:
        dataset: List of dictionaries containing 'image' as key and a
            numpy array representing an image as value.
        detector : Function for performing inference
        class_to_arg: Dict. of class names and their id
        iou_thresh: Float indicating intersection over union threshold for
            assigning a prediction as correct.
    # Returns:
    """
    positives, score, match = compute_matches(
        dataset, detector, class_to_arg, iou_thresh)
    precision, recall = calculate_relevance_metrics(positives, score, match)
    average_precisions = calculate_average_precisions(
        precision, recall, use_07_metric)
    return {'ap': average_precisions, 'map': np.nanmean(average_precisions)}

# Source https://github.com/oarriaga/paz/blob/c6a884326c73775a27b792ce91b11c0b3d98bf08/paz/optimization/callbacks.py#L86
# Changed the dataset part to use loaded data!
class EvaluateMAP(Callback):
    """Evaluates mean average precision (MAP) of an object detector.

    # Arguments
        data_manager: Data manager and loader class. See ''paz.datasets''
            for examples.
        detector: Tensorflow-Keras model.
        period: Int. Indicates how often the evaluation is performed.
        save_path: Str.
        iou_thresh: Float.
    """
    def __init__(
            self, data_manager, detector, period, save_path, iou_thresh=0.5):
        super(EvaluateMAP, self).__init__()
        self.data_manager = data_manager
        self.detector = detector
        self.period = period
        self.save_path = save_path
        self.dataset = data_manager.get_data()
        self.iou_thresh = iou_thresh
        self.class_names = self.data_manager.class_names
        self.class_dict = {}
        for class_arg, class_name in enumerate(self.class_names):
            self.class_dict[class_name] = class_arg

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.period == 0:
            result = evaluateMAP(
                self.detector,
                self.dataset,
                self.class_dict,
                iou_thresh=self.iou_thresh,
                use_07_metric=True)

            result_str = 'mAP: {:.4f}\n'.format(result['map'])
            metrics = {'mAP': result['map']}
            for arg, ap in enumerate(result['ap']):
                if arg == 0 or np.isnan(ap):  # skip background
                    continue
                metrics[self.class_names[arg]] = ap
                result_str += '{:<16}: {:.4f}\n'.format(
                    self.class_names[arg], ap)
            print(result_str)

            # Saving the evaluation results
            filename = os.path.join(self.save_path, 'MAP_Evaluation_Log.txt')
            with open(filename, 'a') as eval_log_file:
                eval_log_file.write('Epoch: {}\n{}\n'.format(
                    str(epoch), result_str))