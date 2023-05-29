# Adapted from https://github.com/oarriaga/paz/blob/master/paz/pipelines/detection.py
from paz.processor import Processor, Squeeze
from paz.detectsingleshot import DenormalizeBoxes2D, DrawBoxes2D, WrapOutput, Predict, ToBoxes2D, DecodeBoxes, NonMaximumSuppressionPerClass
from paz.sequencer import SequentialProcessor
import logging
import cv2
import csv

class SSDPostprocess(SequentialProcessor):
    """Postprocessing pipeline for SSD.

    # Arguments
        model: Keras model.
        class_names: List, of strings indicating the class names.
        score_thresh: Float, between [0, 1]
        nms_thresh: Float, between [0, 1].
        variances: List, of floats.
        class_arg: Int, index of class to be removed.
        box_method: Int, type of boxes to boxes2D conversion method.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 variances=[0.1, 0.1, 0.2, 0.2], class_arg=0, box_method=0):
        super(SSDPostprocess, self).__init__()
        self.add(Squeeze(axis=None))
        self.add(DecodeBoxes(model.prior_boxes, variances))
        #self.add(pr.RemoveClass(class_names, class_arg, renormalize=False))
        self.add(NonMaximumSuppressionPerClass(
            nms_thresh, conf_thresh=score_thresh))
        self.add(ToBoxes2D(class_names, box_method))


# Slighly modified version of SSD class from Paz. It gives us the ability to count!
class DetectSingleShot(Processor):
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 preprocess=None, postprocess=None,
                 variances=[0.1, 0.1, 0.2, 0.2], draw=True, print_count=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw
        self.print_count = print_count
        
        # Don't perform any pre-processing here. It will just ruin the picture for us.
        # PostProcesing
        postprocess = SSDPostprocess(
                model, class_names, score_thresh, nms_thresh)
        
        super(DetectSingleShot, self).__init__()
        self.predict = Predict(self.model, preprocess, postprocess)
        self.denormalize = DenormalizeBoxes2D()
        self.draw_boxes2D = DrawBoxes2D(self.class_names)
        self.wrap = WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        reshapedImage = image.reshape(1,512,512,3)
        boxes2D = self.predict(reshapedImage)
        boxes2D = self.denormalize(image, boxes2D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        if self.print_count:
            logging.info(f"Found {len(boxes2D)} bacteria!")
            cv2.imshow("Detected Circles", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("output.png", image)
            with open('output.csv', 'w', newline='') as outputfile:
                writer = csv.writer(outputfile, delimiter=' ')
                writer.writerows(boxes2D)
        return self.wrap(image, boxes2D)