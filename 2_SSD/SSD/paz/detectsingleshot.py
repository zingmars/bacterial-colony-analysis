
from .processor import Processor
from .box2d import Box2D
import numpy as np
import colorsys
import random
import cv2
import tensorflow as tf
import logging

def denormalize_box(box, image_shape):
    """Scales corner box coordinates from normalized values to image dimensions

    # Arguments
        box: Numpy array containing corner box coordinates.
        image_shape: List of integers with (height, width).

    # Returns
        returns: box corner coordinates in image dimensions
    """
    x_min, y_min, x_max, y_max = box[:4]
    height, width = image_shape
    x_min = int(x_min * width)
    y_min = int(y_min * height)
    x_max = int(x_max * width)
    y_max = int(y_max * height)
    return (x_min, y_min, x_max, y_max)

class DenormalizeBoxes2D(Processor):
    """Denormalizes boxes shapes to be in accordance to the original
    image size.

    # Arguments:
        image_size: List containing height and width of an image.
    """
    def __init__(self):
        super(DenormalizeBoxes2D, self).__init__()

    def call(self, image, boxes2D):
        shape = image.shape[:2]
        for box2D in boxes2D:
            box2D.coordinates = denormalize_box(box2D.coordinates, shape)
        return boxes2D

def lincolor(num_colors, saturation=1, value=1, normalized=False):
    """Creates a list of RGB colors linearly sampled from HSV space with
        randomised Saturation and Value.

    # Arguments
        num_colors: Int.
        saturation: Float or `None`. If float indicates saturation.
            If `None` it samples a random value.
        value: Float or `None`. If float indicates value.
            If `None` it samples a random value.
        normalized: Bool. If True, RGB colors are returned between [0, 1]
            if False, RGB colors are between [0, 255].

    # Returns
        List, for which each element contains a list with RGB color
    """
    RGB_colors = []
    hues = [value / num_colors for value in range(0, num_colors)]
    for hue in hues:

        if saturation is None:
            saturation = random.uniform(0.6, 1)

        if value is None:
            value = random.uniform(0.5, 1)

        RGB_color = colorsys.hsv_to_rgb(hue, saturation, value)
        if not normalized:
            RGB_color = [int(color * 255) for color in RGB_color]
        RGB_colors.append(RGB_color)
    return RGB_colors

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
def put_text(image, text, point, scale, color, thickness):
    """Puts text on given image.

    # Arguments
        image: Array of shape `(H, W, 3)`, input image.
        text: String, text to show.
        point: Tuple, coordinate of top corner of text.
        scale: Float, scale of text.
        color: Tuple, RGB color coordinates.
        thickness: Int, text thickness.

    # Returns
        Array: Image of shape `(H, W, 3)` with text.
    """
    image = cv2.putText(
        image, text, point, FONT, scale, color, thickness, LINE)
    return image

def draw_rectangle(image, corner_A, corner_B, color, thickness):
    """ Draws a filled rectangle from ``corner_A`` to ``corner_B``.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        corner_A: List of length two indicating ``(y, x)`` openCV coordinates.
        corner_B: List of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer/openCV Flag. Thickness of rectangle line.
            or for filled use cv2.FILLED flag.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with rectangle.
    """
    return cv2.rectangle(
        image, tuple(corner_A), tuple(corner_B), tuple(color), thickness)

class DrawBoxes2D(Processor):
    """Draws bounding boxes from Boxes2D messages.

    # Arguments
        class_names: List of strings.
        colors: List of lists containing the color values
        weighted: Boolean. If ``True`` the colors are weighted with the
            score of the bounding box.
        scale: Float. Scale of drawn text.
    """
    def __init__(self, class_names=None, colors=None,
                 weighted=False, scale=0.7, with_score=True):
        self.class_names = class_names
        self.colors = colors
        self.weighted = weighted
        self.with_score = with_score
        self.scale = scale

        if (self.class_names is not None and
                not isinstance(self.class_names, list)):
            raise TypeError("Class name should be of type 'List of strings'")

        if (self.colors is not None and
                not all(isinstance(color, list) for color in self.colors)):
            raise TypeError("Colors should be of type 'List of lists'")

        if self.colors is None:
            self.colors = lincolor(len(self.class_names))

        if self.class_names is not None:
            self.class_to_color = dict(zip(self.class_names, self.colors))
        else:
            self.class_to_color = {None: self.colors, '': self.colors}
        super(DrawBoxes2D, self).__init__()

    def call(self, image, boxes2D):
        count = 0
        for box2D in boxes2D:
            if (box2D.score > 0.7):
                x_min, y_min, x_max, y_max = box2D.coordinates
                class_name = box2D.class_name
                color = self.class_to_color[class_name]
                if self.weighted:
                    color = [int(channel * box2D.score) for channel in color]
                if self.with_score:
                    text = '{:0.2f}, {}'.format(box2D.score, class_name)
                if not self.with_score:
                    text = '{}'.format(class_name)
                put_text(image, text, (x_min, y_min - 10), self.scale, color, 1)
                draw_rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                count = count + 1
        logging.info(f"Drew {count} boxes!")
        return image

class WrapOutput(Processor):
    """Wraps arguments in dictionary

    # Arguments
        keys: List of strings representing the keys used to wrap the inputs.
            The order of the list must correspond to the same order of
            inputs (''args'').
    """
    def __init__(self, keys):
        if not isinstance(keys, list):
            raise ValueError('``order`` must be a list')
        self.keys = keys
        super(WrapOutput, self).__init__()

    def call(self, *args):
        return dict(zip(self.keys, args))

def predict(x, model, preprocess=None, postprocess=None):
    """Preprocess, predict and postprocess input.
    # Arguments
        x: Input to model
        model: Callable i.e. Keras model.
        preprocess: Callable, used for preprocessing input x.
        postprocess: Callable, used for postprocessing output of model.

    # Note
        If model outputs a tf.Tensor is converted automatically to numpy array.
    """
    if preprocess is not None:
        x = preprocess(x)
    y = model(x)
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    if postprocess is not None:
        y = postprocess(y)
    return y

class Predict(Processor):
    """Perform input preprocessing, model prediction and output postprocessing.

    # Arguments
        model: Class with a ''predict'' method e.g. a Keras model.
        preprocess: Function applied to given inputs.
        postprocess: Function applied to outputted predictions from model.
    """
    def __init__(self, model, preprocess=None, postprocess=None):
        super(Predict, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def call(self, x):
        return predict(x, self.model, self.preprocess, self.postprocess)

def to_corner_form(boxes):
    """Transform from center coordinates to corner coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    center_x, center_y = boxes[:, 0:1], boxes[:, 1:2]
    W, H = boxes[:, 2:3], boxes[:, 3:4]
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return np.concatenate([x_min, y_min, x_max, y_max], axis=1)

def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode default boxes into the ground truth boxes

    # Arguments
        loc: Numpy array of shape `(num_priors, 4)`.
        priors: Numpy array of shape `(num_priors, 4)`.
        variances: List of two floats. Variances of prior boxes.

    # Returns
        decoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    center_x = predictions[:, 0:1] * priors[:, 2:3] * variances[0]
    center_x = center_x + priors[:, 0:1]
    center_y = predictions[:, 1:2] * priors[:, 3:4] * variances[1]
    center_y = center_y + priors[:, 1:2]
    W = priors[:, 2:3] * np.exp(predictions[:, 2:3] * variances[2])
    H = priors[:, 3:4] * np.exp(predictions[:, 3:4] * variances[3])
    boxes = np.concatenate([center_x, center_y, W, H], axis=1)
    boxes = to_corner_form(boxes)
    return np.concatenate([boxes, predictions[:, 4:]], 1)

class DecodeBoxes(Processor):
    """Decodes bounding boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        variances: List of two float values.
    """
    def __init__(self, prior_boxes, variances=[0.1, 0.1, 0.2, 0.2]):
        self.prior_boxes = prior_boxes
        self.variances = variances
        super(DecodeBoxes, self).__init__()

    def call(self, boxes):
        decoded_boxes = decode(boxes, self.prior_boxes, self.variances)
        return decoded_boxes

class BoxesWithOneHotVectorsToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages given boxes
    with scores as one hot vectors.

    # Arguments
        arg_to_class: List, of classes.

    # Properties
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class):
        self.arg_to_class = arg_to_class
        super(BoxesWithOneHotVectorsToBoxes2D, self).__init__()

    def call(self, boxes):
        boxes2D = []
        for box in boxes:
            score = np.max(box[4:])
            class_arg = np.argmax(box[4:])
            class_name = self.arg_to_class[class_arg]
            boxes2D.append(Box2D(box[:4], score, class_name))
        return boxes2D

class BoxesToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages given no
    class names and score.

    # Arguments
        default_score: Float, score to set.
        default_class: Str, class to set.

    # Properties
        default_score: Float.
        default_class: Str.

    # Methods
        call()
    """
    def __init__(self, default_score=1.0, default_class=None):
        self.default_score = default_score
        self.default_class = default_class
        super(BoxesToBoxes2D, self).__init__()

    def call(self, boxes):
        boxes2D = []
        for box in boxes:
            boxes2D.append(
                Box2D(box[:4], self.default_score, self.default_class))
        return boxes2D

class BoxesWithClassArgToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages given boxes
    with class argument.

    # Arguments
        default_score: Float, score to set.
        arg_to_class: List, of classes.

    # Properties
        default_score: Float.
        arg_to_class: List.

    # Methods
        call()
    """
    def __init__(self, arg_to_class, default_score=1.0):
        self.default_score = default_score
        self.arg_to_class = arg_to_class
        super(BoxesWithClassArgToBoxes2D, self).__init__()

    def call(self, boxes):
        boxes2D = []
        for box in boxes:
            class_name = self.arg_to_class[box[-1]]
            boxes2D.append(Box2D(box[:4], self.default_score, class_name))
        return boxes2D

class ToBoxes2D(Processor):
    """Transforms boxes from dataset into `Boxes2D` messages.

    # Arguments
        class_names: List of class names ordered with respect to the
            class indices from the dataset ``boxes``.
        one_hot_encoded: Bool, indicating if scores are one hot vectors.
        default_score: Float, score to set.
        default_class: Str, class to set.
        box_method: Int, method to convert boxes to ``Boxes2D``.

    # Properties
        one_hot_encoded: Bool.
        box_processor: Callable.

    # Methods
        call()
    """
    def __init__(
            self, class_names=None, one_hot_encoded=False,
            default_score=1.0, default_class=None, box_method=0):
        if class_names is not None:
            arg_to_class = dict(zip(range(len(class_names)), class_names))
        self.one_hot_encoded = one_hot_encoded
        method_to_processor = {
            0: BoxesWithOneHotVectorsToBoxes2D(arg_to_class),
            1: BoxesToBoxes2D(default_score, default_class),
            2: BoxesWithClassArgToBoxes2D(arg_to_class, default_score)}
        self.box_processor = method_to_processor[box_method]
        super(ToBoxes2D, self).__init__()

    def call(self, boxes):
        return self.box_processor(boxes)

# Source https://github.com/oarriaga/paz/blob/master/paz/pipelines/detection.py
class DetectSingleShot(Processor):
    """Single-shot object detection prediction.

    # Arguments
        model: Keras model.
        class_names: List of strings indicating the class names.
        preprocess: Callable, pre-processing pipeline.
        postprocess: Callable, post-processing pipeline.
        score_thresh: Float between [0, 1]
        nms_thresh: Float between [0, 1].
        variances: List, of floats.
        draw: Boolean. If ``True`` prediction are drawn in the
            returned image.
    """
    def __init__(self, model, class_names, score_thresh, nms_thresh,
                 preprocess=None, postprocess=None,
                 variances=[0.1, 0.1, 0.2, 0.2], draw=True):
        self.model = model
        self.class_names = class_names
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.variances = variances
        self.draw = draw
        '''
        if preprocess is None:
            preprocess = SSDPreprocess(model)
        if postprocess is None:
            postprocess = SSDPostprocess(
                model, class_names, score_thresh, nms_thresh)
        '''
        super(DetectSingleShot, self).__init__()
        self.predict = Predict(self.model, preprocess, postprocess)
        self.denormalize = DenormalizeBoxes2D()
        self.draw_boxes2D = DrawBoxes2D(self.class_names)
        self.wrap = WrapOutput(['image', 'boxes2D'])

    def call(self, image):
        boxes2D = self.predict(image)
        boxes2D = self.denormalize(image, boxes2D)
        if self.draw:
            image = self.draw_boxes2D(image, boxes2D)
        return self.wrap(image, boxes2D)


def apply_non_max_suppression(boxes, scores, iou_thresh=.45, top_k=200):
    """Apply non maximum suppression.

    # Arguments
        boxes: Numpy array, box coordinates of shape `(num_boxes, 4)`
            where each columns corresponds to x_min, y_min, x_max, y_max.
        scores: Numpy array, of scores given for each box in `boxes`.
        iou_thresh: float, intersection over union threshold for removing
            boxes.
        top_k: int, number of maximum objects per class.

    # Returns
        selected_indices: Numpy array, selected indices of kept boxes.
        num_selected_boxes: int, number of selected boxes.
    """

    selected_indices = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
        return selected_indices
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)
    remaining_sorted_box_indices = np.argsort(scores)
    remaining_sorted_box_indices = remaining_sorted_box_indices[-top_k:]

    num_selected_boxes = 0
    while len(remaining_sorted_box_indices) > 0:
        best_score_args = remaining_sorted_box_indices[-1]
        selected_indices[num_selected_boxes] = best_score_args
        num_selected_boxes = num_selected_boxes + 1
        if len(remaining_sorted_box_indices) == 1:
            break

        remaining_sorted_box_indices = remaining_sorted_box_indices[:-1]

        best_x_min = x_min[best_score_args]
        best_y_min = y_min[best_score_args]
        best_x_max = x_max[best_score_args]
        best_y_max = y_max[best_score_args]

        remaining_x_min = x_min[remaining_sorted_box_indices]
        remaining_y_min = y_min[remaining_sorted_box_indices]
        remaining_x_max = x_max[remaining_sorted_box_indices]
        remaining_y_max = y_max[remaining_sorted_box_indices]

        inner_x_min = np.maximum(remaining_x_min, best_x_min)
        inner_y_min = np.maximum(remaining_y_min, best_y_min)
        inner_x_max = np.minimum(remaining_x_max, best_x_max)
        inner_y_max = np.minimum(remaining_y_max, best_y_max)

        inner_box_widths = inner_x_max - inner_x_min
        inner_box_heights = inner_y_max - inner_y_min

        inner_box_widths = np.maximum(inner_box_widths, 0.0)
        inner_box_heights = np.maximum(inner_box_heights, 0.0)

        intersections = inner_box_widths * inner_box_heights
        remaining_box_areas = areas[remaining_sorted_box_indices]
        best_area = areas[best_score_args]
        unions = remaining_box_areas + best_area - intersections
        intersec_over_union = intersections / unions
        intersec_over_union_mask = intersec_over_union <= iou_thresh
        remaining_sorted_box_indices = remaining_sorted_box_indices[
            intersec_over_union_mask]

    return selected_indices.astype(int), num_selected_boxes

def nms_per_class(box_data, nms_thresh=.45, epsilon=0.01,
                  conf_thresh=0.5, top_k=200):
    """Applies non-maximum-suppression per class.

    # Arguments
        box_data: Array of shape `(num_prior_boxes, 4 + num_classes)`.
        nms_thresh: Float, Non-maximum suppression threshold.
        epsilon: Float, Filter scores with a lower confidence
            value before performing non-maximum supression.
        conf_thresh: Float, Filter out boxes with a confidence value
            lower than this.
        top_k: Int, Maximum number of boxes per class outputted by nms.

    # Returns
        Array of shape `(num_boxes, 4+ num_classes)`.
    """
    decoded_boxes, class_predictions = box_data[:, :4], box_data[:, 4:]
    num_classes = class_predictions.shape[1]
    non_suppressed_boxes = np.array(
        [], dtype=float).reshape(0, box_data.shape[1])
    for class_arg in range(num_classes):
        mask = class_predictions[:, class_arg] >= epsilon
        scores = class_predictions[:, class_arg][mask]
        if len(scores) == 0:
            continue
        boxes = decoded_boxes[mask]
        indices, count = apply_non_max_suppression(
            boxes, scores, nms_thresh, top_k)
        selected_indices = indices[:count]
        classes = class_predictions[mask]
        selections = np.concatenate(
            (boxes[selected_indices],
             classes[selected_indices]), axis=1)
        filter_mask = selections[:, 4 + class_arg] >= conf_thresh
        non_suppressed_boxes = np.concatenate(
            (non_suppressed_boxes, selections[filter_mask]), axis=0)
    return non_suppressed_boxes

class NonMaximumSuppressionPerClass(Processor):
    """Applies non maximum suppression per class.

    # Arguments
        nms_thresh: Float between [0, 1].
        epsilon: Float between [0, 1].
        conf_thresh: Float between [0, 1].
    """
    def __init__(self, nms_thresh=.45, epsilon=0.01, conf_thresh=0.5):
        self.nms_thresh = nms_thresh
        self.epsilon = epsilon
        self.conf_thresh = conf_thresh
        super(NonMaximumSuppressionPerClass, self).__init__()

    def call(self, boxes):
        boxes = nms_per_class(
            boxes, self.nms_thresh, self.epsilon, self.conf_thresh)
        return boxes