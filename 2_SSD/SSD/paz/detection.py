from .sequencer import SequentialProcessor
from .processor import Processor, ControlMap, SequenceWrapper
from .image import load_image, BGR_IMAGENET_MEAN, RGB2BGR, convert_color_space
import numpy as np
import tensorflow as tf

TRAIN = 0
VAL = 1
TEST = 2

class AugmentImage(SequentialProcessor):
    """Augments an RGB image. Disabled to avoid ruining data.
    """
    def __init__(self):
        super(AugmentImage, self).__init__()
        #self.add(pr.RandomContrast())
        #self.add(pr.RandomBrightness())
        #self.add(pr.RandomSaturation())
        #self.add(pr.RandomHue())

class ConvertColorSpace(Processor):
    """Converts image to a different color space.

    # Arguments
        flag: Flag found in ``processors``indicating transform e.g.
            ``pr.BGR2RGB``
    """
    def __init__(self, flag):
        self.flag = flag
        super(ConvertColorSpace, self).__init__()

    def call(self, image):
        return convert_color_space(image, self.flag)

class CastImage(Processor):
    """Cast image to given dtype.

    # Arguments
        dtype: Str or np.dtype
    """
    def __init__(self, dtype):
        self.dtype = dtype
        super(CastImage, self).__init__()

    def call(self, image):
        return image.astype(self.dtype)

class SubtractMeanImage(Processor):
    """Subtract channel-wise mean to image.

    # Arguments
        mean: List of length 3, containing the channel-wise mean.
    """
    def __init__(self, mean):
        self.mean = mean
        super(SubtractMeanImage, self).__init__()

    def call(self, image):
        return image - self.mean

class PreprocessImage(SequentialProcessor):
    """Preprocess RGB image by resizing it to the given ``shape``. If a
    ``mean`` is given it is substracted from image and it not the image gets
    normalized.

    # Arguments
        shape: List of two Ints.
        mean: List of three Ints indicating the per-channel mean to be
            subtracted.
    """
    def __init__(self, shape, mean=BGR_IMAGENET_MEAN):
        super(PreprocessImage, self).__init__()
        # Disabled because doing this would ruin the picture which is not what we want.
        #self.add(pr.ResizeImage(shape))
        #self.add(CastImage(float))
        #if mean is None:
        #self.add(pr.NormalizeImage())
        #else:
        #self.add(SubtractMeanImage(mean))

def to_image_coordinates(boxes, image):
    """Transforms normalized box coordinates into image coordinates.
    # Arguments
        image: Numpy array.
        boxes: Numpy array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        Numpy array of shape `[num_boxes, N]`.
    """
    height, width = image.shape[:2]
    image_boxes = boxes.copy()
    image_boxes[:, 0] = boxes[:, 0] * width
    image_boxes[:, 2] = boxes[:, 2] * width
    image_boxes[:, 1] = boxes[:, 1] * height
    image_boxes[:, 3] = boxes[:, 3] * height
    return image_boxes

class ToImageBoxCoordinates(Processor):
    """Convert normalized box coordinates to image-size box coordinates.
    """
    def __init__(self):
        super(ToImageBoxCoordinates, self).__init__()

    def call(self, image, boxes):
        boxes = to_image_coordinates(boxes, image)
        return image, boxes

def compute_ious(boxes_A, boxes_B):
    """Calculates the intersection over union between `boxes_A` and `boxes_B`.
    For each box present in the rows of `boxes_A` it calculates
    the intersection over union with respect to all boxes in `boxes_B`.
    The variables `boxes_A` and `boxes_B` contain the corner coordinates
    of the left-top corner `(x_min, y_min)` and the right-bottom
    `(x_max, y_max)` corner.

    # Arguments
        boxes_A: Numpy array with shape `(num_boxes_A, 4)`.
        boxes_B: Numpy array with shape `(num_boxes_B, 4)`.

    # Returns
        Numpy array of shape `(num_boxes_A, num_boxes_B)`.
    """
    xy_min = np.maximum(boxes_A[:, None, 0:2], boxes_B[:, 0:2])
    xy_max = np.minimum(boxes_A[:, None, 2:4], boxes_B[:, 2:4])
    intersection = np.maximum(0.0, xy_max - xy_min)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    areas_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    areas_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    # broadcasting for outer sum i.e. a sum of all possible combinations
    union_area = (areas_A[:, np.newaxis] + areas_B) - intersection_area
    union_area = np.maximum(union_area, 1e-8)
    return np.clip(intersection_area / union_area, 0.0, 1.0)

class RandomSampleCrop(Processor):
    """Crops image while adjusting the normalized corner form
    bounding boxes.

    # Arguments
        probability: Float between ''[0, 1]''.
    """
    def __init__(self, probability=0.50, max_trials=50):
        self.probability = probability
        self.max_trials = max_trials
        self.jaccard_min_max = (
            None,
            (0.1, np.inf),
            (0.3, np.inf),
            (0.7, np.inf),
            (0.9, np.inf),
            (-np.inf, np.inf))

    def call(self, image, boxes):

        if self.probability < np.random.rand():
            return image, boxes

        labels = boxes[:, -1:]
        boxes = boxes[:, :4]
        H_original, W_original = image.shape[:2]

        mode = np.random.randint(0, len(self.jaccard_min_max), 1)[0]
        if self.jaccard_min_max[mode] is not None:
            min_iou, max_iou = self.jaccard_min_max[mode]
            for trial_arg in range(self.max_trials):
                W = np.random.uniform(0.3 * W_original, W_original)
                H = np.random.uniform(0.3 * H_original, H_original)
                aspect_ratio = H / W
                if (aspect_ratio < 0.5) or (aspect_ratio > 2):
                    continue
                x_min = np.random.uniform(W_original - W)
                y_min = np.random.uniform(H_original - H)
                x_max = int(x_min + W)
                y_max = int(y_min + H)
                x_min = int(x_min)
                y_min = int(y_min)

                image_crop_box = np.array([x_min, y_min, x_max, y_max])
                overlap = compute_ious(image_crop_box, boxes)
                if ((overlap.max() < min_iou) or (overlap.min() > max_iou)):
                    continue

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                centers_above_x_min = x_min < centers[:, 0]
                centers_above_y_min = y_min < centers[:, 1]
                centers_below_x_max = x_max > centers[:, 0]
                centers_below_y_max = y_max > centers[:, 1]
                mask = (centers_above_x_min * centers_above_y_min *
                        centers_below_x_max * centers_below_y_max)
                if not mask.any():
                    continue

                cropped_image = image[y_min:y_max, x_min:x_max, :].copy()
                masked_boxes = boxes[mask, :].copy()
                masked_labels = labels[mask].copy()
                # should we use the box left and top corner or the crop's
                masked_boxes[:, :2] = np.maximum(masked_boxes[:, :2],
                                                 image_crop_box[:2])
                # adjust to crop (by substracting crop's left,top)
                masked_boxes[:, :2] -= image_crop_box[:2]
                masked_boxes[:, 2:] = np.minimum(masked_boxes[:, 2:],
                                                 image_crop_box[2:])
                # adjust to crop (by substracting crop's left,top)
                masked_boxes[:, 2:] -= image_crop_box[:2]
                return cropped_image, np.hstack([masked_boxes, masked_labels])

        boxes = np.hstack([boxes, labels])
        return image, boxes

def flip_left_right(image):
    return tf.image.flip_left_right(image)

class RandomFlipBoxesLeftRight(Processor):
    """Flips image and implemented labels horizontally.
    """
    def __init__(self):
        super(RandomFlipBoxesLeftRight, self).__init__()

    def call(self, image, boxes):
        if np.random.randint(0, 2):
            boxes = flip_left_right(boxes, image.shape[1])
            image = image[:, ::-1]
        return image, boxes

def to_normalized_coordinates(boxes, image):
    """Transforms coordinates in image dimensions to normalized coordinates.
    # Arguments
        image: Numpy array.
        boxes: Numpy array of shape `[num_boxes, N]` where N >= 4.
    # Returns
        Numpy array of shape `[num_boxes, N]`.
    """
    height, width = image.shape[:2]
    normalized_boxes = boxes.copy()
    normalized_boxes[:, 0] = boxes[:, 0] / width
    normalized_boxes[:, 2] = boxes[:, 2] / width
    normalized_boxes[:, 1] = boxes[:, 1] / height
    normalized_boxes[:, 3] = boxes[:, 3] / height
    return normalized_boxes

class ToNormalizedBoxCoordinates(Processor):
    """Convert image-size box coordinates to normalized box coordinates.
    """
    def __init__(self):
        super(ToNormalizedBoxCoordinates, self).__init__()

    def call(self, image, boxes):
        boxes = to_normalized_coordinates(boxes, image)
        return image, boxes

class AugmentBoxes(SequentialProcessor):
    """Perform data augmentation with bounding boxes.

    # Arguments
        mean: List of three elements used to fill empty image spaces.
    """
    def __init__(self, mean=BGR_IMAGENET_MEAN):
        super(AugmentBoxes, self).__init__()
        # Manual edit: Disabled processing
        self.add(ToImageBoxCoordinates())
        #self.add(pr.Expand(mean=mean))
        #self.add(RandomSampleCrop()) #Broken
        #self.add(RandomFlipBoxesLeftRight())
        self.add(ToNormalizedBoxCoordinates())

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

def match(boxes, prior_boxes, iou_threshold=0.5):
    """Matches each prior box with a ground truth box (box from `boxes`).
    It then selects which matched box will be considered positive e.g. iou > .5
    and returns for each prior box a ground truth box that is either positive
    (with a class argument different than 0) or negative.

    # Arguments
        boxes: Numpy array of shape `(num_ground_truh_boxes, 4 + 1)`,
            where the first the first four coordinates correspond to
            box coordinates and the last coordinates is the class
            argument. This boxes should be the ground truth boxes.
        prior_boxes: Numpy array of shape `(num_prior_boxes, 4)`.
            where the four coordinates are in center form coordinates.
        iou_threshold: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.

    # Returns
        numpy array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """
    ious = compute_ious(boxes, to_corner_form(np.float32(prior_boxes)))
    per_prior_which_box_iou = np.max(ious, axis=0)
    per_prior_which_box_arg = np.argmax(ious, 0)

    #  overwriting per_prior_which_box_arg if they are the best prior box
    per_box_which_prior_arg = np.argmax(ious, 1)
    per_prior_which_box_iou[per_box_which_prior_arg] = 2
    for box_arg in range(len(per_box_which_prior_arg)):
        best_prior_box_arg = per_box_which_prior_arg[box_arg]
        per_prior_which_box_arg[best_prior_box_arg] = box_arg

    matches = boxes[per_prior_which_box_arg]
    matches[per_prior_which_box_iou < iou_threshold, 4] = 0
    return matches

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

def to_center_form(boxes):
    """Transform from corner coordinates to center coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    x_min, y_min = boxes[:, 0:1], boxes[:, 1:2]
    x_max, y_max = boxes[:, 2:3], boxes[:, 3:4]
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    W = x_max - x_min
    H = y_max - y_min
    return np.concatenate([center_x, center_y, W, H], axis=1)

def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Encode the variances from the priorbox layers into the ground truth
    boxes we have matched (based on jaccard overlap) with the prior boxes.

    # Arguments
        matched: Numpy array of shape `(num_priors, 4)` with boxes in
            point-form.
        priors: Numpy array of shape `(num_priors, 4)` with boxes in
            center-form.
        variances: (list[float]) Variances of priorboxes

    # Returns
        encoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    boxes = matched[:, :4]
    boxes = to_center_form(boxes)
    center_difference_x = boxes[:, 0:1] - priors[:, 0:1]
    encoded_center_x = center_difference_x / priors[:, 2:3]
    center_difference_y = boxes[:, 1:2] - priors[:, 1:2]
    encoded_center_y = center_difference_y / priors[:, 3:4]
    encoded_center_x = encoded_center_x / variances[0]
    encoded_center_y = encoded_center_y / variances[1]
    encoded_W = np.log((boxes[:, 2:3] / priors[:, 2:3]) + 1e-8)
    encoded_H = np.log((boxes[:, 3:4] / priors[:, 3:4]) + 1e-8)
    encoded_W = encoded_W / variances[2]
    encoded_H = encoded_H / variances[3]
    encoded_boxes = [encoded_center_x, encoded_center_y, encoded_W, encoded_H]
    return np.concatenate(encoded_boxes + [matched[:, 4:]], axis=1)

class EncodeBoxes(Processor):
    """Encodes bounding boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        variances: List of two float values.
    """
    def __init__(self, prior_boxes, variances=[0.1, 0.1, 0.2, 0.2]):
        self.prior_boxes = prior_boxes
        self.variances = variances
        super(EncodeBoxes, self).__init__()

    def call(self, boxes):
        encoded_boxes = encode(boxes, self.prior_boxes, self.variances)
        return encoded_boxes

def to_one_hot(class_indices, num_classes):
    """ Transform from class index to one-hot encoded vector.

    # Arguments
        class_indices: Numpy array. One dimensional array specifying
            the index argument of the class for each sample.
        num_classes: Integer. Total number of classes.

    # Returns
        Numpy array with shape `(num_samples, num_classes)`.
    """
    one_hot_vectors = np.zeros((len(class_indices), num_classes))
    for vector_arg, class_args in enumerate(class_indices):
        one_hot_vectors[vector_arg, class_args] = 1.0
    return one_hot_vectors

class MatchBoxes(Processor):
    """Match prior boxes with ground truth boxes.

    # Arguments
        prior_boxes: Numpy array of shape (num_boxes, 4).
        iou: Float in [0, 1]. Intersection over union in which prior boxes
            will be considered positive. A positive box is box with a class
            different than `background`.
        variance: List of two floats.
    """
    def __init__(self, prior_boxes, iou=.5):
        self.prior_boxes = prior_boxes
        self.iou = iou
        super(MatchBoxes, self).__init__()

    def call(self, boxes):
        boxes = match(boxes, self.prior_boxes, self.iou)
        return boxes

class BoxClassToOneHotVector(Processor):
    """Transform box data with class index to a one-hot encoded vector.

    # Arguments
        num_classes: Integer. Total number of classes.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(BoxClassToOneHotVector, self).__init__()

    def call(self, boxes):
        class_indices = boxes[:, 4].astype('int')
        one_hot_vectors = to_one_hot(class_indices, self.num_classes)
        one_hot_vectors = one_hot_vectors.reshape(-1, self.num_classes)
        boxes = np.hstack([boxes[:, :4], one_hot_vectors.astype('float')])
        return boxes

class PreprocessBoxes(SequentialProcessor):
    """Preprocess bounding boxes

    # Arguments
        num_classes: Int.
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, num_classes, prior_boxes, IOU, variances):
        super(PreprocessBoxes, self).__init__()
        self.add(MatchBoxes(prior_boxes, IOU),)
        self.add(EncodeBoxes(prior_boxes, variances))
        self.add(BoxClassToOneHotVector(num_classes))

class UnpackDictionary(Processor):
    """Unpacks dictionary into a tuple.
    # Arguments
        order: List of strings containing the keys of the dictionary.
            The order of the list is the order in which the tuple
            would be ordered.
    """
    def __init__(self, order):
        if not isinstance(order, list):
            raise ValueError('``order`` must be a list')
        self.order = order
        super(UnpackDictionary, self).__init__()

    def call(self, kwargs):
        args = tuple([kwargs[name] for name in self.order])
        return args

class LoadImage(Processor):
    """Loads image.

    # Arguments
        num_channels: Integer, valid integers are: 1, 3 and 4.
    """
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        super(LoadImage, self).__init__()

    def call(self, image):
        return load_image(image, self.num_channels)

class AugmentDetection(SequentialProcessor):
    """Augment boxes and images for object detection.

    # Arguments
        prior_boxes: Numpy array of shape ``[num_boxes, 4]`` containing
            prior/default bounding boxes.
        split: Flag from `paz.processors.TRAIN`, ``paz.processors.VAL``
            or ``paz.processors.TEST``. Certain transformations would take
            place depending on the flag.
        num_classes: Int.
        size: Int. Image size.
        mean: List of three elements indicating the per channel mean.
        IOU: Float. Intersection over union used to match boxes.
        variances: List of two floats indicating variances to be encoded
            for encoding bounding boxes.
    """
    def __init__(self, prior_boxes, split=TRAIN, num_classes=21, size=512,
                 mean=BGR_IMAGENET_MEAN, IOU=.5,
                 variances=[0.1, 0.1, 0.2, 0.2]):
        super(AugmentDetection, self).__init__()

        # image processors
        self.augment_image = AugmentImage()
        self.preprocess_image = PreprocessImage((size, size), mean)
        self.preprocess_image.insert(0, ConvertColorSpace(RGB2BGR))

        # box processors
        self.augment_boxes = AugmentBoxes()
        args = (num_classes, prior_boxes, IOU, variances)
        self.preprocess_boxes = PreprocessBoxes(*args)

        # pipeline
        self.add(UnpackDictionary(['image', 'boxes']))
        self.add(ControlMap(LoadImage(), [0], [0]))
        if split == TRAIN:
            self.add(ControlMap(self.augment_image, [0], [0]))
            self.add(ControlMap(self.augment_boxes, [0, 1], [0, 1]))
        self.add(ControlMap(self.preprocess_image, [0], [0]))
        self.add(ControlMap(self.preprocess_boxes, [1], [1]))
        self.add(SequenceWrapper(
            {0: {'image': [size, size, 3]}},
            {1: {'boxes': [len(prior_boxes), 4 + num_classes]}}))