import cv2

def convert_color_space(image, flag):
    """Convert image to a different color space.

    # Arguments
        image: Numpy array.
        flag: PAZ or openCV flag. e.g. paz.backend.image.RGB2BGR.

    # Returns
        Numpy array.
    """
    return cv2.cvtColor(image, flag)

B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN = 104, 117, 123
BGR_IMAGENET_MEAN = (B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN)
RGB_IMAGENET_MEAN = (R_IMAGENET_MEAN, G_IMAGENET_MEAN, B_IMAGENET_MEAN)
B_IMAGENET_STDEV, G_IMAGENET_STDEV, R_IMAGENET_STDEV = 57.3, 57.1, 58.4
RGB_IMAGENET_STDEV = (R_IMAGENET_STDEV, G_IMAGENET_STDEV, B_IMAGENET_STDEV)

'''
RGB2BGR = 4
BGR2RGB = 4
RGB2GRAY = 7
RGB2HSV = 41
HSV2RGB = 55
'''

RGB2BGR = cv2.COLOR_RGB2BGR
BGR2RGB = cv2.COLOR_BGR2RGB
BGRA2RGBA = cv2.COLOR_BGRA2RGBA
RGB2GRAY = cv2.COLOR_RGB2GRAY
RGB2HSV = cv2.COLOR_RGB2HSV
HSV2RGB = cv2.COLOR_HSV2RGB
_CHANNELS_TO_FLAG = {1: cv2.IMREAD_GRAYSCALE,
                     3: cv2.IMREAD_COLOR,
                     4: cv2.IMREAD_UNCHANGED}
CUBIC = cv2.INTER_CUBIC
BILINEAR = cv2.INTER_LINEAR

'''
def load_image(filepath, num_channels=3):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, num_channels, expand_animations=False)
    return image
'''
_CHANNELS_TO_FLAG = {1: cv2.IMREAD_GRAYSCALE,
                     3: cv2.IMREAD_COLOR,
                     4: cv2.IMREAD_UNCHANGED}
def load_image(filepath, num_channels=3):
    """Load image from a ''filepath''.

    # Arguments
        filepath: String indicating full path to the image.
        num_channels: Int.

    # Returns
        Numpy array.
    """
    if num_channels not in [1, 3, 4]:
        raise ValueError('Invalid number of channels')

    image = cv2.imread(filepath, _CHANNELS_TO_FLAG[num_channels])
    if num_channels == 3:
        image = convert_color_space(image, BGR2RGB)
    elif num_channels == 4:
        image = convert_color_space(image, BGRA2RGBA)
    return image