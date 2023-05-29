# Model adapted from: https://github.com/oarriaga/paz/blob/master/paz/models/detection/ssd512.py (MIT license)
# Removed references to test data, so that the model can actually be used.
import logging
from keras.layers import Conv2D, Input, MaxPooling2D, ZeroPadding2D, BatchNormalization, Flatten, Concatenate, Reshape, Activation, InputLayer
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Layer
import keras.backend as K
from itertools import product
from keras.initializers import Constant
import numpy as np

# Source https://github.com/oarriaga/paz/blob/c6a884326c73775a27b792ce91b11c0b3d98bf08/paz/models/detection/utils.py
def create_multibox_head(tensors, num_classes, num_priors, l2_loss=0.0005,
                         num_regressions=4, l2_norm=False, batch_norm=False):
    classification_layers, regression_layers = [], []
    for layer_arg, base_layer in enumerate(tensors):
        if l2_norm:
            base_layer = Conv2DNormalization(20)(base_layer)

        # classification leaf -------------------------------------------------
        num_kernels = num_priors[layer_arg] * num_classes
        class_leaf = Conv2D(num_kernels, 3, padding='same',
                            kernel_regularizer=l2(l2_loss))(base_layer)
        if batch_norm:
            class_leaf = BatchNormalization()(class_leaf)
        class_leaf = Flatten()(class_leaf)
        classification_layers.append(class_leaf)

        # regression leaf -----------------------------------------------------
        num_kernels = num_priors[layer_arg] * num_regressions
        regress_leaf = Conv2D(num_kernels, 3, padding='same',
                              kernel_regularizer=l2(l2_loss))(base_layer)
        if batch_norm:
            regress_leaf = BatchNormalization()(regress_leaf)

        regress_leaf = Flatten()(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = Concatenate(axis=1)(classification_layers)
    regressions = Concatenate(axis=1)(regression_layers)
    num_boxes = K.int_shape(regressions)[-1] // num_regressions
    classifications = Reshape((num_boxes, num_classes))(classifications)
    classifications = Activation('softmax')(classifications)
    regressions = Reshape((num_boxes, num_regressions))(regressions)
    outputs = Concatenate(
        axis=2, name='boxes')([regressions, classifications])
    return outputs

def get_prior_box_configuration():
        return {
            'feature_map_sizes': [64, 32, 16, 8, 4, 2, 1],
            'image_size': 512,
            'steps': [8, 16, 32, 64, 128, 256, 512],
            'min_sizes': [21, 51, 133, 215, 297, 379, 461],
            'max_sizes': [51, 133, 215, 297, 379, 461, 542],
            'aspect_ratios': [[2], [2, 3], [2, 3],
                              [2, 3], [2, 3], [2], [2]],
            'variance': [0.1, 0.2]}

def create_prior_boxes():
    configuration = get_prior_box_configuration()
    image_size = configuration['image_size']
    feature_map_sizes = configuration['feature_map_sizes']
    min_sizes = configuration['min_sizes']
    max_sizes = configuration['max_sizes']
    steps = configuration['steps']
    model_aspect_ratios = configuration['aspect_ratios']
    mean = []
    for feature_map_arg, feature_map_size in enumerate(feature_map_sizes):
        step = steps[feature_map_arg]
        min_size = min_sizes[feature_map_arg]
        max_size = max_sizes[feature_map_arg]
        aspect_ratios = model_aspect_ratios[feature_map_arg]
        for y, x in product(range(feature_map_size), repeat=2):
            f_k = image_size / step
            center_x = (x + 0.5) / f_k
            center_y = (y + 0.5) / f_k
            s_k = min_size / image_size
            mean = mean + [center_x, center_y, s_k, s_k]
            s_k_prime = np.sqrt(s_k * (max_size / image_size))
            mean = mean + [center_x, center_y, s_k_prime, s_k_prime]
            for aspect_ratio in aspect_ratios:
                mean = mean + [center_x, center_y, s_k * np.sqrt(aspect_ratio),
                               s_k / np.sqrt(aspect_ratio)]
                mean = mean + [center_x, center_y, s_k / np.sqrt(aspect_ratio),
                               s_k * np.sqrt(aspect_ratio)]

    output = np.asarray(mean).reshape((-1, 4))
    return output

# Source https://github.com/oarriaga/paz/blob/c6a884326c73775a27b792ce91b11c0b3d98bf08/paz/models/layers.py
class Conv2DNormalization(Layer):
    def __init__(self, scale, axis=3, **kwargs):
        self.scale = scale
        self.axis = axis
        super(Conv2DNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma', shape=(input_shape[self.axis]),
            initializer=Constant(self.scale), trainable=True)

    def output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, self.axis)

def SSD512(num_classes=14, input_shape=(512, 512, 3), num_priors=[4, 6, 6, 6, 6, 4, 4],
           l2_loss=0.0005, return_base=False, trainable_base=True, ssd_weights=None, output_summary=False):
    image = Input(shape=input_shape, name='image')
    #image = InputLayer(input_shape=input_shape, name="image")

    # VGG16 blocks
    # Block 1 -----------------------------------------------------------------
    conv1_1 = Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv1_1')(image)
    conv1_2 = Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same', )(conv1_2)

    # Block 2 -----------------------------------------------------------------
    conv2_1 = Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv2_2)

    # Block 3 -----------------------------------------------------------------
    conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv3_3)

    # Block 4 -----------------------------------------------------------------
    conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv4_3')(conv4_2)
    conv4_3_norm = Conv2DNormalization(20, name='branch_1')(conv4_3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                         padding='same')(conv4_3)

    # Block 5 -----------------------------------------------------------------
    conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     trainable=trainable_base,
                     name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                         padding='same')(conv5_3)
    
    # SSD starts here
    # Dense 6/7 --------------------------------------------------------------
    pool5z = ZeroPadding2D(padding=(6, 6))(pool5)
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6),
                 padding='valid', activation='relu',
                 kernel_regularizer=l2(l2_loss),
                 trainable=trainable_base,
                 name='fc6')(pool5z)
    fc7 = Conv2D(1024, (1, 1), padding='same',
                 activation='relu',
                 kernel_regularizer=l2(l2_loss),
                 trainable=trainable_base,
                 name='branch_2')(fc6)

    # Block 6 -----------------------------------------------------------------
    conv6_1 = Conv2D(256, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv6_1')(fc7)
    conv6_1z = ZeroPadding2D()(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                     activation='relu', name='branch_3',
                     kernel_regularizer=l2(l2_loss))(conv6_1z)

    # Block 7 -----------------------------------------------------------------
    conv7_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv7_1')(conv6_2)
    conv7_1z = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_4',
                     kernel_regularizer=l2(l2_loss))(conv7_1z)

    # Block 8 -----------------------------------------------------------------
    conv8_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv8_1')(conv7_2)
    conv8_1z = ZeroPadding2D()(conv8_1)
    conv8_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_5',
                     kernel_regularizer=l2(l2_loss))(conv8_1z)

    # Block 9 -----------------------------------------------------------------
    conv9_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss),
                     name='conv9_1')(conv8_2)
    conv9_1z = ZeroPadding2D()(conv9_1)
    conv9_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_6',
                     kernel_regularizer=l2(l2_loss))(conv9_1z)

    # Block 10 ----------------------------------------------------------------
    conv10_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                      kernel_regularizer=l2(l2_loss),
                      name='conv10_1')(conv9_2)
    conv10_1z = ZeroPadding2D()(conv10_1)
    conv10_2 = Conv2D(256, (4, 4), padding='valid', strides=(1, 1),
                      activation='relu', name='branch_7',
                      kernel_regularizer=l2(l2_loss))(conv10_1z)

    branch_tensors = [conv4_3_norm, fc7, conv6_2, conv7_2,
                      conv8_2, conv9_2, conv10_2]
    if return_base:
        output_tensor = branch_tensors
    else:
        output_tensor = create_multibox_head(branch_tensors, num_classes, num_priors, l2_loss)

    model = Model(inputs=image, outputs=output_tensor, name='SSD512')
    model.prior_boxes = create_prior_boxes()
    
    # Load either existing weights or VGG16 weights
    if ssd_weights is not None:
        logging.info(f"Loading weights from {ssd_weights}!")
        model.load_weights(ssd_weights)

    if output_summary:
        model.summary()
    return model