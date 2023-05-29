import numpy as np
import logging
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, InputLayer
from keras.callbacks import ModelCheckpoint, CSVLogger
from vgg_dataset import vgg_dataset
from tensorflow import one_hot
from keras.utils import load_img, img_to_array
import os

class Model:
    def __init__(self, data_folder = "dataset", load_dataset = True):
        self.model = None
        if load_dataset:
            self.dataset = vgg_dataset(data_folder)
            self.dataset.load()
        
    # Using VGG16 here, though ResNet50 and others can be used as well
    # ConvNet Configuration C (16-weight-layers)
    def build_regional_proposal_network(self, shape = (512,512,3), train_vgg = True, train_classes = 9):
        self.classes = train_classes

        model = Sequential() # OR BRANCH
        model.add(InputLayer(shape))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2))) # For use in transfer learning this should be 3,3 with stride of 1,1. Remove altogether?

        # VGG alone can do prediction, but additional work is required found bounding box regression. We don't include these last steps in that case.
        if train_vgg:
            model.add(Flatten())
            model.add(Dense(4096, activation='relu'))
            model.add(Dense(4096, activation='relu'))
            model.add(Dense(train_classes, activation='softmax', name='classification'))

        self.model = model

    def compile_model(self, output_summary = True):        
        if self.model is None:
            raise Exception("Model not defined!")
        # TODO: Try SGD optimizer? Adam?
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        if output_summary:
            self.model.summary()

    def load_model(self, filename=None):
        if self.model is not None:
            if filename is not None and os.path.isfile(filename):
                logging.info(f"Loading weights from {filename}!")
                self.model.load_weights(filename)
        else:
            logging.critical("Cannot load weights. Either the model is not set up or the file does not exist!")
            raise Exception("Could not find weights")
        
    def get_model(self):
        return self.model

    def save_model(self, filename="model.save"):
        if self.model is not None:
            self.model.save_weights(filename, overwrite=True)
        else:
            logging.critical("Cannot save weights. Is the model not set up?")
            raise Exception("Failed to load weights")
        
    def predict_using_regional_proposal_network(self, image):
        img = np.array([img_to_array(load_img(image))])
        prediction = self.model.predict(img)
        return prediction

    def train_regional_proposal_network(self, epochs = 500, save_log=True, log_name="history.csv", final_log_name="history2.csv"):
        if self.model is not None:
            logging.info("Training the model...")

            weights_backup = ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=5)
            logger = CSVLogger(log_name, separator=",", append=True)

            (train_dataset, validation_dataset) = self.dataset.get_dataset()

            # Change label size to make it compatible with the last dense layer using categorical_crossentropy
            one_hot_encoded_train_ds = train_dataset.map(lambda x, y: (x, one_hot(y, depth=self.classes)))
            one_hot_encoded_val_ds = validation_dataset.map(lambda x, y: (x, one_hot(y, depth=self.classes)))

            training = self.model.fit(one_hot_encoded_train_ds, epochs=epochs, validation_data=one_hot_encoded_val_ds, callbacks=[weights_backup, logger])
            
            logging.info("Training finished.")
            if save_log is True:
                logging.info("Exporting statistics.")
                import csv
                with open(final_log_name, 'w') as f:
                    w = csv.DictWriter(f, training.history.keys())
                    w.writeheader()
                    w.writerow(training.history)
            self.save_model()
        else:
            logging.error("Model not prepared, cannot train it! Please run build_regional_proposal_network() first!")
