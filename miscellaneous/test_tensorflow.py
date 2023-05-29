# Test whether tensorflow has picked up our GPU.
from tensorflow.python.client import device_lib
import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")
print("Detected devices for processing:")
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))
