Single-Shot MultiBox Detector
===

This folder contains a (botched) implementation of Single-Shot multi-box detector. Unfortunately this approach is not currently functional. To make it work one might look into data pre-processing techniques to make data more paletable for the neural network. Performance is poor and most of the time it won't converge properly.

## VGG

This implementation is based on the VGG16 network. For more information see the [VGG](./VGG16/) folder!

## SSD

This implementation is based on [PAZ](https://github.com/oarriaga/paz), but it's slightly adapted to work. For more information see the [SSD](./SSD/) folder!

#### NOTE: When generating classes, please remember that SSD requires class 0 to be background!
#### NOTE2: Your data needs to have bounding boxes. The model will not process pictures that don't have any! In fact, it will throw an error in that situation.