VGG16 Implementation
===

This is a basic VGG16 implementation for use with the the SSD dataset for transferance learning

Image size is assumed to be 512x512, jpeg compression.

## Training

Folder structure should match Keras' default inferred, namely:

```
root
  -- class
     -- 1.jpg
     -- 2.jpg
  -- class2
     -- 1.jpg
     -- 2.jpg
```

You can also generate class file using [generate-class-map.py](./generate-class-map.py)!

## Results

Took 2 hours to train 500 epochs, batch size 13 on 500 images on a RTX 3090 to get 60% accuracy