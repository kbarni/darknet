# Image classification, object detection and image classification using neural networks

This **Darknet** fork is based mainly on [AlexeyAB](https://github.com/AlexeyAB/darknet)'s Darknet fork, which is the best maintained and containing Yolo V2-3-4 detection networks, after [pjreddie](https://github.com/pjreddie/darknet)'s original Darknet implementation (mainly for classification) and completed with [ArtyZe](https://github.com/ArtyZe/yolo_segmentation)'s code for image segmentation.

For more information about the useage, please refer to the readme files of the respective forks.

## Main features

- Image classification using ~20 state of the art networks (https://pjreddie.com/darknet/imagenet/)
- Object detection using Yolo networks (https://pjreddie.com/darknet/yolo/)
- Image segmentation using unet networks (https://github.com/ArtyZe/yolo_segmentation/blob/master/README.md)
- Text generation using rnns, GAN nightmare, etc...

## Building

### Recommended prerequisites

- CMake 3.18+
- OpenCV 2.4+
- CUDA 10.2+
- cuDNN 8.0.2+

### Building

#### Linux/MacOS

Using CMake:

```
git clone https://github.com/kbarni/darknet
cd darknet
mkdir build_release
cd build_release
cmake ..
cmake --build . --target install --parallel 8
```

Using make: edit the options at the beginning of the provided `Makefile`, then run `make`.

#### Windows

- Clone this repository
- Run CMake (gui)
- Select the source and build folders, run `Configure` and `Generate`
- Open the created file with `Visual studio`, then `Build solution`.

# Useage

## Classification

- Create a file containing the class names (labels.txt), the training images (training.txt) and validation images (validation.txt). The image names must contain the class name (as defined in `lables.txt`).
- Create a `data` file containing:

```
classes = [n]
train = train.txt
valid = valid.txt
labels = labels.txt
backup = backup/
```

- Create a config file or choose one from the `cfg` folder
- Train the model:

    ./darknet classifier train mydata.data network.cfg
    
- Test the trained network on an image:

    ./darknet classifier test mydata.data network.cfg network.weights testimage.jpg
    

For a more detailed description check the [CIFAR-10 classifier training tutorial](https://pjreddie.com/darknet/train-cifar/).

## Detection

- Generate label file for your files in format (x,y,w,h are relative to the image size):

    <object-class> <x> <y> <width> <height>
    
- Create a `data` file as above, using a yolo network
- Train the model:

    ./darknet detector train mydata.data network.cfg
    
- Test the trained network on an image:

    ./darknet detector test mydata.data network.cfg network.weights testimage.jpg

For more details, refer to the [Yolo tutorial](https://pjreddie.com/darknet/yolo/) and [AlexeyAB's wiki](https://github.com/AlexeyAB/darknet/wiki/Train-and-Evaluate-Detector-on-Pascal-VOC-(VOCtrainval-2007-2012)-dataset).

## Segmentation

- Create a file containing the class names (names.list), the training images (training.txt) and image labels (labels.txt). The label images must have the same name than the training images; they have to be 8 bit PNG images with every gray level corresponding to a class.
- Create a `data` file containing:

```
classes = [n]
train = train.txt
labels = labels.txt
backup = backup/
```

- Choose a unet network from the `cfg` folder
- Train the model:

    ./darknet segmenter train mydata.data network.cfg
    
- Test the trained network on an image:

    ./darknet segmenter test mydata.data network.cfg network.weights testimage.jpg

## C++

For using Darknet in C++ with OpenCV, please refer to the sample files in the `examples` folder.

# Further reading:

- Joseph Redmon's (pjreddie) darknet homepage: https://pjreddie.com/darknet/
- Alexey Bochkovskiy's (AlexeyAB) darknet wiki: https://github.com/AlexeyAB/darknet/wiki
- AlexeyAB's darknet readme: https://github.com/AlexeyAB/darknet/blob/master/README.md
- ArtyZe's readme: https://github.com/ArtyZe/yolo_segmentation/blob/master/README.md