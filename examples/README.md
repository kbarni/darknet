# C++ useage examples for Darknet

This folder contains three simple C++ examples for image classification, detection and segmentation.

These examples require OpenCV to run. But you can just use them for your own code, converting your image format to input buffers.

Before compiling the examples, make sure to define the `NETWORK_CFG` and `NETWORK_WEIGHTS` as well as the image to use.

What's missing from these examples is the step of reading of the class names (normally from a `*.shortnames.list` file) and displaying the resulting class name.
