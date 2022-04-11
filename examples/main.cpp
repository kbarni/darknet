#include <iostream>
#include <opencv2/opencv.hpp>
#include "darknet.h"

using namespace cv;
using namespace std;

int main()
{
    Mat img=imread("");
    network *net = load_network("netfile.cfg","/home/barna/source-installers/darknet_weights/darknet.weights",0);
    return 0;
}
