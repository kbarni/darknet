#include <iostream>
#include <opencv2/opencv.hpp>
#include "darknet.h"
#include "utils.hpp"

#define NETWORK_CFG "/home/barna/source-installers/darknet/cfg/unet_segment.cfg"
#define NETWORK_WEIGHTS "/home/barna/source-installers/darknet_modded_yolo/test/instance_segment_161000.weights"
#define IMAGE "/home/barna/source-installers/darknet_modded_yolo/test/pic2.png"

int main()
{
    std::cout<<"Classification example using Darknet\n\n -> Loading network..."<<std::endl;
    network *net = load_network(NETWORK_CFG,NETWORK_WEIGHTS,0);
    std::cout<<" -> Loading image..."<<std::endl;
    cv::Mat img=cv::imread(IMAGE);
    float *blob = MatToBuffer(img,net);
    std::cout<<" -> Predicting..."<<std::endl;
    float *pred = network_predict(*net,blob);
    std::cout<<"Ready"<<std::endl;
    cv::Mat result=resultToMat(pred,net);
    cv::namedWindow("Result",cv::WINDOW_NORMAL);
    cv::imshow("Result",result);
    cv::waitKey();
    free_network_ptr(net);
    free(pred);
    delete[] blob;
    return 0;
}
