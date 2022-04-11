#include <iostream>
#include <opencv2/opencv.hpp>
#include "darknet.h"
#include "utils.hpp"

#define NETWORK_CFG ""
#define NETWORK_WEIGHTS ""

int main()
{
    std::cout<<"Classification example using Darknet\n\n -> Loading network..."<<std::endl;
    network *net = load_network(NETWORK_CFG,NETWORK_WEIGHTS,0);
    std::cout<<" -> Loading image..."<<std::endl;
    cv::Mat img = cv::imread("../data/eagle.jpg");
    float *blob = MatToBuffer(img,net);
    std::cout<<" -> Predicting..."<<std::endl;
    float *pred = network_predict(*net,blob);
    delete[] blob;
    int cls = 0;
    float max = 0;
    for(int i=0;i<net->outputs;i++){
        if(pred[i]>max){max=pred[i];cls=i;}
    }
    std::cout<<"      Predicted class: "<<cls<<" probability: "<<max<<std::endl;
    free_network_ptr(net);
    free(pred);
    std::cout<<"Ready"<<std::endl;
    return 0;
}
