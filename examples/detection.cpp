#define OPENCV

#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo_v2_class.hpp"
#include "utils.hpp"

#define NETWORK_CFG ""
#define NETWORK_WEIGHTS ""

int main()
{
    cv::Mat img=cv::imread("../data/dog.jpg");
    std::vector<bbox_t> markers;
    Detector det(NETWORK_CFG,NETWORK_WEIGHTS);
    markers = det.detect(img);
    for(size_t i=0;i<markers.size();i++){
        if(markers[i].prob>0.3){
            cv::Point2i p1,p2;
            int prob=(int)(markers[i].prob*100);
            cv::String s="Class"+std::to_string(markers[i].obj_id)+" "+std::to_string(prob)+"%";
            p1 = cv::Point2i(markers[i].x, markers[i].y);
            p2 = cv::Point2i(markers[i].x + markers[i].w, markers[i].y + markers[i].h);
            rectangle(img,p1, p2, colorize(i), 2);
            cv::putText(img,s,p1,cv::FONT_HERSHEY_COMPLEX_SMALL,1,colorize(i));
        }
    }
    cv::imshow("Image",img);
    cv::waitKey();
    return 0;
}
