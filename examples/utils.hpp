#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include "darknet.h"

cv::Vec3b colorize(int n)
{
    uchar collist[]={255,255,255,127,0,0,0,0,0,127,255,255};
    uchar r,g,b;
    r=collist[n%12]; //12 colors
    g=collist[(n-4)%12];
    b=collist[(n-8)%12];
    return cv::Vec3b(b,g,r);
}

float *MatToBuffer(cv::Mat img, network *net)
{
    int blobsize=net->w*net->h*3;
    float *buffer=new float[blobsize];
    cv::Mat imr;
    cv::resize(img,imr,cv::Size(net->w,net->h));
    int plan_size=imr.cols*imr.rows;
    int plan_size_2=2*plan_size;
    uchar *pl;
    int il;
    for(int y=0;y<imr.rows;y++){
        pl = imr.ptr(y);
        il = y * net->w;
        for(int x=0;x<imr.cols;x++){
            buffer[il+x]=pl[3*x+2]/255.0f;
            buffer[plan_size+il+x]=pl[3*x+1]/255.0f;
            buffer[plan_size_2+il+x]=pl[3*x]/255.0f;
        }
    }
    return buffer;
}

cv::Mat resultToMat(float *pred, network *net)
{
    cv::Mat result(net->h,net->w,CV_8UC3);
    int net_size=net->h*net->w;
    int ch=net->outputs/net_size;
    cv::Vec3b *ptr=(cv::Vec3b*)result.ptr();
    for(int p=0;p<net_size;p++){
        float mp=0;
        int cl=0;
        for(int k=0;k<ch;k++){
            if(pred[k*net_size+p]>mp){mp=pred[k*net_size+p];cl=k;}
        }
        ptr[p]=colorize(cl);
    }
    return result;
}

#endif // UTILS_HPP
