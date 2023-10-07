#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class SingleTracker {
public:
    SingleTracker() {}

    virtual  ~SingleTracker() {}

    virtual void init_roi(const cv::Rect &roi, cv::Mat image) = 0;

    virtual cv::Rect update_roi(cv::Mat image) = 0;

    virtual cv::Rect updateMemory_roi(cv::Mat image) = 0;


protected:
    cv::Rect_<float> _roi;
};



