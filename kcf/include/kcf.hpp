#pragma once

#include <opencv2/opencv.hpp>

typedef struct {
    cv::Rect box;
    int state;
} KcfResult;

KcfResult kcfTrack(cv::Mat &image, const cv::Rect &box);
