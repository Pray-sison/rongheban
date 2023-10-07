//
// Created by Pray on 2023/8/10.
//

#ifndef RKNN_DETECT_FEATURE_H
#define RKNN_DETECT_FEATURE_H


#include <set>
#include <vector>
#include <rknn_api.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "dataType.h"


class Feature {
private:

    rknn_context ctx;
    int model_data_size = 0;
    int feature_length = 512;

    int ret = 0;

    int channel = 3;
    int width = 0;
    int height = 0;

    rknn_tensor_attr input_attrs[1];
    rknn_tensor_attr output_attrs[1];
    rknn_input inputs[1];
    rknn_output outputs[1];

    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);

    void dump_tensor_attr(rknn_tensor_attr *attr);

    unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);

    unsigned char *load_model(const char *filename, int *model_size);



    cv::Mat image_preporcess(cv::Mat orig_frame);

    inline int32_t __clip(float val, float min, float max) {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    std::vector<float> normalize_data(const std::vector<float>& input_data);

public:
    Feature();

    int init(const char *filename);

    Feature(const char *filename);

    FEATURE feature_detect(const cv::Mat ori_frame);
};


#endif //RKNN_DETECT_FEATURE_H
