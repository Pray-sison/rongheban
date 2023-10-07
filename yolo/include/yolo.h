//
// Created by Pray on 2023/8/9.
//


#ifndef YOLO_DEEPSORT_RKNN_YOLOV5_H
#define YOLO_DEEPSORT_RKNN_YOLOV5_H

#include <set>
#include <vector>
#include <rknn_api.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "detectDataTyper.h"
//#include "feature.h"
#include "model.h"


//#define OBJ_NUMB_MAX_SIZE 64
//#define NMS_THRESH        0.45
//#define BOX_THRESH        0.60

class YOLOv5 {
private:
    int max_num = 80;
    float conf;
    float nms;
    int obj_class_num;
    int prop_box_size;

    rknn_context ctx;

    int model_data_size = 0;

    int ret = 0;

    int channel = 3;
    int width = 0;
    int height = 0;

    const int anchor0[6] = {10, 13, 16, 30, 33, 23};
    const int anchor1[6] = {30, 61, 62, 45, 59, 119};
    const int anchor2[6] = {116, 90, 156, 198, 373, 326};

    rknn_tensor_attr input_attrs[1];
    rknn_tensor_attr output_attrs[3];
    rknn_input inputs[1];
    rknn_output outputs[3];


    inline int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

    float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                           float ymax1);

    int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);

    inline float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

    inline float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

    inline int32_t __clip(float val, float min, float max) {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

    float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);

    void dump_tensor_attr(rknn_tensor_attr *attr);

    unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);

    unsigned char *load_model(const char *filename, int *model_size);

    cv::Mat image_preporcess(cv::Mat orig_frame);

    int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                int32_t zp, float scale);


    int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                     float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                     std::vector<float> &qnt_scales, detect_result_group_t *group);

    int nms_run(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
                int filterId);


public:
    YOLOv5();

    int init(const char *filename, float conf, float nms);

    detect_result_group_t yolo_detect(const cv::Mat ori_frame);

//    void format_conversion(const cv::Mat &ori_image, detect_result_group_t detect_result_group, DETECTIONS &detections,
//                           Feature featureModel);
    void format_conversion(detect_result_group_t detect_result_group, DETECTIONS &detections);
};


#endif //YOLO_DEEPSORT_RKNN_YOLOV5_H
