//
// Created by Pray on 2023/8/10.
//

#ifndef RKNN_DETECT_DETECT_DATA_H
#define RKNN_DETECT_DETECT_DATA_H
#include "yolo.h"

#endif //RKNN_DETECT_DETECT_DATA_H


#define OBJ_NUMB_MAX_SIZE 80
//#define NMS_THRESH        0.45
//#define BOX_THRESH        0.60


// 框定义
typedef struct {
    int left;
    int right;
    int top;
    int bottom;
} box_rect;

typedef struct DETECT_RESULT_t {
    // 目标类型
    int class_id;
    // 目标位置
    box_rect box;
    // 置信度
    float conf;
} detect_result_t;

typedef struct DETECT_RESULT_GROUP_t {
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];

} detect_result_group_t;
