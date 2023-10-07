#pragma once
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

// 检测框格式
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;

// 检测框集合格式
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;

// 状态向量格式
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;

// 状态转移矩阵格式
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;

// 观测向量格式
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;

// 观测矩阵格式
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;

// 先验估计格式
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;

// 后验估计格式
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

// 索引匹配存储
using MATCH_DATA = std::pair<int, int>;

// 匹配结果格式
typedef struct {
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

// iou损失矩阵格式
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;


// 跟踪返回结果
typedef struct TRACKER_RESULT_DATA_T{
    int class_id;
    int track_id;
    float conf;
    DETECTBOX box;
}tracker_result_data_t;











