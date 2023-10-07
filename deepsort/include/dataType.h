#pragma once
#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>


const int k_feature_dim=512;//feature dim
// 检测框
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;

// 检测框集合
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;

// 特征向量
typedef Eigen::Matrix<float, 1, k_feature_dim, Eigen::RowMajor> FEATURE;

// 特征向量集合
typedef Eigen::Matrix<float, Eigen::Dynamic, k_feature_dim, Eigen::RowMajor> FEATURESS;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;


// 状态向量
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;

// 状态转移矩阵
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;

// 观测向量
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;

// 观测矩阵
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;

// 先验估计
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;

// 后验估计
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//main
// 检测结果存储
using RESULT_DATA = std::pair<int, DETECTBOX>;

//tracker:
// 轨迹编号存储
using TRACKER_DATA = std::pair<int, FEATURESS>;

// 索引匹配存储
using MATCH_DATA = std::pair<int, int>;

// 匹配结果
typedef struct t{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

// 跟踪返回结果
typedef struct TRACKER_RESULT_DATA_T{
    int class_id;
    int track_id;
    float conf;
    DETECTBOX box;
}tracker_result_data_t;

