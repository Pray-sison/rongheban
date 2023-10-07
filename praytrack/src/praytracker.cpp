#include "praytracker.h"
#include "model.h"
#include "linear_assignment.h"
using namespace std;

//#define MY_inner_DEBUG
#ifdef MY_inner_DEBUG
#include <string>
#include <iostream>
#endif

/**
 * 构造跟踪器对象
 * @param iou距离 0.7
 * @param 丢失轨迹最大存活帧数 30
 */

PrayTracker::PrayTracker()
{

}
void PrayTracker::init(float max_iou_distance, int max_live)
{
    this->max_iou_distance = max_iou_distance;
    this->kf = new KalmanFilter();
    // 清空std::vector<Track> tracks
    this->tracks.clear();
    // id初始化
    this->_next_idx = 1;
    // 丢失轨迹最大保存数
    this->max_live = max_live;
}

/**
 * 所有轨迹使用卡尔曼滤波预测
 */
void PrayTracker::predict()
{
    for (Track &track : tracks)
    {
        track.predit(kf);
    }
}

/**
 * 更新轨迹，新建轨迹，删除不满足条件的轨迹
 * @param 检测集合
 */
void PrayTracker::update(const DETECTIONS &detections)
{
    TRACHER_MATCHD res;
    // 轨迹匹配 res是匹配后的结果
    _match(detections, res);

    vector<MATCH_DATA> &matches = res.matches;

    for (MATCH_DATA &data : matches)
    {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx]);
    }
    // 标记未匹配成功轨迹且丢失次数超过30的轨迹为删除状态
    vector<int> &unmatched_tracks = res.unmatched_tracks;
    for (int &track_idx : unmatched_tracks)
    {
        this->tracks[track_idx].mark_missed();
    }
    // 未匹配的检测框生成新的轨迹
    vector<int> &unmatched_detections = res.unmatched_detections;
    for (int &detection_idx : unmatched_detections)
    {
        this->_initiate_track(detections[detection_idx]);
    }
    vector<Track>::iterator it;
    // 遍历所有轨迹，删除标记为deleted的轨迹
    for (it = tracks.begin(); it != tracks.end();)
    {
        if ((*it).is_delete())
            it = tracks.erase(it);
        else
            ++it;
    }
}


/**
 * 检测集合与轨迹集合匹配
 * @param 检测集合
 * @param 三种匹配结果
 */
void PrayTracker::_match(const DETECTIONS &detections, TRACHER_MATCHD &res)
{
    vector<int> need_match_tracks;
    vector<int> need_match_detections;
    int idx = 0;
    for (Track &t : tracks)
    {
        need_match_tracks.push_back(idx++);
    }
    for(size_t i = 0; i < detections.size(); i++) {
        need_match_detections.push_back(int(i));
    }
    vector<int> iou_track_candidates;
    iou_track_candidates.assign(need_match_tracks.begin(), need_match_tracks.end());
    TRACHER_MATCHD match = linear_assignment::getInstance()->min_cost_matching(
        this, &PrayTracker::iou_cost,
        this->max_iou_distance,
        this->tracks,
        detections,
        iou_track_candidates,
        need_match_detections);
    res.matches.assign(
            match.matches.begin(),
            match.matches.end());
    res.unmatched_tracks.assign(
        match.unmatched_tracks.begin(),
        match.unmatched_tracks.end());
    res.unmatched_detections.assign(
            match.unmatched_detections.begin(),
            match.unmatched_detections.end());
}

/**
 * 使用检测框新建轨迹
 * @param detection
 */
void PrayTracker::_initiate_track(const DETECTION_ROW &detection)
{
    // 初始化卡尔曼滤波器
    KAL_DATA data = kf->initiate(detection.to_xyah());
    // 状态向量
    KAL_MEAN mean = data.first;
    // 状态协方差矩阵
    KAL_COVA covariance = data.second;
    // 添加一个轨迹，id+1
    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->max_live, detection.class_id, detection.conf));
    _next_idx += 1;
}


/**
 *
 * @param 轨迹集合
 * @param 检测集合
 * @param track_indices
 * @param detection_indices
 * @return iou损失矩阵
 */
DYNAMICM
PrayTracker::iou_cost(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices)
{
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);

    for (int i = 0; i < rows; i++)
    {
        int track_idx = track_indices[i];
        DETECTBOX bbox = tracks[track_idx].to_tlwh();
        int csize = detection_indices.size();
        DETECTBOXSS candidates(csize, 4);
        for (int k = 0; k < csize; k++)
            candidates.row(k) = dets[detection_indices[k]].tlwh;
        
        Eigen::RowVectorXf rowV = (1. - iou(bbox, candidates).array()).matrix().transpose();
        cost_matrix.row(i) = rowV;
    }
    return cost_matrix;
}

/**
 * 计算一个检测与所有轨迹的iou，生成一行数据
 * @param 目标检测框
 * @param
 * @return
 */
Eigen::VectorXf
PrayTracker::iou(DETECTBOX &bbox, DETECTBOXSS &candidates)
{
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];

    Eigen::Matrix<float, -1, 2> candidates_tl;
    Eigen::Matrix<float, -1, 2> candidates_br;
    candidates_tl = candidates.leftCols(2);
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    Eigen::VectorXf res(size);
    for (int i = 0; i < size; i++)
    {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1;
        w = (w < 0 ? 0 : w);
        float h = br_2 - tl_2;
        h = (h < 0 ? 0 : h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection / (area_bbox + area_candidates - area_intersection);
    }
    
    return res;
}

void PrayTracker::get_tracker_result(std::vector <tracker_result_data_t> &tracker_result_data) {
    for (Track &track: tracks) {
        if (!track.is_live()) continue;
        tracker_result_data_t track_result;
        track_result.class_id = track.class_id;
        track_result.track_id = track.track_id;
        track_result.conf = track.conf;
        track_result.box = track.to_tlwh();
        tracker_result_data.push_back(track_result);
    }
}
