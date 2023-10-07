#include "multitracker.h"
#include "nn_matching.h"
#include "model.h"
#include "linear_assignment.h"
using namespace std;

////#define MY_inner_DEBUG
//#ifdef MY_inner_DEBUG
//#include <string>
//#include <iostream>
//
//#endif

MultiTracker::MultiTracker() {
}

/**
 * 初始化方法
 * @param max_cosine_distance 0.3
 * @param nn_budget 100
 * @param max_iou_distance 0.7
 * @param max_age 300
 * @param n_init 1
 */

void MultiTracker::init(float max_cosine_distance, int nn_budget,
                        float max_iou_distance,
                        int max_age, int n_init) {
    this->metric = new NearNeighborDisMetric(
            NearNeighborDisMetric::METRIC_TYPE::cosine,
            max_cosine_distance, nn_budget);
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;

    this->kf = new KalmanFilter();
    // 清空std::vector<Track> tracks
    this->tracks.clear();
    // id初始化
    this->_next_idx = 1;
}

// 对所有轨迹进行卡尔曼预测
void MultiTracker::predict() {
    for (Track &track: tracks) {
        track.predit(kf);
    }
}


void MultiTracker::update(const DETECTIONS &detections) {
    TRACHER_MATCHD res;
    // 轨迹匹配
    _match(detections, res);
    // 匹配成功的轨迹
    vector <MATCH_DATA> &matches = res.matches;
    for (MATCH_DATA &data: matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks[track_idx].update(this->kf, detections[detection_idx]);
    }
    // 标记需要删除的轨迹
    vector<int> &unmatched_tracks = res.unmatched_tracks;
    for (int &track_idx: unmatched_tracks) {
        this->tracks[track_idx].mark_missed();
    }
    // 未匹配的检测框生成新的轨迹
    vector<int> &unmatched_detections = res.unmatched_detections;
    for (int &detection_idx: unmatched_detections) {
        this->_initiate_track(detections[detection_idx]);
    }
    vector<Track>::iterator it;
    // 遍历所有轨迹，删除标记为deleted的轨迹
    for (it = tracks.begin(); it != tracks.end();) {
        if ((*it).is_deleted())
            it = tracks.erase(it);
        else
            ++it;
    }

    vector<int> active_targets;
    // using TRACKER_DATA = std::pair<int, FEATURESS>;
    vector <TRACKER_DATA> tid_features;
    // 处理未匹配的预测框
    for (Track &track: tracks) {
        // 如果是新来的什么也不做
        if (track.is_confirmed() == false)
            continue;
        // confirmed状态的track
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        FEATURESS t = FEATURESS(0, k_feature_dim);
        // 特征置空
        track.features = t;
    }
    // tid_features： confirmed状态的track(id, features)
    // active_targets: confirmed状态的track id
    // 轨迹中的特征向量处理
    this->metric->partial_fit(tid_features, active_targets);
}


/**
 * 进行匹配计算
 * typedef struct t{
        std::vector<MATCH_DATA> matches;        匹配成功的集合由std::pair<int, int>收集;
        std::vector<int> unmatched_tracks;      没有匹配的轨迹
        std::vector<int> unmatched_detections;  没有匹配的检测
   }TRACHER_MATCHD;
 * @param 所有检测
 * @param 匹配结果
 */
void MultiTracker::_match(const DETECTIONS &detections, TRACHER_MATCHD &res) {
    // 初始化确认与未确认集合
    vector<int> confirmed_tracks;
    vector<int> unconfirmed_tracks;
    int idx = 0;
    // 确认态与非确认态分开
    for (Track &t: tracks) {
        if (t.is_confirmed())
            confirmed_tracks.push_back(idx);
        else
            unconfirmed_tracks.push_back(idx);
        idx++;
    }
    // 确认态进行级联匹配
    TRACHER_MATCHD matcha = linear_assignment::getInstance()->matching_cascade(
            this, &MultiTracker::gated_matric,
            this->metric->mating_threshold,
            this->max_age,
            this->tracks,
            detections,
            confirmed_tracks);
    vector<int> iou_track_candidates;
    // 非确认态用IOU匹配
    iou_track_candidates.assign(unconfirmed_tracks.begin(), unconfirmed_tracks.end());
    vector<int>::iterator it;
    for (it = matcha.unmatched_tracks.begin(); it != matcha.unmatched_tracks.end();) {
        int idx = *it;
        if (tracks[idx].time_since_update == 1) { // push into unconfirmed
            iou_track_candidates.push_back(idx);
            it = matcha.unmatched_tracks.erase(it);
            continue;
        }
        ++it;
    }
    TRACHER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
            this, &MultiTracker::iou_cost,
            this->max_iou_distance,
            this->tracks,
            detections,
            iou_track_candidates,
            matcha.unmatched_detections);
    // get result:
    res.matches.assign(matcha.matches.begin(), matcha.matches.end());
    res.matches.insert(res.matches.end(), matchb.matches.begin(), matchb.matches.end());
    // unmatched_tracks;
    res.unmatched_tracks.assign(
            matcha.unmatched_tracks.begin(),
            matcha.unmatched_tracks.end());
    res.unmatched_tracks.insert(
            res.unmatched_tracks.end(),
            matchb.unmatched_tracks.begin(),
            matchb.unmatched_tracks.end());
    res.unmatched_detections.assign(
            matchb.unmatched_detections.begin(),
            matchb.unmatched_detections.end());
}

/**
 * 初始化轨迹
 * @param detection 一个检测框
 */
void MultiTracker::_initiate_track(const DETECTION_ROW &detection) {
    // 初始化卡尔曼滤波器
    KAL_DATA data = kf->initiate(detection.to_xyah());
    // 状态向量
    KAL_MEAN mean = data.first;
    // 状态协方差矩阵
    KAL_COVA covariance = data.second;
    // 添加一个轨迹，id+1
    this->tracks.push_back(Track(mean, covariance, this->_next_idx, this->n_init,
                                 this->max_age, detection.feature, detection.class_id,
                                 detection.conf));
    _next_idx += 1;
}

/**
 * 表观特征匹配
 * @param tracks
 * @param dets
 * @param track_indices
 * @param detection_indices
 * @return
 */
DYNAMICM MultiTracker::gated_matric(
        std::vector <Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices) {
    FEATURESS features(detection_indices.size(), k_feature_dim);
    int pos = 0;
    for (int i: detection_indices) {
        features.row(pos++) = dets[i].feature;
    }
    vector<int> targets;
    for (int i: track_indices) {
        targets.push_back(tracks[i].track_id);
    }
    // 计算代价矩阵
    DYNAMICM cost_matrix = this->metric->distance(features, targets);
    // 轨迹分配
    DYNAMICM res = linear_assignment::getInstance()->gate_cost_matrix(
            this->kf, cost_matrix, tracks, dets, track_indices,
            detection_indices);
    return res;
}

/**
 * 构建IOU损失矩阵
 * @param tracks
 * @param dets
 * @param track_indices
 * @param detection_indices
 * @return
 */
DYNAMICM
MultiTracker::iou_cost(
        std::vector <Track> &tracks,
        const DETECTIONS &dets,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices) {
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for (int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        if (tracks[track_idx].time_since_update > 1) {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
            continue;
        }
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
 * 计算一个检测框与所有预测框的IOU
 * @param bbox          检测框
 * @param candidates    所有预测框
 * @return              返回数组为检测框与所有预测框的IOU
 */
Eigen::VectorXf
MultiTracker::iou(DETECTBOX &bbox, DETECTBOXSS &candidates) {
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
    //    Eigen::VectorXf area_intersection(size);
    //    Eigen::VectorXf area_candidates(size);
    Eigen::VectorXf res(size);
    for (int i = 0; i < size; i++) {
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
    //#ifdef MY_inner_DEBUG
    //        std::cout << res << std::endl;
    //#endif
    return res;
}


void MultiTracker::get_tracker_result(std::vector <tracker_result_data_t> &tracker_result_data) {
    for (Track &track: tracks) {
        if (!track.is_confirmed() || track.time_since_update > 1) continue;
        tracker_result_data_t track_result;
        track_result.class_id = track.class_id;
        track_result.track_id = track.track_id;
        track_result.conf = track.conf;
        track_result.box = track.to_tlwh();
        tracker_result_data.push_back(track_result);
    }
}