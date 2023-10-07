#ifndef PRAY_TRACKER_H
#define PRAY_TRACKER_H
#include <vector>

#include <iostream>
#include "kalmanfilter.h"
#include "track.h"
#include "model.h"


class PrayTracker
{
public:
    float max_iou_distance;
    KalmanFilter* kf;
    int _next_idx;
    int max_live;

public:
    std::vector<Track> tracks;
    PrayTracker();
    void predict();
    void init(float max_iou_distance = 0.9, int max_live = 60);
    void update(const DETECTIONS& detections);
    typedef DYNAMICM (PrayTracker::* GATED_METRIC_FUNC)(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    void get_tracker_result(std::vector <tracker_result_data_t> &tracker_result_data);
private:    
    void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
    void _initiate_track(const DETECTION_ROW& detection);
public:
    DYNAMICM gated_matric(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    DYNAMICM iou_cost(
            std::vector<Track>& tracks,
            const DETECTIONS& dets,
            const std::vector<int>& track_indices,
            const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DETECTBOX& bbox,
            DETECTBOXSS &candidates);
};

#endif // PRAY_TRACKER_H
