#ifndef MULTITRACKER_H
#define MULTITRACKER_H

#include <vector>
#include "kalmanfilter.h"
#include "track.h"
#include "model.h"


class NearNeighborDisMetric;

class MultiTracker {
public:
    NearNeighborDisMetric *metric;
    float max_iou_distance;
    int max_age;
    int n_init;
    KalmanFilter *kf;
    int _next_idx;
public:

    std::vector<Track> tracks;

    MultiTracker();

    void init(float max_cosine_distance = 0.3, int nn_budget = 1,
              float max_iou_distance = 0.9,
              int max_age = 1, int n_init = 1);


    void predict();

    void update(const DETECTIONS &detections);

    typedef DYNAMICM (MultiTracker::* GATED_METRIC_FUNC)(
            std::vector<Track> &tracks,
            const DETECTIONS &dets,
            const std::vector<int> &track_indices,
            const std::vector<int> &detection_indices);

    void get_tracker_result(std::vector<tracker_result_data_t> &tracker_result_data);

private:
    void _match(const DETECTIONS &detections, TRACHER_MATCHD &res);

    void _initiate_track(const DETECTION_ROW &detection);

public:
    DYNAMICM gated_matric(
            std::vector<Track> &tracks,
            const DETECTIONS &dets,
            const std::vector<int> &track_indices,
            const std::vector<int> &detection_indices);

    DYNAMICM iou_cost(
            std::vector<Track> &tracks,
            const DETECTIONS &dets,
            const std::vector<int> &track_indices,
            const std::vector<int> &detection_indices);

    Eigen::VectorXf iou(DETECTBOX &bbox,
                        DETECTBOXSS &candidates);
};

#endif // MULTITRACKER_H
