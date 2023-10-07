#include "linear_assignment.h"
#include "hungarianoper.h"
#include <map>
#include <iostream>

linear_assignment *linear_assignment::instance = NULL;
linear_assignment::linear_assignment()
{
}

linear_assignment *linear_assignment::getInstance()
{
    if(instance == NULL) instance = new linear_assignment();
    return instance;
}

/**
 * iou
 * @param distance_metric
 * @param distance_metric_func
 * @param max_distance
 * @param tracks
 * @param detections
 * @param track_indices
 * @param detection_indices
 */
TRACHER_MATCHD
linear_assignment::min_cost_matching(PrayTracker *distance_metric,
                                     PrayTracker::GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int> &track_indices,
        std::vector<int> &detection_indices)
{
    TRACHER_MATCHD res;

    if((detection_indices.size() == 0) || (track_indices.size() == 0)) {
        res.matches.clear();
        res.unmatched_tracks.assign(track_indices.begin(), track_indices.end());
        res.unmatched_detections.assign(detection_indices.begin(), detection_indices.end());
        return res;
    }
    DYNAMICM cost_matrix = (distance_metric->*(distance_metric_func))(
                tracks, detections, track_indices, detection_indices);
    for(int i = 0; i < cost_matrix.rows(); i++) {
        for(int j = 0; j < cost_matrix.cols(); j++) {
            float tmp = cost_matrix(i,j);
            if(tmp > max_distance) {
                cost_matrix(i,j) = max_distance + 1e-5;
            }
        }
    }
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = HungarianOper::Solve(cost_matrix);
    res.matches.clear();
    res.unmatched_tracks.clear();
    res.unmatched_detections.clear();
    for(size_t col = 0; col < detection_indices.size(); col++) {
        bool flag = false;
        for(int i = 0; i < indices.rows(); i++)
            if(indices(i, 1) == col) { flag = true; break;}
        if(flag == false)res.unmatched_detections.push_back(detection_indices[col]);
    }
    for(size_t row = 0; row < track_indices.size(); row++) {
        bool flag = false;
        for(int i = 0; i < indices.rows(); i++)
            if(indices(i, 0) == row) { flag = true; break; }
        if(flag == false) res.unmatched_tracks.push_back(track_indices[row]);
    }
    for(int i = 0; i < indices.rows(); i++) {
        int row = indices(i, 0);
        int col = indices(i, 1);

        int track_idx = track_indices[row];
        int detection_idx = detection_indices[col];
        if(cost_matrix(row, col) > max_distance) {
            res.unmatched_tracks.push_back(track_idx);
            res.unmatched_detections.push_back(detection_idx);
        } else res.matches.push_back(std::make_pair(track_idx, detection_idx));
    }
    return res;
}



