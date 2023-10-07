#ifndef TRACK_H
#define TRACK_H

#include "dataType.h"
#include "kalmanfilter.h"
#include "model.h"
#include <iostream>
class Track
{
    enum TrackState {Live = 1, Memory, Delete};

public:
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id, int max_live, int class_id, float conf);
    void predit(KalmanFilter *kf);
    void update(KalmanFilter * const kf, const DETECTION_ROW &detection);
    void mark_missed();
    bool is_live();
    bool is_delete();
    // 调试用
    bool is_memory();
    DETECTBOX to_tlwh();
    int time_since_update;
    int track_id;
    int class_id;
    float conf;
    KAL_MEAN mean;
    KAL_COVA covariance;
    TrackState state;
    int max_live;
};

#endif // TRACK_H
