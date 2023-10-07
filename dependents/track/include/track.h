#ifndef TRACK_H
#define TRACK_H

#include "dataType.h"
#include "kalmanfilter.h"
#include "model.h"

class Track
{
    enum TrackState {Tentative = 1, Confirmed, Deleted};

public:
    Track(KAL_MEAN &mean, KAL_COVA &covariance, int track_id,
          int n_init, int max_age, const FEATURE &feature,  int class_id, float conf);
    void predit(KalmanFilter *kf);
    void update(KalmanFilter * const kf, const DETECTION_ROW &detection);
    void mark_missed();
    bool is_confirmed();
    bool is_deleted();
    bool is_tentative();
    DETECTBOX to_tlwh();
    int time_since_update;
    int track_id;
    int class_id;
    float conf;
    FEATURESS features;
    KAL_MEAN mean;
    KAL_COVA covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;
private:
    void featuresAppendOne(const FEATURE& f);
};

#endif // TRACK_H
