#ifndef LINEAR_ASSIGNMENT_H
#define LINEAR_ASSIGNMENT_H
#include "dataType.h"
#include "praytracker.h"

class PrayTracker;
class linear_assignment
{
    linear_assignment();
    linear_assignment(const linear_assignment& );
    linear_assignment& operator=(const linear_assignment&);
    static linear_assignment* instance;

public:
    static linear_assignment* getInstance();
    TRACHER_MATCHD min_cost_matching(
            PrayTracker* distance_metric,
            PrayTracker::GATED_METRIC_FUNC distance_metric_func,
            float max_distance,
            std::vector<Track>& tracks,
            const DETECTIONS& detections,
            std::vector<int>& track_indices,
            std::vector<int>& detection_indices);
};

#endif // LINEAR_ASSIGNMENT_H
