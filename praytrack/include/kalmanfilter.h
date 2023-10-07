#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "dataType.h"

class KalmanFilter
{
public:
    KalmanFilter();
    KAL_DATA initiate(const DETECTBOX& measurement);
    void predict(KAL_MEAN& mean, KAL_COVA& covariance);
    void predict_memory(KAL_MEAN& mean);
    KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
    KAL_DATA update(const KAL_MEAN& mean,
                    const KAL_COVA& covariance,
                    const DETECTBOX& measurement);

private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
};

#endif // KALMANFILTER_H
