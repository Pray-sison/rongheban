#include "track.h"
/**
 * tracker::_initiate_track时调用
 */
Track::Track(KAL_MEAN &mean, KAL_COVA &covariance, int track_id, int n_init, int max_age, const FEATURE &feature,  int class_id, float conf)
{
    // 目标检测框
    this->mean = mean;
    this->covariance = covariance;
    // 轨迹号
    this->track_id = track_id;
    // 目标类型
    this->class_id = class_id;
    // 目标类型
    this->conf = conf;

    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    // Tentative = 1, Confirmed, Deleted
    this->state = TrackState::Tentative;
    // 特征向量集合
    features = FEATURESS(1, k_feature_dim);
    features.row(0) = feature; // features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = max_age;
}

/**
 * 调用卡尔曼滤波预测，并更新属性
 */
void Track::predit(KalmanFilter *kf)
{
    // 更新了track的mean和covariance
    kf->predict(this->mean, this->covariance);
    // 年龄+1
    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(KalmanFilter *const kf, const DETECTION_ROW &detection)
{
    // 调用卡尔曼滤波利用观测值后验估计，并更新属性 tlwh -> xyah
    KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;
    //  添加一个新的表观特征
    featuresAppendOne(detection.feature);
    this->hits += 1;
    this->time_since_update = 0;
    this->conf = detection.conf;
    // 如果当前轨迹为Tentative并且hits>=3更新为Confirmed
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init)
    {
        this->state = TrackState::Confirmed;
    }
}

/**
 * 标记需要丢弃的轨迹
 */
void Track::mark_missed()
{
    if (this->state == TrackState::Tentative)
    {
        this->state = TrackState::Deleted;
    }
    else if (this->time_since_update > this->_max_age)
    {
        this->state = TrackState::Deleted;
    }
}


bool Track::is_confirmed()
{
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted()
{
    return this->state == TrackState::Deleted;
}

bool Track::is_tentative()
{
    return this->state == TrackState::Tentative;
}

/**
 * box格式转换
 */
DETECTBOX Track::to_tlwh()
{
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2) / 2);
    return ret;
}

/**
 * 将特征向量添加到轨迹
 * @param 表观特征向量
 */
void Track::featuresAppendOne(const FEATURE &f)
{
    int size = this->features.rows();
    FEATURESS newfeatures = FEATURESS(size + 1, k_feature_dim);
    newfeatures.block(0, 0, size, k_feature_dim) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}
