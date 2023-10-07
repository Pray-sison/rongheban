#include "track.h"
/**
 * tracker::_initiate_track时调用
 */
Track::Track(KAL_MEAN &mean, KAL_COVA &covariance, int track_id, int max_live, int class_id, float conf)
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
    // 初始化状态为存活
    this->state = this->state = TrackState::Live;
    // 丢失次数
    this->time_since_update = 0;
    // 最大存活时间
    this->max_live = max_live;
}

/**
 * 调用卡尔曼滤波预测，分为Live状态和Memory状态
 */
void Track::predit(KalmanFilter *kf)
{
    if(this->time_since_update + 1 > 1){
        this->state = TrackState::Memory;
        kf->predict_memory(this->mean);
    }else{
        this->state = TrackState::Live;
        kf->predict(this->mean, this->covariance);
    }
    this->time_since_update +=1;
}

/**
 * 卡尔曼滤波后验更新
 * @param kf 卡尔曼滤波器
 * @param detection 检测数据
 */
void Track::update(KalmanFilter *const kf, const DETECTION_ROW &detection)
{
    // 调用卡尔曼滤波利用观测值后验估计，并更新属性
    KAL_DATA pa = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = pa.first;
    this->covariance = pa.second;
    this->time_since_update = 0;
    this->conf = detection.conf;
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

bool Track::is_live()
{
    return this->state == TrackState::Live;
}

bool Track::is_memory()
{
    return this->state == TrackState::Memory;
}

bool Track::is_delete()
{
    return this->state == TrackState::Delete;
}

void Track::mark_missed()

{
    if (this->state == TrackState::Memory && this->time_since_update > this->max_live)
    {
        this->state = TrackState::Delete;
    }
}


