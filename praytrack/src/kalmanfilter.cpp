#include "kalmanfilter.h"
#include <Eigen/Cholesky>
KalmanFilter::KalmanFilter()
{
    int ndim = 4;
    double dt = 1.;

    _motion_mat = Eigen::MatrixXf::Identity(8, 8);

    for (int i = 0; i < ndim; i++)
    {
        _motion_mat(i, ndim + i) = dt;
    }

    _update_mat = Eigen::MatrixXf::Identity(4, 8);

    this->_std_weight_position = 1. / 20;

    this->_std_weight_velocity = 1. / 160;
}

/**
 * 卡尔曼滤波初始化
 * @param measurement 检测框
 * @return 初始化好的预测结果 X 和协方差矩阵 P
 */
KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement)
{
    DETECTBOX mean_pos = measurement;
    DETECTBOX mean_vel;
    for (int i = 0; i < 4; i++)
        mean_vel(i) = 0;
    // 初始化状态向量
    KAL_MEAN mean;
    for (int i = 0; i < 8; i++)
    {
        if (i < 4)
            mean(i) = mean_pos(i);
        else
            mean(i) = mean_vel(i - 4);
    }
    // 协方差矩阵构建
    KAL_MEAN std;
    std(0) = 2 * _std_weight_position * measurement[3];
    std(1) = 2 * _std_weight_position * measurement[3];
    std(2) = 1e-2;
    std(3) = 2 * _std_weight_position * measurement[3];
    std(4) = 10 * _std_weight_velocity * measurement[3];
    std(5) = 10 * _std_weight_velocity * measurement[3];
    std(6) = 1e-5;
    std(7) = 10 * _std_weight_velocity * measurement[3];

    // 求平方
    KAL_MEAN tmp = std.array().square();
    // 作为对角线构建矩阵
    KAL_COVA var = tmp.asDiagonal();
    // 返回当前状态与状态协方差矩阵
    return std::make_pair(mean, var);
}

/**
 * 卡尔曼滤波预测过程
 * @param mean 状态向量
 * @param covariance 上一帧的状态协方差矩阵
 */
void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
{
    DETECTBOX std_pos;
    std_pos << _std_weight_position * mean(3),
        _std_weight_position * mean(3),
        1e-2,
        _std_weight_position * mean(3);
    DETECTBOX std_vel;
    std_vel << _std_weight_velocity * mean(3),
        _std_weight_velocity * mean(3),
        1e-5,
        _std_weight_velocity * mean(3);
    KAL_MEAN tmp;
    // 位置与速度
    tmp.block<1, 4>(0, 0) = std_pos;
    tmp.block<1, 4>(0, 4) = std_vel;

    tmp = tmp.array().square();
    // 过程误差协方差矩阵  Q
    KAL_COVA motion_cov = tmp.asDiagonal();
    // 状态预测
    KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
    // 状态协方差矩阵更新
    KAL_COVA covariance1 = this->_motion_mat * covariance * (_motion_mat.transpose());
    // + Q
    covariance1 += motion_cov;

    mean = mean1;
    covariance = covariance1;
}

/**
 * 记忆状态下卡尔曼滤波预测，锁死宽高
 * @param mean 状态向量
 * @param covariance 上一帧的状态协方差矩阵
 */
void KalmanFilter::predict_memory(KAL_MEAN &mean)
{
    KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
    mean1(2) = mean(2);
    mean1(3) = mean(3);
    mean1(6) = 0;
    mean1(7) = 0;
    mean = mean1;
}

/**
 * 该方法计算的是卡尔曼系数K的中间变量，与图中的（）区域的内容匹配
 * 是更新过程的子过程
 * @param mean 当前状态X
 * @param covariance 协方差矩阵P
 * @return
 */
KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance)
{   // R
    DETECTBOX std;
    std << _std_weight_position * mean(3), _std_weight_position * mean(3),
        1e-1, _std_weight_position * mean(3);
    // 观测矩阵H * X
    KAL_HMEAN mean1 = _update_mat * mean.transpose();
    // 卡尔曼增益矩阵 括号里的HPH
    KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
    // R
    Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
    diag = diag.array().square().matrix();
    // 括号整体部分
    covariance1 += diag;
    //    covariance1.diagonal() << diag;
    return std::make_pair(mean1, covariance1);
}

/**
 * 卡尔曼滤波更新过程
 * @param mean 状态值
 * @param covariance 先验状态协方差
 * @param measurement 观测值
 * @return 后验更新好的X和P
 */
KAL_DATA
KalmanFilter::update(
    const KAL_MEAN &mean,
    const KAL_COVA &covariance,
    const DETECTBOX &measurement)
{
    KAL_HDATA pa = project(mean, covariance);
    KAL_HMEAN projected_mean = pa.first;
    KAL_HCOVA projected_cov = pa.second;

    // 计算括号外面的部分
    Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
    // 计算出卡尔曼系数K
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
    // 残差Y
    Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean;                // eg.1x4
    auto tmp = innovation * (kalman_gain.transpose());
    // 后验估计更新
    KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
    KAL_COVA new_covariance = covariance - kalman_gain * projected_cov * (kalman_gain.transpose());
    return std::make_pair(new_mean, new_covariance);
}
