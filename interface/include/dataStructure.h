//
// Created by Pray on 2023/9/14.
//

#ifndef SUNWENBIN_DATASTRUCTURE_H
#define SUNWENBIN_DATASTRUCTURE_H

/**
 * 定义文件名路径长度
 */
#define ALG_NAME_SIZE

/**
 * 图像像素格式
 */
typedef enum AlgPixelFmt_e
{
    ALG_FMT_NONE = 0,
    ALG_FMT_YUV420SP,
    ALG_FMT_RGB888,
    ALG_FMT_BUTT,
}ALG_PIXEL_FMT_E;

/**
 * 跟踪目标类型
 * ALG_TRACK_TYPE_ORDINARY：普通目标
 * ALG_TRACK_TYPE_MOVING：移动目标
 */
typedef enum AlgTrackType_e
{
    ALG_TRACK_TYPE_NONE = 0,
    ALG_TRACK_TYPE_ORDINARY,
    ALG_TRACK_TYPE_MOVING,
    ALG_TRACK_TYPE_BUFF,
}ALG_TRACK_TYPE_E;

/**
 * 识别目标类型
 * 0：人
 * 1：普通车辆
 * 2：装甲车
 * 3：坦克
 * 4：指挥所
 * 其余预留
 */
typedef unsigned int ALG_TARGET_TYPE;

/**
 * 跟踪状态
 * ALG_TRACK_TYPE_NORMAL：正常跟踪状态
 * ALG_TRACK_TYPE_MEMORY：记忆（滑行）状态
 * ALG_TRACK_TYPE_LOSS：目标丢失状态
 */
typedef enum AlgTrackState_e
{
    ALG_TRACK_STATE_NONE = 0,
    ALG_TRACK_STATE_NORMAL,
    ALG_TRACK_STATE_MEMORY,
    ALG_TRACK_STATE_LOSS,
    ALG_TRACK_STATE_BUFF,
}ALG_TRACK_STATE_E;

/**
 * 图像信息
 * ALG_TRACK_TYPE_NORMAL：正常跟踪状态
 * ALG_TRACK_TYPE_MEMORY：记忆（滑行）状态
 * ALG_TRACK_TYPE_LOSS：目标丢失状态
 */
typedef struct AlgImgInfo_s
{
    ALG_PIXEL_FMT_E ePixelFmt;
    unsigned int u32Width;
    unsigned int u32Height;
    void *pImgData;
}ALG_IMG_INFO_S;

/**
 * 区域信息
 * u32Area_X：区域中心X点坐标
 * u32Area_Y：区域中心Y点坐标
 * u32Width：区域宽
 * u32Height：区域高
 */
typedef struct AlgArea_s
{
    unsigned int u32Area_X;
    unsigned int u32Area_Y;
    unsigned int u32Width;
    unsigned int u32Height;
}ALG_AREA_S;

/**
 * 跟踪输出化参数设置
 * stTrackArea：初始化区域
 */
typedef struct AlgTrackAttr_s
{
    ALG_AREA_S stTrackArea;
}ALG_TRACK_ATTR_S;

/**
 * 检测算法参数设置
 * aszModelPath：模型路径
 * f32NMS：非极大值抑制阈值 取值[0,1]
 * f32Conf: 置信度阈值 取值[0,1]
 */
typedef struct AlgDetectAttr_s
{
    float f32NMS;
    float f32Conf;
    const char *detectModelPath; // 使用指针管理不定长数组
    const char *trackModelPath; // 使用指针管理另一个不定长数组
}ALG_DETECT_ATTR_S;

/**
 * 算法模块属性
 * stTrackAttr：跟踪算法属性
 * stDetectAttr：检测算法属性
 */
typedef struct AlgAttr_s
{
    ALG_TRACK_ATTR_S stTrackAttr;
    ALG_DETECT_ATTR_S stDetectAttr;
}ALG_ATTR_S;

/**
 * 跟踪处理结果
 * eTrackState：跟踪状态
 * stArea：目标区域
 * f32TrackOffset：跟踪偏差
 */
typedef struct AlgTrackRet_s
{
    ALG_TRACK_STATE_E eTrackState;
    ALG_AREA_S stArea;
    float f32TrackOffset;
}ALG_TRACK_RET_S;

/**
 * 检测处理结果
 * s32TargetID：跟踪目标编号
 * u32TargetType：目标类型
 * stArea：目标区域
 * f32conf：置信度
 */
typedef struct AlgDetectRet
{
    int s32TargetID;
    ALG_TARGET_TYPE u32TargetType;
    ALG_AREA_S stArea;
    float f32conf;
    int isMove;
}ALG_DETECT_RET_S;


/**
 * 用于API接口执行结果返回，成功返回ALG_SUCCESS，失败返回对应错误类型。
 * ALG_SUCCESS：成功
 * ALG_ERR_XXX：XXX异常
 */
typedef enum AlgErrId_e
{
    ALG_SUCCESS = 0,
    ALG_ERR_XXX,
}ALG_ERR_ID_E;
#endif //SUNWENBIN_DATASTRUCTURE_H
