//
// Created by Pray on 2023/9/14.
//

#define PRINT_ENABLED true

#ifndef SUNWENBIN_INTERFACE_H
#define SUNWENBIN_INTERFACE_H


#include "dataStructure.h"
#include "yolo.h"
#include "praytracker.h"
#include "kcftracker.hpp"

/**
 * 初始化算法
 * @param pstAlgAttr 初始化参数设置
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_Init(ALG_ATTR_S *pstAlgAttr);

/**
 * 反初始化算法
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_DeInit();

/**
 * 跟踪算法处理
 * @param pstImgInfo 图像信息（传入）
 * @param pstTrackRet 跟踪处理接触（传入传出）
 * @param restartFlag 每次重新开始传入1，检测状态传入0
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_TrackProc(ALG_IMG_INFO_S *pstImgInfo, ALG_TRACK_RET_S *pstTrackRet, int restartFlag);

/**
 * 更新跟踪区域，需要修改跟踪区域时调用该接口
 * @param pstTrackArea
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_UpdateTrackArea(ALG_AREA_S *pstTrackArea);

/**
 * 设置跟踪目标类型，用于切换跟踪目标类型时调用该结构
 * @param eTrackType 跟踪目标类型（输入）
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_SetTrackType(ALG_TRACK_STATE_E eTrackType);

/**
 * 识别处理算法
 * @param pstImgInfo 图像信息（传入参数）
 * @param u32TargetType 识别目标类型（传入）
 * @param s32Targets 检测个数（传入）
 * @param astDetectRet 检测结果（传入传出）
 * @param restartFlag 每次重新开始传入1，检测状态传入0
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_DetectProc(ALG_IMG_INFO_S *pstImgInfo, ALG_TARGET_TYPE u32TargetType, int s32Targets, ALG_DETECT_RET_S astDetectRet[], int restartFlag);



#endif //SUNWENBIN_INTERFACE_H
