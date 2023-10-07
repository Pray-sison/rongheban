//
// Created by Pray on 2023/9/14.
//
#include "interface.h"
#include <bitset>
#include <chrono>

YOLOv5 *yolov5 = nullptr;
PrayTracker *prayTracker = nullptr;

//Feature *feature = nullptr;
//MultiTracker *multiTracker = nullptr;
//

bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool LAB = true;

KCFTracker *kcfTracker = nullptr;

// 跟踪区域
cv::Rect kcfArea;

int KCF_X;
int KCF_Y;
int KCF_W;
int KCF_H;

// KCF是否初始化标志，默认为初始化状态
int isInit = 1;

// KCF是否在记忆状态，默认为不在记忆状态
int isMemory = 0;

// 初始视频状态
ALG_PIXEL_FMT_E curVideoFMT = ALG_FMT_NONE;

// 记录丢失时间的工具
std::chrono::steady_clock::time_point preTime;
std::chrono::steady_clock::time_point curTime;

// 最大记忆时间
const int maxTime = 10000;

// 用于上一帧目标轨迹状态
std::map<int, std::array<int, 2>> detectMap;
const float disThreshold = 0.5;

// 用于差分的前置图像
cv::Mat preFrame;

cv::Ptr<cv::BackgroundSubtractorMOG2> fgbg = cv::createBackgroundSubtractorMOG2(1, 8, false);


// 计算框的中心点距离屏幕中心的距离
//static float distanceToCenter(const box_rect* box, int screen_width, int screen_height) {
//    int center_x = (box->left + box->right) / 2;
//    int center_y = (box->top + box->bottom) / 2;
//    int center_screen_x = screen_width / 2;
//    int center_screen_y = screen_height / 2;
//
//    return sqrt(pow(center_x - center_screen_x, 2) + pow(center_y - center_screen_y, 2));
//}

//int compare(const void* a, const void* b) {
//    detect_result_t* result1 = (detect_result_t*)a;
//    detect_result_t* result2 = (detect_result_t*)b;
//
//    int screen_width = 1920; // 屏幕宽度
//    int screen_height = 1080; // 屏幕高度
//
//    float distance1 = distanceToCenter(&result1->box, screen_width, screen_height);
//    float distance2 = distanceToCenter(&result2->box, screen_width, screen_height);
//
//    // 如果距离不相等，根据距离排序
//    if (distance1 < distance2) {
//        return -1;
//    } else if (distance1 > distance2) {
//        return 1;
//    } else {
//        return 0;
//    }
//}

/**
 * 初始化算法
 * @param pstAlgAttr 初始化参数设置
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_Init(ALG_ATTR_S *pstAlgAttr) {
    // 创建类
    yolov5 = new YOLOv5();
//    feature = new Feature();
    prayTracker = new PrayTracker();
    kcfTracker = new KCFTracker();
    // 调用初始化方法
    yolov5->init(pstAlgAttr->stDetectAttr.detectModelPath, pstAlgAttr->stDetectAttr.f32Conf, pstAlgAttr->stDetectAttr.f32NMS);
//    feature->init(pstAlgAttr->stDetectAttr.trackModelPath);
    prayTracker->init();
    kcfTracker->init(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    KCF_X = pstAlgAttr->stTrackAttr.stTrackArea.u32Area_X;
    KCF_Y = pstAlgAttr->stTrackAttr.stTrackArea.u32Area_Y;
    KCF_W = pstAlgAttr->stTrackAttr.stTrackArea.u32Width;
    KCF_H = pstAlgAttr->stTrackAttr.stTrackArea.u32Height;
    // 初始化KCF跟踪区域
    kcfArea = cv::Rect(KCF_X, KCF_Y, KCF_W, KCF_H);

    return ALG_SUCCESS;
}

/**
 * 反初始化算法
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_DeInit() {
    delete yolov5;
    yolov5 = nullptr;
//    delete feature;
//    feature = nullptr;
    delete prayTracker;
    prayTracker = nullptr;
    delete kcfTracker;
    kcfTracker = nullptr;
    return ALG_SUCCESS;
}

/**
 * 跟踪算法处理
 * @param pstImgInfo 图像信息（传入）
 * @param pstTrackRet 跟踪处理接触（传入传出）
 * @param restartFlag 每次重新开始传入1，检测状态传入0
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_TrackProc(ALG_IMG_INFO_S *pstImgInfo, ALG_TRACK_RET_S *pstTrackRet, int restartFlag) {
    // 如果图像类型发生发生变化，更改全局视频格式，同时让跟踪表示置1
    if(pstImgInfo->ePixelFmt == ALG_FMT_NONE){
        return ALG_ERR_XXX;
    }
    if(restartFlag == 1){
        isInit = 1;
    }
    if (pstImgInfo->ePixelFmt != curVideoFMT) {
        curVideoFMT = pstImgInfo->ePixelFmt;
        isInit = 1;
    }
    unsigned int width = pstImgInfo->u32Width;
    unsigned int height = pstImgInfo->u32Height;
    cv::Mat image(height, width, CV_8UC3);
    // 将 void* 指针转换为 unsigned char*（即 uint8_t*）
    unsigned char *imageData = static_cast<unsigned char *>(pstImgInfo->pImgData);
    // 复制图像数据
    std::memcpy(image.data, imageData, width * height * 3);

    // 获取目标跟踪结果
    // 如果是第一次，或者图像类型发生改变 重新init_roi
    if (isInit == 1) {
        // 初始化区域
        kcfTracker->init_roi(kcfArea, image);
        // 初始化后进入跟踪状态
        isInit = 0;
    } else {    // 正常状态下，继续更新
        if (isMemory == 0) {
            cv::Rect kcfRes = kcfTracker->update_roi(image);
            // 发现找不到目标
            if (kcfTracker->trustable < 0.30) {
                // 进入记忆状态
                isMemory = 1;
                // 记录时间
                preTime = std::chrono::steady_clock::now();
                // 返回记忆状态, 不返回跟踪区域
                pstTrackRet->eTrackState = ALG_TRACK_STATE_MEMORY;
            } else {    // 返回正常状态跟踪结果区域
                pstTrackRet->eTrackState = ALG_TRACK_STATE_NORMAL;
                pstTrackRet->stArea.u32Area_X = kcfRes.x;
                pstTrackRet->stArea.u32Area_Y = kcfRes.y;
                pstTrackRet->stArea.u32Width = kcfRes.width;
                pstTrackRet->stArea.u32Height = kcfRes.height;
                // 正常状态一直重置丢失时间点
                preTime = std::chrono::steady_clock::now();
#ifdef DEBUG_MODE
                cv::rectangle(image, kcfRes, cv::Scalar(0, 255, 0), 2);
#endif
            }
        } else {    // 记忆状态下，在丢失周围继续搜索
            cv::Rect kcfRes = kcfTracker->updateMemory_roi(image);
            // 在周围搜索到目标，进入正常状态
            if (kcfTracker->trustable >= 0.30) {
                isMemory = 0;
                // 返回正常状态跟踪结果区域
                pstTrackRet->eTrackState = ALG_TRACK_STATE_NORMAL;
                pstTrackRet->stArea.u32Area_X = kcfRes.x;
                pstTrackRet->stArea.u32Area_Y = kcfRes.y;
                pstTrackRet->stArea.u32Width = kcfRes.width;
                pstTrackRet->stArea.u32Height = kcfRes.height;
                // 正常状态一直重置丢失时间点
                preTime = std::chrono::steady_clock::now();
#ifdef DEBUG_MODE
                cv::rectangle(image, kcfRes, cv::Scalar(0, 255, 0), 2);
#endif
            } else {    // 返回记忆状态, 不返回跟踪区域
                curTime = std::chrono::steady_clock::now();
                // 获取时间差
                std::chrono::steady_clock::duration timeDifference = curTime - preTime;
                // 获取时间差的毫秒数
                std::chrono::milliseconds timeDifferenceMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeDifference);
                // 超过了最大记忆时间。返回目标丢失
                if (timeDifferenceMs.count() > maxTime){
                    pstTrackRet->eTrackState = ALG_TRACK_STATE_LOSS;
                    kcfArea = cv::Rect(KCF_X, KCF_Y, KCF_W, KCF_H);
                    isInit = 1;
                } else{
                    pstTrackRet->eTrackState = ALG_TRACK_STATE_MEMORY;
                }
            }
        }
    }
#ifdef DEBUG_MODE
    // 显示图像
    cv::imshow("Tracked Image", image);
    cv::waitKey(1);
#endif
    return ALG_SUCCESS;
}

/**
 * 更新跟踪区域，需要修改跟踪区域时调用该接口
 * @param pstTrackArea
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_UpdateTrackArea(ALG_AREA_S *pstTrackArea) {
    kcfArea.x = pstTrackArea->u32Area_X;
    kcfArea.y = pstTrackArea->u32Area_Y;
    kcfArea.width = pstTrackArea->u32Width;
    kcfArea.height = pstTrackArea->u32Height;
    isInit = 1;
    return ALG_SUCCESS;
}

/**
 * 设置是否开启目标跟踪自适应大小
 * @param isSelfAdaption 自适应标志 0：关闭自适应 1或其他：开启自适应
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_SetSelfAdaption(int isSelfAdaption) {
    if(isSelfAdaption == 0){
        HOG = true;
        FIXEDWINDOW = true;
        MULTISCALE = false;
        LAB = true;
        kcfTracker->init(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    } else{
        HOG = true;
        FIXEDWINDOW = false;
        MULTISCALE = true;
        LAB = true;
        kcfTracker->init(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    }
    return ALG_SUCCESS;
}


/**
 * 识别处理算法
 * @param pstImgInfo 图像信息（传入参数）
 * @param u32TargetType 识别目标类型（传入）
 * @param s32Targets 检测个数（传入）
 * @param astDetectRet 检测结果（传入传出）
 * @param restartFlag 每次重新开始传入1，检测状态传入0
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_DetectProc(ALG_IMG_INFO_S *pstImgInfo, ALG_TARGET_TYPE u32TargetType,
                       int s32Targets, ALG_DETECT_RET_S astDetectRet[], int restartFlag) {
    if(restartFlag == 1){
        prayTracker->tracks.clear();
        prayTracker->_next_idx = 1;
        detectMap.clear();
    }
    if (pstImgInfo->ePixelFmt != ALG_FMT_RGB888){
        prayTracker->tracks.clear();
        prayTracker->_next_idx = 1;
        detectMap.clear();
        return ALG_ERR_XXX;
    }

    unsigned int width = pstImgInfo->u32Width;
    unsigned int height = pstImgInfo->u32Height;
    cv::Mat image(height, width, CV_8UC3);
    // 将 void* 指针转换为 unsigned char*（即 uint8_t*）
    unsigned char *imageData = static_cast<unsigned char *>(pstImgInfo->pImgData);
    // 复制图像数据
    std::memcpy(image.data, imageData, width * height * 3);
    // 获取目标检测结果
    detect_result_group_t detect_result_group;
    detect_result_group = yolov5->yolo_detect(image);
    // 将整数转换为二进制表示
    std::bitset<32> binary(u32TargetType); // 32位表示
    DETECTIONS detections;
    yolov5->format_conversion(detect_result_group, detections);
    DETECTIONS detections2;
    // 按目标类型过滤
    int num = 0;
    for (DETECTION_ROW det: detections) {
        for (int i = 0; i < 32; ++i) {
            if (binary[i] == 1 && det.class_id == i) {
                detections2.push_back(det);
                num++;
                break;
            }
        }
        if (num >= s32Targets){
            break;
        }
    }
    // 轨迹预测与更新
    prayTracker->predict();
    prayTracker->update(detections2);
    // 获取目标跟踪结果
    std::vector<tracker_result_data_t> tracker_result_data;
    prayTracker->get_tracker_result(tracker_result_data);
    // 封装结构体返回
    for (int i = 0; i < s32Targets; i++) {
        if(i < tracker_result_data.size()){
            // 编号
            astDetectRet[i].s32TargetID = tracker_result_data.at(i).track_id;
            // 类别
            astDetectRet[i].u32TargetType = tracker_result_data.at(i).class_id;
            // 位置
            astDetectRet[i].stArea.u32Area_X = (unsigned int)tracker_result_data.at(i).box(0);
            astDetectRet[i].stArea.u32Area_Y = (unsigned int)tracker_result_data.at(i).box(1);
            astDetectRet[i].stArea.u32Width = (unsigned int)tracker_result_data.at(i).box(2);
            astDetectRet[i].stArea.u32Height = (unsigned int)tracker_result_data.at(i).box(3);
            // 置信度
            astDetectRet[i].f32conf = tracker_result_data.at(i).conf;
            // 判断是否为新轨迹
            auto it = detectMap.find(astDetectRet[i].s32TargetID);
            if (it != detectMap.end()) {
                // 找到了特定的键，计算偏移距离

                float moveDis2 = std::pow(((int)astDetectRet[i].stArea.u32Area_X   - detectMap[astDetectRet[i].s32TargetID][0]), 2)
                               + std::pow(((int)astDetectRet[i].stArea.u32Area_Y - detectMap[astDetectRet[i].s32TargetID][1]), 2);
                float moveDis = std::sqrt(moveDis2);
                if(moveDis >= disThreshold){
                    astDetectRet[i].isMove = 1;
                    cv::rectangle(image, cv::Rect(astDetectRet[i].stArea.u32Area_X, astDetectRet[i].stArea.u32Area_Y,
                                                  astDetectRet[i].stArea.u32Width, astDetectRet[i].stArea.u32Height
                    ), cv::Scalar(255, 0, 0), 2);
                } else{
                    astDetectRet[i].isMove = 0;
                    cv::rectangle(image, cv::Rect(astDetectRet[i].stArea.u32Area_X, astDetectRet[i].stArea.u32Area_Y,
                                                  astDetectRet[i].stArea.u32Width, astDetectRet[i].stArea.u32Height
                    ), cv::Scalar(0, 255, 0), 2);
                }
                // 重新记录数据
                std::array<int, 2> values = {{(int)astDetectRet[i].stArea.u32Area_X, (int)astDetectRet[i].stArea.u32Area_Y}};
                detectMap[astDetectRet[i].s32TargetID] = values;

            } else {
                // 未找到特定的键，添加到记录中
                std::array<int, 2> values = {{(int)astDetectRet[i].stArea.u32Area_X, (int)astDetectRet[i].stArea.u32Area_Y}};
                detectMap[astDetectRet[i].s32TargetID] = values;
            }
        } else{
            // 编号
            astDetectRet[i].s32TargetID = -1;
        }
    }
#ifdef DEBUG_MODE
    // 显示图像
    cv::imshow("Tracked Image", image);
    cv::waitKey(1);
#endif
    return ALG_SUCCESS;
}

/**
 * 动目标检测
 * @param pstImgInfo 图像信息（传入参数）
 * @param astDetectRet 检测结果（传入传出）
 * @param restartFlag 是否重新启动（传入参数）
 * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
 */
int XXX_ALG_MoveDetectProc(ALG_IMG_INFO_S *pstImgInfo, int s32Targets, ALG_DETECT_RET_S astDetectRet[], int restartFlag) {
    unsigned int width = pstImgInfo->u32Width;
    unsigned int height = pstImgInfo->u32Height;
    cv::Mat image(height, width, CV_8UC3);
    // 将 void* 指针转换为 unsigned char*（即 uint8_t*）
    unsigned char *imageData = static_cast<unsigned char *>(pstImgInfo->pImgData);
    // 复制图像数据
    std::memcpy(image.data, imageData, width * height * 3);

    if (restartFlag == 1) {
        preFrame = image.clone();
        return ALG_SUCCESS;
    } else {
        // 缩小图像一半
        cv::Mat small_image;
        cv::resize(image, small_image, cv::Size(width / 4, height / 4));

        cv::Mat fgmask;
        fgbg->apply(small_image, fgmask);

        cv::morphologyEx(fgmask, fgmask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t i = 0, j = 0; i < contours.size(); i++) {
            if (cv::contourArea(contours[i]) > 30) {
                if (j < s32Targets){
                    cv::Rect rect = cv::boundingRect(contours[i]);
                    // 将边界框坐标还原到原图
                    rect.x *= 4;
                    rect.y *= 4;
                    rect.width *= 4;
                    rect.height *= 4;
#ifdef DEBUG_MODE
                    cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
#endif
                    // 编号
                    astDetectRet[j].s32TargetID = -1;
                    // 类别
                    astDetectRet[j].u32TargetType = 0;
                    // 位置
                    astDetectRet[j].stArea.u32Area_X = rect.x;
                    astDetectRet[j].stArea.u32Area_Y = rect.y;
                    astDetectRet[j].stArea.u32Width = rect.width;
                    astDetectRet[j].stArea.u32Height = rect.height;
                    j++;
                }else{
                    break;
                }
            }
        }
        preFrame = image.clone();
#ifdef DEBUG_MODE
        cv::imshow("Video", image);
        cv::waitKey(1);
#endif
    }
    return ALG_SUCCESS;
}















///**
// * 识别处理算法
// * @param pstImgInfo 图像信息（传入参数）
// * @param u32TargetType 识别目标类型（传入）
// * @param s32Targets 检测个数（传入）
// * @param astDetectRet 检测结果（传入传出）
// * @param restartFlag 每次重新开始传入1，检测状态传入0
// * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
// */
//int XXX_ALG_DetectProc(ALG_IMG_INFO_S *pstImgInfo, ALG_TARGET_TYPE u32TargetType,
//                       int s32Targets, ALG_DETECT_RET_S astDetectRet[], int restartFlag) {
//    if(restartFlag == 1){
//        multiTracker->tracks.clear();
//    }
//    if (pstImgInfo->ePixelFmt != ALG_FMT_RGB888){
//        return ALG_ERR_XXX;
//    }
//    unsigned int width = pstImgInfo->u32Width;
//    unsigned int height = pstImgInfo->u32Height;
//    cv::Mat image(height, width, CV_8UC3);
//    // 将 void* 指针转换为 unsigned char*（即 uint8_t*）
//    unsigned char *imageData = static_cast<unsigned char *>(pstImgInfo->pImgData);
//    // 复制图像数据
//    std::memcpy(image.data, imageData, width * height * 3);
//    // 获取目标检测结果
//    detect_result_group_t detect_result_group;
//    detect_result_group = yolov5->yolo_detect(image);
//    // 将整数转换为二进制表示
//    std::bitset<32> binary(u32TargetType); // 32位表示
//    detect_result_group_t detect_result_group2;
//    detect_result_group2.count = 0;
//    for(int i = 0, j = 0; i < detect_result_group.count; i++){
//        for(int k = 0; k < 32; k++){
//            if (binary[k] == 1 && detect_result_group.results[i].class_id == k){
//                detect_result_group2.count ++;
//                detect_result_group2.results[j++] = detect_result_group.results[i];
//                break;
//            }
//        }
//        if (j >= s32Targets){
//            break;
//        }
//    }
//    // 排序结果数组
//    qsort(detect_result_group2.results, detect_result_group2.count, sizeof(detect_result_t), compare);
//
//    // 封装结构体返回
//    for (int i = 0; i < s32Targets; i++) {
//        if(i < detect_result_group2.count){
//            // 编号
//            astDetectRet[i].s32TargetID = i + 1;
//            // 类别
//            astDetectRet[i].u32TargetType = detect_result_group2.results[i].class_id;
//            // 位置
//            astDetectRet[i].stArea.u32Area_X = detect_result_group2.results[i].box.left;
//            astDetectRet[i].stArea.u32Area_Y = detect_result_group2.results[i].box.top;
//            astDetectRet[i].stArea.u32Width = detect_result_group2.results[i].box.right - detect_result_group2.results[i].box.left;
//            astDetectRet[i].stArea.u32Height = detect_result_group2.results[i].box.bottom - detect_result_group2.results[i].box.top;
//            // 置信度
//            astDetectRet[i].f32conf = detect_result_group2.results[i].conf;
//        } else{
//            // 编号
//            astDetectRet[i].s32TargetID = -1;
//        }
//    }
//    return ALG_SUCCESS;
//}

///**
// * 识别处理算法
// * @param pstImgInfo 图像信息（传入参数）
// * @param u32TargetType 识别目标类型（传入）
// * @param s32Targets 检测个数（传入）
// * @param astDetectRet 检测结果（传入传出）
// * @param restartFlag 每次重新开始传入1，检测状态传入0
// * @return 状态值 ALG_SUCCESS：成功 ALG_ERR_XXX 对应的失败类型
// */
//int XXX_ALG_DetectProc(ALG_IMG_INFO_S *pstImgInfo, ALG_TARGET_TYPE u32TargetType,
//                       int s32Targets, ALG_DETECT_RET_S astDetectRet[], int restartFlag) {
//    std::chrono::steady_clock::time_point p = std::chrono::steady_clock::now();
//    if(restartFlag == 1){
//        multiTracker->tracks.clear();
//    }
//    if (pstImgInfo->ePixelFmt != ALG_FMT_RGB888){
//        return ALG_ERR_XXX;
//    }
//    unsigned int width = pstImgInfo->u32Width;
//    unsigned int height = pstImgInfo->u32Height;
//    cv::Mat image(height, width, CV_8UC3);
//    // 将 void* 指针转换为 unsigned char*（即 uint8_t*）
//    unsigned char *imageData = static_cast<unsigned char *>(pstImgInfo->pImgData);
//    // 复制图像数据
//    std::memcpy(image.data, imageData, width * height * 3);
//    std::chrono::steady_clock::time_point q = std::chrono::steady_clock::now();;
//    // 计算时间差
//    std::chrono::duration<double, std::milli> durationp = q - p;
//    // 输出时间差
//    std::cout << "convertFmt: " << durationp.count() << " ms" << std::endl;
//    // 获取目标检测结果
//    detect_result_group_t detect_result_group;
//    std::chrono::steady_clock::time_point a = std::chrono::steady_clock::now();
//    detect_result_group = yolov5->yolo_detect(image);
//    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();;
//    // 计算时间差
//    std::chrono::duration<double, std::milli> duration = b - a;
//
//    // 输出时间差
//    std::cout << "detect: " << duration.count() << " ms" << std::endl;
//
//    // 将整数转换为二进制表示
//    std::chrono::steady_clock::time_point c = std::chrono::steady_clock::now();;
//    std::bitset<32> binary(u32TargetType); // 32位表示
//    DETECTIONS detections;
//    yolov5->format_conversion(image, detect_result_group, detections, *feature);
//    yolov5->format_conversion(image, detect_result_group, detections, *feature);
//    std::chrono::steady_clock::time_point d = std::chrono::steady_clock::now();
//    DETECTIONS detections2;
//    int num = 0;
//    for (DETECTION_ROW det: detections) {
//        for (int i = 0; i < 32; ++i) {
//            if (binary[i] == 1 && det.class_id == i) {
//                detections2.push_back(det);
//                num++;
//                break;
//            }
//        }
//        if (num >= s32Targets){
//            break;
//        }
//    }
//    std::chrono::duration<double, std::milli> duration3 = d - c;
//
//    // 输出时间差
//    std::cout << "feature: " << duration3.count() << " ms" << std::endl;
//
//    // 轨迹预测与更新
//    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();;
//    multiTracker->predict();
//    multiTracker->update(detections2);
//    std::chrono::steady_clock::time_point r = std::chrono::steady_clock::now();;
//    // 计算时间差
//    std::chrono::duration<double, std::milli> duration4 = r - e;
//
//    // 输出时间差
//    std::cout << "track: " << duration4.count() << " ms" << std::endl;
//    // 获取目标跟踪结果
//
//    std::vector<tracker_result_data_t> tracker_result_data;
//
//    multiTracker->get_tracker_result(tracker_result_data);
//
//    // 过滤有效目标
//
//    // 封装结构体返回
//    for (int i = 0; i < s32Targets; i++) {
//        if(i < tracker_result_data.size()){
//            // 编号
//            astDetectRet[i].s32TargetID = tracker_result_data.at(i).track_id;
//            // 类别
//            astDetectRet[i].u32TargetType = tracker_result_data.at(i).class_id;
//            // 位置
//            astDetectRet[i].stArea.u32Area_X = (unsigned int)tracker_result_data.at(i).box(0) + (unsigned int)tracker_result_data.at(i).box(2) / 2;
//            astDetectRet[i].stArea.u32Area_Y = (unsigned int)tracker_result_data.at(i).box(1) + (unsigned int)tracker_result_data.at(i).box(3) / 2;
//            astDetectRet[i].stArea.u32Width = (unsigned int)tracker_result_data.at(i).box(2);
//            astDetectRet[i].stArea.u32Height = (unsigned int)tracker_result_data.at(i).box(3);
//            // 置信度
//            astDetectRet[i].f32conf = tracker_result_data.at(i).conf;
//        } else{
//            // 编号
//            astDetectRet[i].s32TargetID = -1;
//        }
//    }
//    std::cout << "======================================" << std::endl;
//    return ALG_SUCCESS;
//}

