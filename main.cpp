#include <interface.h>

void tsetMoveDetect() {
    int flag = 1;
    cv::VideoCapture cap(9);
    ALG_ATTR_S pstAlgAttr; // 创建结构体对象
    pstAlgAttr.stDetectAttr.detectModelPath = "../visdrone.rknn"; //检测模型
    pstAlgAttr.stDetectAttr.trackModelPath = "../feature.rknn";     //跟踪模型
    pstAlgAttr.stDetectAttr.f32Conf = 0.6;
    pstAlgAttr.stDetectAttr.f32NMS = 0.4;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Area_X = 960;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Area_Y = 540;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Width = 256;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Height = 128;

    XXX_ALG_Init(&pstAlgAttr);

    XXX_ALG_SetSelfAdaption(0);

    cv::Mat orig_frame;
    while (cap.read(orig_frame)) {
        // 获取图像的宽度和高度
        // 强制将图像大小调整为1080x1920
        cv::resize(orig_frame, orig_frame, cv::Size(1920, 1080));
        ALG_IMG_INFO_S pstImgInfo; // 假设已经初始化了 pstImgInfo
        pstImgInfo.u32Width = static_cast<unsigned int>(orig_frame.cols);
        pstImgInfo.u32Height = static_cast<unsigned int>(orig_frame.rows);

        // 设置图像的像素格式，假设是 BGR 格式
        pstImgInfo.ePixelFmt = ALG_FMT_RGB888;

        // 分配内存并将图像数据复制到 pstImgInfo.pImgData
        pstImgInfo.pImgData = malloc(orig_frame.total() * orig_frame.elemSize());
        if (pstImgInfo.pImgData != nullptr) {
            // 复制图像数据
            std::cerr << "Memory allocation success." << std::endl;
            std::memcpy(pstImgInfo.pImgData, orig_frame.data, orig_frame.total() * orig_frame.elemSize());
        } else {
            // 处理内存分配失败的情况
            std::cerr << "Memory allocation failed." << std::endl;
        }
        int targetSize = 20;
        ALG_DETECT_RET_S astDetectRet[targetSize];

        if(flag == 1){
            XXX_ALG_MoveDetectProc(&pstImgInfo, targetSize, astDetectRet, flag);
            flag = 0;
        } else{
            XXX_ALG_MoveDetectProc(&pstImgInfo, targetSize, astDetectRet, flag);
        }
    }
    XXX_ALG_DeInit();
}


void tsetTrack() {
    cv::VideoCapture cap(9);
    ALG_ATTR_S pstAlgAttr; // 创建结构体对象
    pstAlgAttr.stDetectAttr.detectModelPath = "../visdrone.rknn"; //检测模型
    pstAlgAttr.stDetectAttr.trackModelPath = "../feature.rknn";     //跟踪模型
    pstAlgAttr.stDetectAttr.f32Conf = 0.6;
    pstAlgAttr.stDetectAttr.f32NMS = 0.4;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Area_X = 960;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Area_Y = 540;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Width = 256;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Height = 128;

    XXX_ALG_Init(&pstAlgAttr);

    XXX_ALG_SetSelfAdaption(0);

    cv::Mat orig_frame;
    while (cap.read(orig_frame)) {
        // 获取图像的宽度和高度
        // 强制将图像大小调整为1080x1920
        cv::resize(orig_frame, orig_frame, cv::Size(1920, 1080));
        ALG_IMG_INFO_S pstImgInfo; // 假设已经初始化了 pstImgInfo
        pstImgInfo.u32Width = static_cast<unsigned int>(orig_frame.cols);
        pstImgInfo.u32Height = static_cast<unsigned int>(orig_frame.rows);

        // 设置图像的像素格式，假设是 BGR 格式
        pstImgInfo.ePixelFmt = ALG_FMT_RGB888;

        // 分配内存并将图像数据复制到 pstImgInfo.pImgData
        pstImgInfo.pImgData = malloc(orig_frame.total() * orig_frame.elemSize());
        if (pstImgInfo.pImgData != nullptr) {
            // 复制图像数据
            std::cerr << "Memory allocation success." << std::endl;
            std::memcpy(pstImgInfo.pImgData, orig_frame.data, orig_frame.total() * orig_frame.elemSize());
        } else {
            // 处理内存分配失败的情况
            std::cerr << "Memory allocation failed." << std::endl;
        }
        ALG_TRACK_RET_S alg_track_ret_s;
        XXX_ALG_TrackProc(&pstImgInfo, &alg_track_ret_s, 0);
    }
    XXX_ALG_DeInit();
}


void tsetDetect() {
    cv::VideoCapture cap("../test_1.avi");

    ALG_ATTR_S pstAlgAttr; // 创建结构体对象
    pstAlgAttr.stDetectAttr.detectModelPath = "../visdrone.rknn";
    pstAlgAttr.stDetectAttr.trackModelPath = "../feature.rknn";
    pstAlgAttr.stDetectAttr.f32Conf = 0.6;
    pstAlgAttr.stDetectAttr.f32NMS = 0.4;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Area_X = 960;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Area_Y = 540;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Width = 256;
    pstAlgAttr.stTrackAttr.stTrackArea.u32Height = 128;
    XXX_ALG_Init(&pstAlgAttr);
    cv::Mat orig_frame;
    while (cap.read(orig_frame)) {

        // 获取图像的宽度和高度
        ALG_IMG_INFO_S pstImgInfo; // 假设已经初始化了 pstImgInfo
        pstImgInfo.u32Width = static_cast<unsigned int>(orig_frame.cols);
        pstImgInfo.u32Height = static_cast<unsigned int>(orig_frame.rows);

        // 设置图像的像素格式，假设是 BGR 格式
        pstImgInfo.ePixelFmt = ALG_FMT_RGB888;

        // 分配内存并将图像数据复制到 pstImgInfo.pImgData
        pstImgInfo.pImgData = malloc(orig_frame.total() * orig_frame.elemSize());
        if (pstImgInfo.pImgData != nullptr) {
            // 复制图像数据
            std::cerr << "Memory allocation success." << std::endl;
            std::memcpy(pstImgInfo.pImgData, orig_frame.data, orig_frame.total() * orig_frame.elemSize());
        } else {
            // 处理内存分配失败的情况
            std::cerr << "Memory allocation failed." << std::endl;
        }
        int targetSize = 20;
        ALG_DETECT_RET_S astDetectRet[targetSize];
        XXX_ALG_DetectProc(&pstImgInfo, 1, targetSize, astDetectRet, 0);
//        for (int i = 0; i < targetSize; i++) {
//            std::cout << "ID: " << astDetectRet[i].s32TargetID << std::endl;
//            std::cout << "Type：" << astDetectRet[i].u32TargetType << std::endl;
//            std::cout << "Conf: " << astDetectRet[i].f32conf << std::endl;
//            std::cout << "X: " << astDetectRet[i].stArea.u32Area_X << std::endl;
//            std::cout << "Y: " << astDetectRet[i].stArea.u32Area_Y << std::endl;
//            std::cout << "W: " << astDetectRet[i].stArea.u32Width << std::endl;
//            std::cout << "H: " << astDetectRet[i].stArea.u32Height << std::endl;
//            std::cout << "isMove: " << astDetectRet[i].isMove << std::endl;
//            std::cout << "-------------------------------------------" << std::endl;
//        }
//        std::cout << "==============================================" << std::endl;
    }
    XXX_ALG_DeInit();
}

int main() {
    tsetDetect();
//    tsetTrack();
//    tsetMoveDetect();
    return 1;
}

