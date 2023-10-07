#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"


// 每个矩形（rect）的数据结构。
// tlwh: 左上角坐标和宽度高度。
// confidence: 检测置信度。
// feature: 矩形的128维特征。
class DETECTION_ROW
{
public:
    DETECTBOX tlwh;
    int class_id;
    float conf;
    FEATURE feature;
    DETECTBOX to_xyah() const;
    DETECTBOX to_tlbr() const;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;



#endif // MODEL_H
