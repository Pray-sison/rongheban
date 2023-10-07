#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"

class DETECTION_ROW
{
public:
    DETECTBOX tlwh;
    int class_id;
    float conf;
    DETECTBOX to_xyah() const;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;

#endif // MODEL_H
