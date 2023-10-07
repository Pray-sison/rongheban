#include "model.h"
#include <algorithm>

const float kRatio=0.5;
// 方便获取值
enum DETECTBOX_IDX {IDX_X = 0, IDX_Y, IDX_W, IDX_H };

// 将矩形转换为(centerx, centery, ratio, h)的形式
DETECTBOX DETECTION_ROW::to_xyah() const
{
	DETECTBOX ret = tlwh;
	ret(0,IDX_X) += (ret(0, IDX_W)*kRatio);
	ret(0, IDX_Y) += (ret(0, IDX_H)*kRatio);
	ret(0, IDX_W) /= ret(0, IDX_H);
	return ret;
}

// 将矩形转换为(x, y, xx, yy)的形式
DETECTBOX DETECTION_ROW::to_tlbr() const
{//(x,y,xx,yy)
	DETECTBOX ret = tlwh;
	ret(0, IDX_X) += ret(0, IDX_W);
	ret(0, IDX_Y) += ret(0, IDX_H);
	return ret;
}

