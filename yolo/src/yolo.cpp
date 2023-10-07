#include "yolo.h"

YOLOv5::YOLOv5() {
}

float YOLOv5::CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                       float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}


int YOLOv5::nms_run(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
                int filterId) {
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > this->nms) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

int YOLOv5::quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

/**
 * 将float映射到int8
 * @param f32 需要映射的float数据
 * @param zp 零点
 * @param scale 缩放因子
 * @return 映射后的int8
 */
int8_t YOLOv5::qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t) __clip(dst_val, -128, 127);
    return res;
}

/**
* 将int8映射到float
* @param qnt 需要映射的int8数据
* @param zp 零点
* @param scale 缩放因子
* @return 映射后的float
*/
float YOLOv5::deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float) qnt - (float) zp) * scale; }

int YOLOv5::process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                    std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                    int32_t zp, float scale) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid(this->conf);
    int8_t thres_i8 = qnt_f32_to_affine(thres, zp, scale);
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float) stride;
                    box_y = (box_y + i) * (float) stride;
                    box_w = box_w * box_w * (float) anchor[a * 2];
                    box_h = box_h * box_h * (float) anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < obj_class_num; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale)) *
                                           sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

int YOLOv5::post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                         float scale_w, float scale_h, std::vector <int32_t> &qnt_zps,
                         std::vector<float> &qnt_scales, detect_result_group_t *group) {
    memset(group, 0, sizeof(detect_result_group_t));
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    validCount0 = process(input0, (int *) anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes,
                          objProbs,
                          classId, qnt_zps[0], qnt_scales[0]);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = process(input1, (int *) anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes,
                          objProbs,
                          classId, qnt_zps[1], qnt_scales[1]);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = process(input2, (int *) anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes,
                          objProbs,
                          classId, qnt_zps[2], qnt_scales[2]);

    int validCount = validCount0 + validCount1 + validCount2;
    // no object detect
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c: class_set) {
        nms_run(validCount, filterBoxes, classId, indexArray, c);
    }

    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= this->max_num) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        group->results[last_count].box.left = (int) (clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top = (int) (clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right = (int) (clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int) (clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].conf = obj_conf;
        group->results[last_count].class_id = id;
        last_count++;
    }
    group->count = last_count;
    return 0;
}

// 用于模型加载
void YOLOv5::dump_tensor_attr(rknn_tensor_attr *attr) {
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i) {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
           "type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
           attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
    obj_class_num = attr->dims[1] / 3 - 5;
    printf("obj_class_num: %d\n", obj_class_num);
    prop_box_size = 5 + obj_class_num;

}

unsigned char *YOLOv5::load_data(FILE *fp, size_t ofst, size_t sz) {
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *) malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

unsigned char *YOLOv5::load_model(const char *filename, int *model_size) {
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int YOLOv5::init(const char *filename, float conf, float nms) {
    this->conf = conf;
    this->nms = nms;
    std::cout << "Loading mode......" << std::endl;
    unsigned char *model_data = load_model(filename, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        std::cout << "rknn_init error ret=" << ret << std::endl;
        return -1;
    }
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        std::cout << "rknn_init error ret=" << ret << std::endl;
        return -1;
    }
    std::cout << "sdk version: " << version.api_version << "driver version: " << version.drv_version << std::endl;
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        std::cout << "rknn_init error ret=" << ret << std::endl;
        return -1;
    }
    std::cout << "model input num: " << io_num.n_input << "output num: " << io_num.n_output << std::endl;

    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            std::cout << "rknn_init error ret=" << ret << std::endl;
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        std::cout << "model is NCHW input fmt" << std::endl;
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    } else {
        std::cout << "model is NHWC input fmt" << std::endl;
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }


    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    std::cout << "model input height=" << height << ", " << "weight=" << width << ", " << "channel=" << channel
              << std::endl;
    return 0;
}

// 模型推理
detect_result_group_t YOLOv5::yolo_detect(const cv::Mat ori_frame) {
    int img_width = ori_frame.cols;
    int img_height = ori_frame.rows;
    cv::Mat resize_mat = image_preporcess(ori_frame);
    inputs[0].buf = (void *) resize_mat.data;
    rknn_inputs_set(ctx, 1, inputs);
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; i++) {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, 3, outputs, NULL);

    float scale_w = (float) width / img_width;
    float scale_h = (float) height / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector <int32_t> out_zps;
    for (int i = 0; i < 3; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    post_process((int8_t *) outputs[0].buf, (int8_t *) outputs[1].buf, (int8_t *) outputs[2].buf, height, width,
                 scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    ret = rknn_outputs_release(ctx, 3, outputs);
    return detect_result_group;
}

/**
 * 图像预处理，包括通道重拍，以及图像缩放
 * @param orig_frame 待遇处理图像
 * @return 处理后的图像
 */
cv::Mat YOLOv5::image_preporcess(cv::Mat orig_frame) {
    cv::Mat det_frame;
    cv::cvtColor(orig_frame, det_frame, cv::COLOR_BGR2RGB);
    cv::resize(det_frame, det_frame, cv::Size(width, height));
    return det_frame;
}


/**
 * ------------yolo2dtrack格式转换，转换后输入到跟踪器中----------------
 * @param detect_result_group 检测结果
 * @param detections 转换后的检测结果
 */
//void YOLOv5::format_conversion(const cv::Mat& ori_image, detect_result_group_t detect_result_group, DETECTIONS &detections, Feature featureModel){
//    for (int i = 0; i < detect_result_group.count; i++) {
//        detect_result_t *det_result = &(detect_result_group.results[i]);
//        int xl = det_result->box.left;
//        int yt = det_result->box.top;
//        int w = det_result->box.right - det_result->box.left;
//        int h = det_result->box.bottom - det_result->box.top;
//        // 确保截取区域不超出图像范围
//        cv::Rect crop_region(xl, yt, w, h);
//        crop_region &= cv::Rect(0, 0, ori_image.cols, ori_image.rows);
//        // 截取图像
//        cv::Mat cropped_image = ori_image(crop_region).clone();
//        DETECTION_ROW tmpRow;
//        tmpRow.class_id = det_result->class_id;
//        tmpRow.tlwh = DETECTBOX(xl, yt, w, h);//DETECTBOX(x, y, w, h);
//        tmpRow.feature = featureModel.feature_detect(cropped_image);
//        tmpRow.conf = det_result->conf * 100;
//        detections.push_back(tmpRow);
//    }
//}



/**
 * ------------yolo2dtrack格式转换，转换后输入到跟踪器中----------------
 * @param detect_result_group 检测结果
 * @param detections 转换后的检测结果
 */
void YOLOv5::format_conversion(detect_result_group_t detect_result_group, DETECTIONS &detections){
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        int xl = det_result->box.left;
        int yt = det_result->box.top;
        int w = det_result->box.right - det_result->box.left;
        int h = det_result->box.bottom - det_result->box.top;
        // 确保截取区域不超出图像范围
        cv::Rect crop_region(xl, yt, w, h);
        DETECTION_ROW tmpRow;
        tmpRow.class_id = det_result->class_id;
        tmpRow.tlwh = DETECTBOX(xl, yt, w, h);//DETECTBOX(x, y, w, h);
        tmpRow.conf = det_result->conf * 100;
        detections.push_back(tmpRow);
    }
}










