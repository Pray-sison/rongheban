//
// Created by Pray on 2023/8/10.
//

#include "feature.h"

Feature::Feature() {
}


/**
 * 将float映射到int8
 * @param f32 需要映射的float数据
 * @param zp 零点
 * @param scale 缩放因子
 * @return 映射后的int8
 */
int8_t Feature::qnt_f32_to_affine(float f32, int32_t zp, float scale) {
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
float Feature::deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float) qnt - (float) zp) * scale; }


// 用于模型加载
void Feature::dump_tensor_attr(rknn_tensor_attr *attr) {
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
}

unsigned char *Feature::load_data(FILE *fp, size_t ofst, size_t sz) {
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

unsigned char *Feature::load_model(const char *filename, int *model_size) {
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

int Feature::init(const char *filename) {

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
FEATURE Feature::feature_detect(const cv::Mat ori_frame) {
    int img_width = ori_frame.cols;
    int img_height = ori_frame.rows;
    cv::Mat resize_mat = image_preporcess(ori_frame);
    inputs[0].buf = (void *) resize_mat.data;
    rknn_inputs_set(ctx, 1, inputs);
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 1; i++) {
        outputs[i].want_float = 0;
    }
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    std::vector<float> out_scales;
    std::vector <int32_t> out_zps;
    for (int i = 0; i < 1; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    const int8_t *output_data = reinterpret_cast<const int8_t *>(outputs[0].buf);

    std::vector<float> fp32_output_data; // 用于存储转换后的 fp32 数据

    for (int i = 0; i < feature_length; ++i) {
        int8_t qnt_value = output_data[i];
        float fp32_value = Feature::deqnt_affine_to_f32(qnt_value, out_zps[0], out_scales[0]);
        fp32_output_data.push_back(fp32_value);
    }

    std::vector<float> normalized_data = normalize_data(fp32_output_data);
    // 将 normalized_data 的数据拷贝到 FEATURE 对象中
    FEATURE feature_vector;
    std::memcpy(feature_vector.data(), normalized_data.data(), feature_length * sizeof(float));

    // 打印归一化后的数据
    ret = rknn_outputs_release(ctx, 1, outputs);
    return feature_vector;
}

/**
 * 图像预处理，包括通道重拍，以及图像缩放
 * @param orig_frame 待遇处理图像
 * @return 处理后的图像
 */
cv::Mat Feature::image_preporcess(cv::Mat orig_frame) {
    cv::Mat det_frame;
    cv::cvtColor(orig_frame, det_frame, cv::COLOR_BGR2RGB);
    cv::resize(det_frame, det_frame, cv::Size(this->width, this->height));
    return det_frame;
}


std::vector<float> Feature::normalize_data(const std::vector<float>& input_data) {
    Eigen::Map<const Eigen::MatrixXf> x(input_data.data(), input_data.size(), 1);
    Eigen::VectorXf norms = x.colwise().norm();  // 计算L2范数
    std::vector<float> normalized_output(input_data.size());
    for (int i = 0; i < input_data.size(); ++i) {
        normalized_output[i] = input_data[i] / norms(0);
    }
    return normalized_output;
}





