// 导入相关库
#include <opencv2/opencv.hpp>

#include "engine/engine.h"
#include "process/postprocess.h"
#include "process/preprocess.h"
#include "utils/logging.h"

int main(int argc, char **argv) {
    // 模型文件路径
    const char *model_file = argv[1];
    // 图片文件路径
    const char *img_file = argv[2];

    // ==================== 1. 加载引擎 ====================
    // 使用shared_ptr智能指针创建引擎，类型是接口中定义的NNEngine，但是实际上是RKEngine
    std::shared_ptr<NNEngine> engine = CreateRKNNEngine();

    // 加载模型文件、初始化rknn context、获取rknn版本信息、获取输入输出张量的信息
    engine->LoadModelFile(model_file);

    // 获取输入输出张量的形状
    auto input_attrs = engine->GetInputShapes();
    auto output_attrs = engine->GetOutputShapes();

    NN_LOG_INFO("processing input data");

    uint32_t height =
        input_attrs[0].dims[1];  // 根据输入属性查询输入张量的高度（参考：dims=[1, 224, 224, 3]）
    uint32_t width = input_attrs[0].dims[2];  // 宽度

    // ==================== 2. 读取、预处理图片 ====================
    // 加载图片
    cv::Mat img = cv::imread(img_file);
    // 预处理 bgr->rgb, resize
    cv::Mat img_resized;
    imgPreprocess(img, img_resized, width, height);

    // ==================== 3. 设置输入、输出属性 ====================
    // 输入数据
    tensor_data_s input;
    input.attr.n_dims = 4;
    input.attr.dims[0] = 1;
    input.attr.dims[1] = height;
    input.attr.dims[2] = width;
    input.attr.dims[3] = 3;
    input.attr.size = 3 * height * width;
    input.attr.type = NN_TENSOR_UINT8;
    input.attr.layout = NN_TENSOR_NHWC;
    input.data = malloc(input.attr.size * sizeof(uint8_t));
    // 将图片数据拷贝到input.data中
    memcpy(input.data, img_resized.data, input.attr.size);
    // 将input追加到inputs中
    std::vector<tensor_data_s> inputs;
    inputs.push_back(input);
    NN_LOG_DEBUG("done preprocessing input data");

    // 输出数据
    std::vector<tensor_data_s> outputs;
    tensor_data_s output;
    output.data = malloc(output_attrs[0].n_elems * sizeof(float));
    outputs.push_back(output);

    // ==================== 4. 推理 ====================
    NN_LOG_INFO("running...");
    engine->Run(inputs, outputs, true);

    // ==================== 5. 后处理 ====================
    uint32_t MaxClass[5];
    float fMaxProb[5];
    float *buffer = (float *)output.data;
    uint32_t sz = outputs[0].attr.size / 4;

    NN_LOG_DEBUG("output size: %d", sz);

    get_top(buffer, fMaxProb, MaxClass, sz, 5);

    printf(" --- Top5 ---\n");
    for (int i = 0; i < 5; i++) {
        printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
    }
    return 0;
}
