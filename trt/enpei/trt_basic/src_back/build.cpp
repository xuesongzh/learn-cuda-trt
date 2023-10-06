/*
TensorRT build engine过程

1.创建builder
2.创建网络定义：builder --> network
3.配置参数：builder --> config
4.生成engine：builder --> engine (* network , * config)
5.序列化保存：engine --> serialize
6.释放资源：delete
*/

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>

#include <NvInfer.h>

// logger 用于管控打印的日志级别
// TRTLLogger继承自nvinfer1::ILogger
class TRTLLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 屏蔽info级别的日志
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

// 保存权重
void saveWeights(const std::string &filename, const float *data, int size)
{
    std::ofstream file(filename, std::ios::binary);
    assert(file.is_open() && "open file failed"); // 断言，如果file.is_open()为false，打印"open file failed"
    // reinterpret_cast用于类型转换，将size转为char类型的指针
    file.write(reinterpret_cast<const char *>(&size), sizeof(int));         // 写入权重大小
    file.write(reinterpret_cast<const char *>(data), size * sizeof(float)); // 写入权重
    file.close();
}
// 读取权重
std::vector<float> loadWeights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open() && "open file failed");
    int size;
    file.read(reinterpret_cast<char *>(&size), sizeof(int));                // 读取权重大小
    std::vector<float> data(size);                                          // 创建一个float类型的vector，大小为size
    file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float)); // 读取权重
    file.close();
    return data;
}

int main()
{
    //  =============== 1. 创建builder ===============
    TRTLLogger logger;
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);

    // ================ 2. 创建网络定义 =================
    // 1U << 0 = 1，bit shift 二进制移位，左移0位，即1（y左移x位，相当于y乘以2的x次方）
    auto explicitBact = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 调用createNetworkV2创建网络，参数explicitBact表示使用显式batch
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBact);

    // 定义网络
    // mlp多层感知机: input -> fc1 -> sigmod -> output

    // 创建一个input tensor，参数分别为：名字，数据类型，维度
    const int input_size = 3;
    nvinfer1::ITensor *input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, input_size, 1, 1});

    // 创建一个全连接层fc1
    // 权重和偏置
    const float *fc1_weight_data = new float[6]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    const float *fc1_bias_data = new float[2]{0.1, 0.5};

    // 保存为文件：也可以引用其他来源的权重，比如训练的模型
    saveWeights("./model/fc1.wts", fc1_weight_data, 6);
    saveWeights("./model/fc1.bias", fc1_bias_data, 2);

    // 重新在文件中读取权重
    auto fc1_weight_vec = loadWeights("./model/fc1.wts");
    auto fc1_bias_vec = loadWeights("./model/fc1.bias");

    const int output_size = 2;
    // 转为nvinfer1::Weights类型，参数分别为：数据类型，数据指针，数据大小
    nvinfer1::Weights fc1_weights{nvinfer1::DataType::kFLOAT, fc1_weight_vec.data(), fc1_weight_vec.size()};
    nvinfer1::Weights fc1_bias{nvinfer1::DataType::kFLOAT, fc1_bias_vec.data(), fc1_bias_vec.size()};
    // 创建全连接层，参数分别为：输入，输出大小，权重，偏置
    nvinfer1::IFullyConnectedLayer *fc1 = network->addFullyConnected(*input, output_size, fc1_weights, fc1_bias);

    // 添加激活层，参数为：输入，激活函数类型（sigmoid）
    nvinfer1::IActivationLayer *sigmod = network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 设置输出名字
    sigmod->getOutput(0)->setName("output");
    // 标记输出，没有被标记为输出的张量被认为是瞬时值，可以被构建者优化掉。
    network->markOutput(*sigmod->getOutput(0));

    // 设置运行时最大batch size
    builder->setMaxBatchSize(1);

    // ================= 3. 配置参数 =================
    // 添加配置，用来指定TensorRT应该如何优化模型
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // 设置最大工作空间大小（指定TensorRT可以使用的GPU内存大小），单位为字节
    config->setMaxWorkspaceSize(1 << 28); // 256M

    // ================= 4. 构建engine =================
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        std::cout << "Failed to create engine." << std::endl;
        return -1;
    }
    // ================= 5. 序列化engine =================
    // 序列化：将对象转换为字节序列的过程，反序列化：将字节序列转换为对象的过程
    nvinfer1::IHostMemory *serialized_engine = engine->serialize();
    // 存入文件
    std::ofstream file("./model/mlp.engine", std::ios::binary);
    assert(file.is_open() && "open file failed");
    file.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    // ================= 6. 释放资源 =================
    // 理论上，前面申请的资源都应该在这里释放，但是这里只是为了演示，所以只释放了部分资源

    file.close();             // 关闭文件
    delete serialized_engine; // 释放序列化的engine
    delete engine;            // 释放engine
    delete config;            // 释放config
    delete network;           // 释放network
    delete builder;           // 释放builder

    std::cout << "engine文件构建成功！！！" << std::endl;

    return 0;
}