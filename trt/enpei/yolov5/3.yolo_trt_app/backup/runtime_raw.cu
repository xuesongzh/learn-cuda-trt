#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "common.h"
#include "buffers.h"
#include "utils/preprocess.h"
#include "utils/postprocess.h"
#include "utils/types.h"
#include "streamer/streamer.hpp"

#include <fstream>
#include "task/border_cross.h"
#include "task/gather.h"

// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}

// 从文件中恢复多边形定点
void readPoints(std::string filename, Polygon &g_ploygon)
{
    std::ifstream file(filename);
    std::string str;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        std::string x, y;
        std::getline(ss, x, ',');
        std::getline(ss, y, ',');
        g_ploygon.push_back({std::stoi(x), std::stoi(y)});
    }
}
void blender_overlay(int x, int y, int radius, cv::Mat &image, float alpha, int height, int width)
{
    // initial
    int rect_l = x - radius;
    int rect_t = y - radius;
    int rect_w = radius * 2;
    int rect_h = radius * 2;

    int point_x = radius;
    int point_y = radius;

    // check if out of range
    if (x + radius > width)
    {
        rect_w = radius + (width - x);
    }
    if (y + radius > height)
    {
        rect_h = radius + (height - y);
    }
    if (x - radius < 0)
    {
        rect_l = 0;
        rect_w = radius + x;
        point_x = x;
    }
    if (y - radius < 0)
    {
        rect_t = 0;
        rect_h = radius + y;
        point_y = y;
    }
    // get roi
    cv::Mat roi = image(cv::Rect(rect_l, rect_t, rect_w, rect_h));
    cv::Mat color;
    roi.copyTo(color);
    // draw circle
    cv::circle(color, cv::Point(point_x, point_y), radius, cv::Scalar(255, 0, 255), -1);
    // blend
    cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);
}

int main(int argc, char **argv)
{
    Polygon g_ploygon;
    // 检查多边形定点配置文件是否存在
    std::ifstream infile("./config/polygon.txt");
    if (infile)
    {
        std::cout << "检测到多边形顶点配置文件，从文件中恢复多边形定点" << std::endl;
        readPoints("./config/polygon.txt", g_ploygon);
    }
    else
    {
        std::cout << "未检测到多边形顶点配置文件" << std::endl;
        return -1;
    }
    if (argc < 5)
    {
        std::cerr << "用法: " << argv[0] << " <engine_file> <input_path_path> <mode> <bitrate> <dist_threshold>" << std::endl;
        return -1;
    }

    auto engine_file = argv[1];                // 模型文件
    std::string input_video_path = argv[2];    // 输入视频文件
    auto mode = std::stoi(argv[3]);            // 模式
    auto bitrate = std::stoi(argv[4]);         // 码率
    float dist_threshold = std::stof(argv[5]); // 距离阈值

    // ========= 1. 创建推理运行时runtime =========
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime)
    {
        std::cout << "runtime create failed" << std::endl;
        return -1;
    }
    // ======== 2. 反序列化生成engine =========
    // 加载模型文件
    auto plan = load_engine_file(engine_file);
    // 反序列化生成engine
    auto mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!mEngine)
    {
        return -1;
    }

    // ======== 3. 创建执行上下文context =========
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cout << "context create failed" << std::endl;
        return -1;
    }

    // ========== 4. 创建输入输出缓冲区 =========
    samplesCommon::BufferManager buffers(mEngine);

    // 如果input_video_path是rtsp，则读取rtsp流
    cv::VideoCapture cap;
    if (input_video_path == "rtsp")
    {
        auto rtsp = "rtsp://localhost:8554/live1.sdp";
        std::cout << "当前使用的是RTSP流" << std::endl;
        cap = cv::VideoCapture(rtsp, cv::CAP_FFMPEG);
    }
    else
    {
        std::cout << "当前使用的是视频文件" << std::endl;
        cap = cv::VideoCapture(input_video_path);
    }

    // 获取画面尺寸
    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // 获取帧率
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "width: " << frameSize.width << " height: " << frameSize.height << " fps: " << video_fps << std::endl;
    // 实例化推流器
    streamer::Streamer streamer;
    streamer::StreamerConfig streamer_config(frameSize.width, frameSize.height,
                                             frameSize.width, frameSize.height,
                                             video_fps, bitrate, "main", "rtmp://localhost/live/mystream");
    streamer.init(streamer_config);

    // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
    // cv::VideoWriter writer("./output/record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), video_fps, frameSize);

    cv::Mat frame;
    int frame_index{0};

    int img_size = frameSize.width * frameSize.height;
    cuda_preprocess_init(img_size); // 申请cuda内存

    while (cap.isOpened())
    {

        cap >> frame;
        // 统计运行时间
        auto start = std::chrono::high_resolution_clock::now();

        if (frame.empty())
        {
            std::cout << "文件处理完毕" << std::endl;
            break;
        }

        frame_index++;

        // 选择预处理方式
        if (mode == 0)
        {
            // 使用CPU做letterbox、归一化、BGR2RGB、NHWC to NCHW
            process_input_cpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
        else if (mode == 1)
        {
            // 使用CPU做letterbox，GPU做归一化、BGR2RGB、NHWC to NCHW
            process_input_cv_affine(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }
        else if (mode == 2)
        {
            // 使用cuda预处理所有步骤
            process_input_gpu(frame, (float *)buffers.getDeviceBuffer(kInputTensorName));
        }

        // ========== 5. 执行推理 =========
        context->executeV2(buffers.getDeviceBindings().data());
        // 拷贝回host
        buffers.copyOutputToHost();

        // 从buffer manager中获取模型输出
        int32_t *num_det = (int32_t *)buffers.getHostBuffer(kOutNumDet); // 检测到的目标个数
        int32_t *cls = (int32_t *)buffers.getHostBuffer(kOutDetCls);     // 检测到的目标类别
        float *conf = (float *)buffers.getHostBuffer(kOutDetScores);     // 检测到的目标置信度
        float *bbox = (float *)buffers.getHostBuffer(kOutDetBBoxes);     // 检测到的目标框
        // 执行nms（非极大值抑制），得到最后的检测框
        std::vector<Detection> bboxs;
        yolo_nms(bboxs, num_det, cls, conf, bbox, kConfThresh, kNmsThresh);

        // 结束时间
        auto end = std::chrono::high_resolution_clock::now();
        // microseconds 微秒，milliseconds 毫秒，seconds 秒，1微妙=0.001毫秒 = 0.000001秒
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f;
        auto time_str = std::to_string(elapsed) + "ms";
        auto fps = 1000.0f / elapsed;
        auto fps_str = std::to_string(fps) + "fps";

        // 记录所有的检测框中心点
        std::vector<Point> all_points;
        // 遍历检测结果
        for (size_t j = 0; j < bboxs.size(); j++)
        {

            cv::Rect r = get_rect(frame, bboxs[j].bbox);
            // 获取检测框中心点
            Point p_center = {r.x + int(r.width / 2), r.y + int(r.height / 2)};
            // 筛选labelid为0的检测框
            if (bboxs[j].class_id == 0)
            {
                all_points.push_back(p_center);
            }
            // 检测框中心点是否在多边形内，在则画红框，不在则画绿框
            if (isInside(g_ploygon, p_center))
            {
                cv::rectangle(frame, r, cv::Scalar(0x00, 0x00, 0xFF), 2);
            }
            else
            {
                cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            }
            // 绘制labelid
            // cv::putText(frame, std::to_string((int)bboxs[j].class_id), cv::Point(r.x, r.y - 10), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }
        // 获取聚集点
        auto gather_points = gather_rule(all_points, dist_threshold);
        for (size_t i = 0; i < gather_points.size(); i++)
        {

            if (gather_points[i].size() < 3)
                continue;
            for (size_t j = 0; j < gather_points[i].size(); j++)
            {
                // std::cout << "聚集点：" << gather_points[i][j].x << "," << gather_points[i][j].y << std::endl;
                // 绘制聚集点
                blender_overlay(gather_points[i][j].x, gather_points[i][j].y, 80, frame, 0.3, frameSize.height, frameSize.width);
            }
        }

        cv::putText(frame, time_str, cv::Point(50, 50), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, fps_str, cv::Point(50, 100), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255, 255, 255), 2);

        // 绘制多边形
        std::vector<cv::Point> polygon;
        for (size_t i = 0; i < g_ploygon.size(); i++)
        {
            polygon.push_back(cv::Point(g_ploygon[i].x, g_ploygon[i].y));
        }
        cv::polylines(frame, polygon, true, cv::Scalar(0, 0, 255), 2);

        // cv::imshow("frame", frame);
        // 写入视频文件
        // writer.write(frame);

        // 推流
        streamer.stream_frame(frame.data);

        // if (cv::waitKey(1) == 27)
        //     break;
    }
    // ========== 6. 释放资源 =========
    // 因为使用了unique_ptr，所以不需要手动释放

    return 0;
}
