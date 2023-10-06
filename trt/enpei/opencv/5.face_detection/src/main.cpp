#include "opencv2/opencv.hpp"
#include <iostream>

// 初始化模型
const std::string tensorflowConfigFile = "./weights/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./weights/opencv_face_detector_uint8.pb";
cv::dnn::Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);


// 检测并绘制矩形框
void detectDrawRect(cv::Mat &frame)
{
    // 获取图像的宽高
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;

    // 预处理，resize + swapRB + mean + scale
    cv::Mat inputBlob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);
    // 推理
    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    // 获取结果
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    // 遍历多个人脸结果
    for (int i = 0; i < detectionMat.rows; i++)
    {
        // 置信度
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > 0.2)
        {
            // 两点坐标
            int l = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int t = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int r = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int b = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            // 画框
            cv::rectangle(frame, cv::Point(l, t), cv::Point(r, b), cv::Scalar(0, 255, 0), 2);
        }
    }
}
// 图片测试
void imageTest()
{
    // 读取图片
    cv::Mat img = cv::imread("./media/test_face.jpg");
    // 推理
    detectDrawRect(img);
    // 显示
    cv::imshow("image test", img);
    // 保存
    cv::imwrite("./output/test_face_result.jpg", img);
    cv::waitKey(0);
}

// 实时视频流检测
void videoTest()
{
    // =========== 摄像头 ===========
    // 先读取camera或文件视频流并显示
    // cv::VideoCapture cap(2);
    // // 设置指定摄像头的分辨率
    // int width = 640;
    // int height = 480;
    // // 设置摄像头宽度和高度
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    // =========== 文件 ===========
    // 先读取camera或文件视频流并显示
    cv::VideoCapture cap("./media/video.mp4");
    // 获取视频流的宽高
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // 构造写入器
    // 写入MP4文件，参数分别是：文件名，编码格式，帧率，帧大小
    cv::VideoWriter writer("./output/record.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 25, cv::Size(width, height));

    if (!cap.isOpened())
    {
        std::cout << "Cannot open the video cam" << std::endl;
        // 退出
        exit(1);
    }
    cv::Mat frame;

    while (true)
    {
        if (!cap.read(frame))
        {
            std::cout << "Cannot read a frame from video stream" << std::endl;
            break;
        }

        // flip
        cv::flip(frame, frame, 1);
        // 推理
        detectDrawRect(frame);

        // 写入
        writer.write(frame);

        // cv::imshow("MyVideo", frame);
        // if (cv::waitKey(1) == 27)
        // {
        //     std::cout << "esc key is pressed by user" << std::endl;
        //     break;
        // }
    }
}

int main(int argc, char **argv)
{
    // 图片测试
    // imageTest();
    // 视频测试
    videoTest();
    return 0;
}