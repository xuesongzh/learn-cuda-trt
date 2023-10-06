// 绘制文字和图形
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

int main()
{
    // 创建一个黑色图像，参数分别是图像大小、图像类型，CV_8UC3表示8位无符号整数，3通道
    cv::Mat image = cv::Mat::zeros(cv::Size(600, 600), CV_8UC3);
    
    // 绘制直线，参数分别是图像、起点、终点、颜色、线宽、线型
    cv::line(image, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    // 绘制矩形，参数分别是图像、左上角、右下角、颜色、线宽、线型
    cv::rectangle(image, cv::Point(50, 50), cv::Point(350, 250), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    // 绘制圆形，参数分别是图像、圆心、半径、颜色、线宽、线型
    cv::circle(image, cv::Point(200, 150), 100, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    // 实心
    cv::circle(image, cv::Point(200, 150), 50, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);

    // // ================== 多边形 ==================
    // cv::Point points[2][4]; // 定义两个多边形的顶点数组
    // // 第一个多边形的顶点
    // points[0][0] = cv::Point(100, 115);
    // points[0][1] = cv::Point(255, 135);
    // points[0][2] = cv::Point(140, 365);
    // points[0][3] = cv::Point(100, 300);
    // // 第二个多边形的顶点
    // points[1][0] = cv::Point(300, 315);
    // points[1][1] = cv::Point(555, 335);
    // points[1][2] = cv::Point(340, 565);
    // points[1][3] = cv::Point(300, 500);
    // // ppt[] 要同时添加两个多边形顶点数组的地址）
    // const cv::Point *pts_v[] = {points[0], points[1]};
    // // npts_v[]要定义每个多边形的顶点数
    // int npts_v[] = {4, 4};
    // // 绘制多边形，参数分别是图像、顶点数组、顶点数、曲线数量、是否闭合、颜色、线宽、线型
    // cv::polylines(image, pts_v, npts_v, 2, true, cv::Scalar(255, 0, 255), 2, 8, 0);

    // ================== 使用vector绘制多边形 ==================
    std::vector<cv::Point> points_v;
    // 随机生成5个点
    for (int i = 0; i < 5; i++)
    {
        points_v.push_back(cv::Point(rand() % 600, rand() % 600));
    }
    // 绘制多边形，参数分别是图像、顶点容器、是否闭合、颜色、线宽、线型
    cv::polylines(image, points_v, true, cv::Scalar(255, 0, 0), 2, 8, 0);

    // ================== 绘制文字 ==================
    // 参数分别是图像、文字、文字位置、字体、字体大小、颜色、线宽、线型
    cv::putText(image, "Hello World!", cv::Point(400, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2, 8, 0);

    

    // 保存
    cv::imwrite("./output/4.drawing.jpg", image);
    return 0;
}