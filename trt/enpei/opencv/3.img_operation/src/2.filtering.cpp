// opencv 图像滤波
#include <iostream>

#include "opencv2/opencv.hpp"

int main() {
    // 读取图片
    cv::Mat src = cv::imread("./media/dog.jpg");
    // 高斯模糊
    cv::Mat blur;
    // 三个参数分别是输入图像、输出图像、卷积核大小
    cv::GaussianBlur(src, blur, cv::Size(7, 7), 0);

    // 膨胀
    cv::Mat dilate;
    // 三个参数分别是输入图像、输出图像、卷积核大小
    cv::dilate(src, dilate, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    // 腐蚀
    cv::Mat erode;
    // 三个参数分别是输入图像、输出图像、卷积核大小
    cv::erode(src, erode, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    // 保存
    cv::imwrite("./output/2.blur.jpg", blur);
    cv::imwrite("./output/2.dilate.jpg", dilate);
    cv::imwrite("./output/2.erode.jpg", erode);
    return 0;
}