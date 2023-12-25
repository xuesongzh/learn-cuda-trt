// 导入opencv 库
#include <iostream>
#include <opencv2/opencv.hpp>
// 导入gflags 库
#include <gflags/gflags.h>
// 定义命令行参数
DEFINE_string(image, "./media/dog.jpg", "Input image");  // 图像路径

int main(int argc, char **argv) {
    // 解析命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // 读取图像，Mat是opencv中的图像数据结构，类似于numpy中的ndarray
    cv::Mat image = cv::imread(FLAGS_image);

    // 输出图片高度和宽度
    std::cout << "image height: " << image.rows << std::endl;
    std::cout << "image width: " << image.cols << std::endl;

    // 输出数据，以numpy和Python list格式输出
    // std::cout << cv::format(image, cv::Formatter::FMT_NUMPY) << std::endl;
    // std::cout << cv::format(image, cv::Formatter::FMT_PYTHON) << std::endl;

    // 判断图像是否读取成功，返回true表示失败
    if (image.empty()) {
        std::cout << "无法读取图片: " << FLAGS_image << std::endl;
        return 1;
    }
    // 创建一个灰度图的变量
    cv::Mat gray_image;
    // 转成灰度图
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    // 保存图像
    cv::imwrite("./output/gray_image.jpg", gray_image);

    // 显示图像
    cv::imshow("raw image", image);
    cv::imshow("gray image", gray_image);

    // 等待按键
    cv::waitKey(0);

    return 0;
}