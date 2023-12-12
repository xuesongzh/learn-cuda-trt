// 预处理

#include "preprocess.h"

#include "utils/logging.h"

void imgPreprocess(const cv::Mat &img, cv::Mat &img_resized, uint32_t width, uint32_t height) {
    // img has to be 3 channels
    if (img.channels() != 3) {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    cv::Mat img_rgb;
    // bggr to rgb
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    // resize img
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
}
