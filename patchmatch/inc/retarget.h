#ifndef RETARGET_H_
#define RETARGET_H_

#include <opencv2/opencv.hpp>

#define MIN_RETARGET_DELTA 10

void retarget(const cv::Mat &src, cv::Mat &dst, int dst_height, int dst_width);

#endif