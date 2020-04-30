#ifndef UTIL_H_
#define UTIL_H_

#include <opencv2/opencv.hpp>

#ifndef DEBUG
#define DEBUG 0
#endif

#define N_CHANNELS 4

void mat_to_array(const cv::Mat &mat, float **arr_ptr);
void array_to_mat(float *arr, cv::Mat &mat, int ny, int nx, int nc);
void clone_array(float *arr, float **out_ptr, int ny, int nx);
void imwrite_array(std::string fname, float *arr, int ny, int nx, int nc);

#endif