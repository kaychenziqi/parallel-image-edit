#include <opencv2/opencv.hpp>
#include "util.h"

using namespace std;
using namespace cv;

void mat_to_array(const cv::Mat &mat, float **arr_ptr)
{
    int ny = mat.rows;
    int nx = mat.cols;
    int nc = mat.channels();

    float *arr = (float *) malloc(ny * nx * N_CHANNELS * sizeof(float));
    
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int idx = y * nx + x;
            Vec3f pixel = mat.at<Vec3f>(y, x);
            for (int c = 0; c < nc; c++) {
                arr[idx * N_CHANNELS + c] = pixel[c];
            }
        }
    }
    *arr_ptr = arr;
}

void array_to_mat(float *arr, cv::Mat &mat, int ny, int nx, int nc)
{
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            float *p = arr + (y * nx + x) * N_CHANNELS;
            for (int c = 0; c < nc; c++) {
                mat.at<Vec3f>(y, x)[c] = p[c];
            }
        }
    }
}

void clone_array(float *arr, float **out_ptr, int ny, int nx)
{
    size_t size = ny * nx * N_CHANNELS * sizeof(float);
    float *new_arr = (float *) malloc(size);
    memcpy(new_arr, arr, size);
    *out_ptr = new_arr;
}

void imwrite_array(string fname, float *arr, int ny, int nx, int nc)
{
    Mat dst(ny, nx, CV_32FC3);
    array_to_mat(arr, dst, ny, nx, nc);
    Mat dst2;
    dst.convertTo(dst2, CV_8UC3);
    imwrite(fname, dst2);
}