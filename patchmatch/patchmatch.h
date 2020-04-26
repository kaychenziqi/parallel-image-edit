#ifndef PATCHMATCH_H_
#define PATCHMATCH_H_

#include <opencv2/opencv.hpp>

#define NUM_ITERATIONS 10
#define MAX_SEARCH_RADIUS 256
#define HALF_PATCH 1
#define SAVE_ITER_OUTPUT 1

// map entry type
typedef struct {
    int x;
    int y;
    float dist;
} map_t;

// patch distance function type
typedef float(*distance_func_t)(const cv::Vec3b &, const cv::Vec3b &);

// distance functions
float sum_squared_diff(const cv::Vec3b &fpixel, const cv::Vec3b &spixel);
float sum_absolute_diff(const cv::Vec3b &fpixel, const cv::Vec3b &spixel);
float patch_distance(const cv::Mat &first, const cv::Mat &second, 
    int fx, int fy, int sx, int sy, int half_patch = 1);

// intialize nearest neighbor field
void init_random_map(const cv::Mat &first, const cv::Mat &second, map_t *map, 
    int half_patch = 1);
void init_retarget_map(const cv::Mat &dst, const cv::Mat &src, map_t *map, 
    int half_patch = 1);

// nearest neighbor field
void nn_search(const cv::Mat &first, const cv::Mat &second, map_t *curMap, 
    int half_patch = 1);
void nn_map(const cv::Mat &src, cv::Mat &dst, map_t *map);
void nn_map_average(const cv::Mat &src, cv::Mat &dst, map_t *map, 
    int half_patch = 1);

void patchmatch(const cv::Mat &src, cv::Mat &dst, int half_patch = 1);

#endif