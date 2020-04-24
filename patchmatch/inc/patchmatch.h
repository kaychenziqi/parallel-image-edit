#ifndef PATCHMATCH_H_
#define PATCHMATCH_H_

#include <opencv2/opencv.hpp>

#define RANDOM_TRIALS 3
#define HALF_PATCH 3

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

// intialize nearest neighbor field
// template<distance_func_t distance_func>
void init_random_map(const cv::Mat &first, const cv::Mat &second, map_t *map);
// template<distance_func_t distance_func>
void init_retarget_map(const cv::Mat &dst, const cv::Mat &src, map_t *map);

// search nearest neighbor field
// template<distance_func_t distance_func>
void nn_search(const cv::Mat &first, const cv::Mat &second, map_t *curMap, map_t *newMap);


#endif