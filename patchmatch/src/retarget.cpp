#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <retarget.h>
#include <patchmatch.h>

using namespace cv;
using namespace std;

void bidirection_similarity_vote()
{
    // TODO: generate reverse map
    // TODO: retarget dst
    return;
}

void bidirection_similarity_em_step(const cv::Mat &src, cv::Mat &dst, map_t *map)
{
    
    int dst_height = dst.rows;
    int dst_width = dst.cols;
    size_t map_size = dst_height * dst_width * sizeof(map_t);

    // allocate and init scratch nn maps
    map_t *prevMap = (map_t *) malloc(map_size);
    map_t *curMap = (map_t *) malloc(map_size);
    memcpy(prevMap, map, map_size);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // perform one iter of nn search
        // nn_search<sum_squared_diff>(dst, src, prevMap, curMap);
        nn_search(dst, src, prevMap, curMap);

        // swap two maps to reuse allocated space
        map_t *tmpMap = prevMap;
        prevMap = curMap;
        curMap = tmpMap;
    }

    bidirection_similarity_vote();

    free(prevMap);
    free(curMap);
}

void bidirection_similarity(const cv::Mat &original, const cv::Mat &src, cv::Mat &dst, 
    int dst_height, int dst_width)
{
    // allocate dst by resizing src
    resize(src, dst, Size(dst_width, dst_height));

    // create nn map for dst
    map_t *map = (map_t *) malloc(dst_height * dst_width * sizeof(map_t));
    // init_retarget_map<sum_squared_diff>(dst, original, map);
    init_retarget_map(dst, original, map);

    // start em iterations
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        bidirection_similarity_em_step(original, dst, map);
    }

    // clean up
    free(map);
}

void retarget(const cv::Mat &src, cv::Mat &dst, int dst_height, int dst_width) 
{
    int src_height = src.rows;
    int src_width = src.cols;

    int cur_height = src_height;
    int cur_width = src_width;

    int diff_height = abs(dst_width - src_width);
    int diff_width = abs(dst_height - src_width);

    int delta_height = max(src_height / 20, MIN_RETARGET_DELTA);
    int delta_width = max(src_width / 20, MIN_RETARGET_DELTA);

    Mat cur = src.clone();

    while ((diff_height > 0) || (diff_width > 0)) {
        int actual_delta_height = min(diff_height, delta_height);
        int actual_delta_width = min(diff_width, delta_width);
        
        int target_height = (src_height > dst_height) ? 
            cur_height - actual_delta_height :
            cur_height + actual_delta_height;
        int target_width = (src_width > dst_width) ? 
            cur_width - actual_delta_width :
            cur_width + actual_delta_width;

        bidirection_similarity(src, cur, dst, target_height, target_width);

        cur.release();  // deallocate cur to avoid memory leak
        cur = dst;
        cur_height = target_height;
        cur_width = target_width;

        diff_height -= actual_delta_height;
        diff_width -= actual_delta_width;
    }
}