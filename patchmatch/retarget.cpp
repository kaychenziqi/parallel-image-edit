#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <opencv2/opencv.hpp>

#include "patchmatch.h"
#include "retarget.h"

using namespace cv;
using namespace std;

void generate_reverse_map(map_t *map, int m_height, int m_width,
    map_t *revMap, int r_height, int r_width)
{
    for (int y = 0; y < r_height; y++) {
        for (int x = 0; x < r_width; x++) {
            revMap[y * r_width + x].dist = FLT_MAX;
        }
    }

    for (int y = 0; y < m_height; y++) {
        for (int x = 0; x < m_width; x++) {
            int idx = y * m_width + x;
            int ridx = map[idx].y * r_width + map[idx].x;

            if (map[idx].dist < revMap[ridx].dist) {
                revMap[ridx].x = x;
                revMap[ridx].y = y;
                revMap[ridx].dist = map[idx].dist;
            }
        }
    }
}

void bidirection_similarity_vote(const cv::Mat &src, cv::Mat &dst, map_t *map)
{
    int s_height = src.rows;
    int s_width = src.cols;
    int d_height = dst.rows;
    int d_width = dst.cols;

    float s_total = s_width * s_height;
    float d_total = d_width * d_height;
    int num_cohere = (2 * HALF_PATCH + 1) * (2 * HALF_PATCH + 1);
    float denom_offset = num_cohere / d_total;

    // generate reverse map
    map_t *revMap = (map_t *) malloc(s_height * s_width * sizeof(map_t));
    generate_reverse_map(map, d_height, d_width, revMap, s_height, s_width);

    // compute measure for each pixel in dst
    for (int dy = 0; dy < d_height; dy++) {
        for (int dx = 0; dx < d_width; dx++) {

            // sums for coherence and completeness
            float cohere_sum[3], complete_sum[3];
            cohere_sum[0]   = cohere_sum[1]   = cohere_sum[2]   = 0;
            complete_sum[0] = complete_sum[1] = complete_sum[2] = 0;
            int num_complete = 0;

            // loop over the patch
            for (int j = -HALF_PATCH; j <= HALF_PATCH; j++) {
                for (int i = -HALF_PATCH; i <= HALF_PATCH; i++) {
                    // dst pixel
                    int d_py = max(0, min(d_height - 1, dy + j));
                    int d_px = max(0, min(d_width - 1, dx + i));
                    int d_pidx = d_py * d_width + d_px;

                    // match for (d_py, d_px)
                    int s_py = map[d_pidx].y;
                    int s_px = map[d_pidx].x;
                    int s_pidx = s_py * s_width + s_px;

                    // match for (dy, dx)
                    s_py -= j;
                    s_px -= i;

                    cohere_sum[0] += src.at<Vec3b>(s_py, s_px)[0];
                    cohere_sum[1] += src.at<Vec3b>(s_py, s_px)[1];
                    cohere_sum[2] += src.at<Vec3b>(s_py, s_px)[2];

                    if ((revMap[s_pidx].x == dx) && (revMap[s_pidx].y == dy)) {
                        complete_sum[0] += src.at<Vec3b>(s_py, s_px)[0];
                        complete_sum[1] += src.at<Vec3b>(s_py, s_px)[1];
                        complete_sum[2] += src.at<Vec3b>(s_py, s_px)[2];
                        num_complete++;
                    }
                }
            }

            float denom = (num_complete / s_total) + denom_offset;
            complete_sum[0] /= s_total;
            complete_sum[1] /= s_total;
            complete_sum[2] /= s_total;

            cohere_sum[0] /= d_total;
            cohere_sum[1] /= d_total;
            cohere_sum[2] /= d_total;

            dst.at<Vec3b>(dy, dx)[0] = (int) floor((complete_sum[0] + cohere_sum[0]) / denom);
            dst.at<Vec3b>(dy, dx)[1] = (int) floor((complete_sum[1] + cohere_sum[1]) / denom);
            dst.at<Vec3b>(dy, dx)[2] = (int) floor((complete_sum[2] + cohere_sum[2]) / denom);
        }
    }

    free(revMap);
    return;
}

void bidirection_similarity_em_step(const cv::Mat &src, cv::Mat &dst, map_t *map)
{
    
    int dst_height = dst.rows;
    int dst_width = dst.cols;
    size_t map_size = dst_height * dst_width * sizeof(map_t);

    // allocate and init scratch nn maps
    map_t *curMap = (map_t *) malloc(map_size);
    memcpy(curMap, map, map_size);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cout << "BDS iteration " << i << endl;
        // perform one iter of nn search
        // nn_search<sum_squared_diff>(dst, src, prevMap, curMap);
        nn_search(dst, src, curMap);
    }

    bidirection_similarity_vote(src, dst, map);

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
        cout << "EM iteration " << i << endl;
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
        cur_height = cur.rows;
        cur_width = cur.cols;

        diff_height -= actual_delta_height;
        diff_width -= actual_delta_width;

        if (SAVE_ITER_OUTPUT) {
            char fname[64];
            sprintf(fname, "scratch/retarget-%i-%i.jpg", cur_width, cur_height);
            imwrite(fname, cur);
        }
    }
}