#include <opencv2/opencv.hpp>
#include <patchmatch.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

inline float square(float x) { return x * x; }

float sum_squared_diff(const cv::Vec3b &fpixel, const cv::Vec3b &spixel)
{
    float dist = sqrt(
        square(fpixel[0] - spixel[0]) +
        square(fpixel[1] - spixel[1]) +
        square(fpixel[2] - spixel[2])
    );
    return dist;
}

float sum_absolute_diff(const cv::Vec3b &fpixel, const cv::Vec3b &spixel)
{
    float dist = sqrt(
        abs(fpixel[0] - spixel[0]) +
        abs(fpixel[1] - spixel[1]) +
        abs(fpixel[2] - spixel[2])
    );
    return dist;
}

// template<distance_func_t distance_func>
float patch_distance(const cv::Mat &first, const cv::Mat &second, 
    int fx, int fy, int sx, int sy, int half_patch)
{
    float dist = 0;
    for (int j = -half_patch; j <= half_patch; j++) {
        for (int i = -half_patch; i <= half_patch; i++) {
            int fx1 = min(first.cols - 1, max(0, fx + i));
            int fy1 = min(first.rows - 1, max(0, fy + i));
            Vec3b fpixel = first.at<Vec3b>(fy1, fx1);

            int sx1 = min(second.cols - 1, max(0, sx + i));
            int sy1 = min(second.rows - 1, max(0, sy + i));
            Vec3b spixel = second.at<Vec3b>(sy1, sx1);

            // dist += distance_func(fpixel, spixel);
            dist += sum_squared_diff(fpixel, spixel);
        }
    }
    return dist;
}


void pick_random_pixel(int radius, int height, int width, 
    int sx, int sy, int *rx_ptr, int *ry_ptr)
{
    int xmin = max(sx - radius, 0);
    int xmax = min(sx + radius, width);
    int ymin = max(sy - radius, 0);
    int ymax = min(sy + radius, height);

    int xlen = xmax - xmin;
    int ylen = ymax - ymin;

    *rx_ptr = (random() % xlen) + xmin;
    *ry_ptr = (random() % ylen) + ymin;
}

// For each pixel in first, random assign a nn pixel in second
// template<distance_func_t distance_func>
void init_random_map(const cv::Mat &first, const cv::Mat &second, map_t *map, int half_patch)
{
    int d_height = first.rows;
    int d_width = first.cols;
    int s_height = second.rows;
    int s_width = second.cols;

    for (int y = 0; y < d_height; y++ ) {
        for (int x = 0; x < d_width; x++ ) {
            int rx = random() % s_width;
            int ry = random() % s_height;
            int idx = y * d_width + x;

            map[idx].x = rx;
            map[idx].y = ry;
            map[idx].dist = patch_distance(first, second, x, y, rx, ry, half_patch);
        }
    }
}

// For each pixel in dst, assign a nn pixel in src
// template<distance_func_t distance_func>
void init_retarget_map(const cv::Mat &dst, const cv::Mat &src, map_t *map, int half_patch)
{
    int d_height = dst.rows;
    int d_width = dst.cols;
    int s_height = src.rows;
    int s_width = src.cols;

    float y_factor = (float) src.rows / (float) dst.rows;
    float x_factor = (float) src.cols / (float) dst.cols;
    float fy = 0;
    float fx = 0;

    for (int dy = 0; dy < dst.rows; dy++) {
        int sy = (int) floor(fy);
        fx = 0;

        for (int dx = 0; dx < dst.cols; dx++) {
            int sx = (int) floor(fx);
            int didx = dy * d_width + dx;
            int sidx = sy * s_width + sx;

            map[didx].x = sx;
            map[didx].y = sy;
            map[didx].dist = patch_distance(dst, src, dx, dy, sx, sy, half_patch);
            
            fx += x_factor;
        }

        fy += y_factor;
    }
}

/**
 * For each pixel in first, search for optimal nn pixel in second 
 */ 
// template<distance_func_t distance_func>
void nn_search(const cv::Mat &first, const cv::Mat &second, map_t *curMap, int half_patch)
{
    int f_width = first.cols;
    int f_height = first.rows;
    int s_width = second.cols;
    int s_height = second.rows;
    int search_radius = min(MAX_SEARCH_RADIUS, min(s_width, s_height));

    for (int fy = 0; fy < f_height; fy++) {
        for (int fx = 0; fx < f_width; fx++) {
            int f = (fy * f_width) + fx;
            int best_x = curMap[f].x; 
            int best_y = curMap[f].y; 
            float best_dist = curMap[f].dist;

            // propagate
            if (fx > 0) {
                // find neighbor's patch
                int pf = f - 1;
                int px = curMap[pf].x + 1;
                int py = curMap[pf].y;
                
                if (px < s_width) { 
                    float dist = patch_distance(first, second, fx, fy, px, py, half_patch);
                    
                    if (dist < best_dist) {
                        best_x = px; 
                        best_y = py;
                        best_dist = dist;
                    }
                }
            }

            if (fy > 0) {
                // find neighbor's patch
                int pf = f - f_width;
                int px = curMap[pf].x;
                int py = curMap[pf].y + 1;
                
                if (py < s_height) { 
                    float dist = patch_distance(first, second, fx, fy, px, py, half_patch);
                    
                    if (dist < best_dist) {
                        best_x = px; 
                        best_y = py;
                        best_dist = dist;
                    }
                }
            }

            // random search
            for (int radius = search_radius; radius >= 1; radius /= 2) {
                int rx, ry;
                pick_random_pixel(radius, s_height, s_width, 
                    best_x, best_y, &rx, &ry);

                float dist = patch_distance(first, second, fx, fy, rx, ry, half_patch);

                if (dist < best_dist) {
                    best_x = rx;
                    best_y = ry;
                    best_dist = dist;
                }
            }
            
            curMap[f].x = best_x;
            curMap[f].y = best_y;
            curMap[f].dist = best_dist;
        }
    }
}

void nn_map(const cv::Mat &src, cv::Mat &dst, map_t *map)
{
    int src_height = src.rows;
    int src_width = src.cols;
    int dst_height = dst.rows;
    int dst_width = dst.cols;

    for (int dy = 0; dy < dst_height; dy++) {
        for (int dx = 0; dx < dst_width; dx++) {
            int idx = dy * dst_width + dx;

            if (map[idx].x < 0 || map[idx].x >= src_width) {
                cout << "Bad X position " << map[idx].x 
                    << " at (" << dx << ", " << dy << ")" << endl;
            }
            else if (map[idx].y < 0 || map[idx].y >= src_height) {
                cout << "Bad Y position " << map[idx].y 
                    << " at (" << dx << ", " << dy << ")" << endl;
            }
            else {
                dst.at<Vec3b>(dy, dx) = src.at<Vec3b>(map[idx].y, map[idx].x);
            }
        }
    }
}

void nn_map_average(const cv::Mat &src, cv::Mat &dst, map_t *map, int half_patch)
{
    int src_height = src.rows;
    int src_width = src.cols;
    int dst_height = dst.rows;
    int dst_width = dst.cols;

    for (int dy = 0; dy < dst_height; dy++) {
        int fy_min = max(dy - half_patch, 0);
        int fy_max = min(dy + half_patch, dst_height - 1);
        int fy_len = fy_max - fy_min + 1;

        for (int dx = 0; dx < dst_width; dx++) {
            int fx_min = max(dx - half_patch, 0);
            int fx_max = min(dx + half_patch, dst_width - 1);
            int fx_len = fx_max - fx_min + 1;

            int pixel_sums[3];
            pixel_sums[0] = pixel_sums[1] = pixel_sums[2] = 0;
            
            for (int fy = fy_min; fy <= fy_max; fy++) {
                for (int fx = fx_min; fx <= fx_max; fx++) {
                    int f = fy * dst_width + fx;
                    int px = map[f].x;
                    int py = map[f].y;

                    Vec3b spixel = src.at<Vec3b>(py, px);
                    pixel_sums[0] += spixel[0];
                    pixel_sums[1] += spixel[1];
                    pixel_sums[2] += spixel[2];
                }
            }

            int num_pixels = fy_len * fx_len;
            dst.at<Vec3b>(dy, dx)[0] = pixel_sums[0] / num_pixels;
            dst.at<Vec3b>(dy, dx)[1] = pixel_sums[1] / num_pixels;
            dst.at<Vec3b>(dy, dx)[2] = pixel_sums[2] / num_pixels;
        }
    }
}

void patchmatch(const cv::Mat &src, cv::Mat &dst, int half_patch)
{
    int dst_height = dst.rows;
    int dst_width = dst.cols;

    map_t *curMap = (map_t *) malloc(dst_height * dst_width * sizeof(map_t));
    init_random_map(dst, src, curMap);

    for (int i = 1; i <= NUM_ITERATIONS; i++) {
        cout << "PATCHMATCH iteration " << i << endl;
        nn_search(dst, src, curMap);

        if (SAVE_ITER_OUTPUT && (i % 4) == 0) {
            char fname[64];
            sprintf(fname, "scratch/pm-iter-%i.jpg", i);
            Mat cur = dst.clone();
            nn_map_average(src, cur, curMap);
            imwrite(fname, cur);
            cur.release();
        }
    }

    nn_map_average(src, dst, curMap);

    free(curMap);
}