#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "patchmatch.h"
#include "cycletimer.h"

using namespace cv;
using namespace std;


inline int get_pidx(int y, int x, int w) { return y * w + x; }

inline int get_cidx(int y, int x, int w, int c) { return (y * w + x) * N_CHANNELS + c; }

inline float square(float x) { return x * x; }

inline float sum_squared_diff(float *fpixel, float *spixel)
{
    float dist = sqrt(
        square(fpixel[0] - spixel[0]) +
        square(fpixel[1] - spixel[1]) +
        square(fpixel[2] - spixel[2])
    );
    return dist;
}

inline float sum_absolute_diff(float *fpixel, float *spixel)
{
    float dist = sqrt(
        abs(fpixel[0] - spixel[0]) +
        abs(fpixel[1] - spixel[1]) +
        abs(fpixel[2] - spixel[2])
    );
    return dist;
}

inline float patch_distance(float *first, float *second, 
    int fx, int fy, int sx, int sy, 
    int height, int width, int half_patch)
{
    float dist = 0;
    for (int j = -half_patch; j <= half_patch; j++) {
        for (int i = -half_patch; i <= half_patch; i++) {
            int fx1 = min(width - 1, max(0, fx + i));
            int fy1 = min(height - 1, max(0, fy + i));
            float *fpixel = first + get_pidx(fy1, fx1, width) * N_CHANNELS;

            int sx1 = min(width - 1, max(0, sx + i));
            int sy1 = min(height - 1, max(0, sy + i));
            float *spixel = second + get_pidx(sy1, sx1, width) * N_CHANNELS;

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
void init_random_map(float *first, float *second, map_t *map, 
    int height, int width, int half_patch)
{
    for (int y = 0; y < height; y++ ) {
        for (int x = 0; x < width; x++ ) {
            int rx = random() % width;
            int ry = random() % height;
            int idx = y * width + x;

            map[idx].x = rx;
            map[idx].y = ry;
            map[idx].dist = patch_distance(first, second, x, y, rx, ry, 
                height, width, half_patch);
        }
    }
}

/**
 * For each pixel in first, search for optimal nn pixel in second 
 */ 
void nn_search(float *first, float *second, map_t *curMap, 
    int height, int width, int half_patch)
{
    // int search_radius = min(MAX_SEARCH_RADIUS, min(width, height));
    // int search_radius = max(width, height);
    int search_radius = min(5, max(width, height));

    for (int fy = 0; fy < height; fy++) {
        for (int fx = 0; fx < width; fx++) {
            int f = (fy * width) + fx;
            int best_x = curMap[f].x; 
            int best_y = curMap[f].y; 
            float best_dist = curMap[f].dist;

            // propagate
            if (fx > 0) {
                // find neighbor's patch
                int pf = f - 1;
                int px = curMap[pf].x + 1;
                int py = curMap[pf].y;
                
                if (px < width) { 
                    float dist = patch_distance(first, second, fx, fy, px, py, height, width, half_patch);
                    
                    if (dist < best_dist) {
                        best_x = px; 
                        best_y = py;
                        best_dist = dist;
                    }
                }
            }

            if (fy > 0) {
                // find neighbor's patch
                int pf = f - width;
                int px = curMap[pf].x;
                int py = curMap[pf].y + 1;
                
                if (py < height) { 
                    float dist = patch_distance(first, second, fx, fy, px, py, height, width, half_patch);
                    
                    if (dist < best_dist) {
                        best_x = px; 
                        best_y = py;
                        best_dist = dist;
                    }
                }
            }

            // random search
            // for (int radius = search_radius; radius >= 1; radius /= 2) {
            for (int radius = search_radius; radius >= 1; radius--) {
                int rx, ry;
                pick_random_pixel(radius, height, width, 
                    best_x, best_y, &rx, &ry);

                float dist = patch_distance(first, second, fx, fy, rx, ry, height, width, half_patch);

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

void nn_map(float *src, float *dst, map_t *map,
    int height, int width)
{
    for (int dy = 0; dy < height; dy++) {
        for (int dx = 0; dx < width; dx++) {
            int idx = get_pidx(dy, dx, width);

            if (map[idx].x < 0 || map[idx].x >= width) {
                cout << "Bad X position " << map[idx].x 
                    << " at (" << dx << ", " << dy << ")" << endl;
            }
            else if (map[idx].y < 0 || map[idx].y >= height) {
                cout << "Bad Y position " << map[idx].y 
                    << " at (" << dx << ", " << dy << ")" << endl;
            }
            else {
                int midx = get_pidx(map[idx].y, map[idx].x, width);
                dst[idx * N_CHANNELS + 0] = src[midx * N_CHANNELS + 0];
                dst[idx * N_CHANNELS + 1] = src[midx * N_CHANNELS + 1];
                dst[idx * N_CHANNELS + 2] = src[midx * N_CHANNELS + 2];
            }
        }
    }
}

void nn_map_average(float *src, float *dst, map_t *map, 
    int height, int width, int half_patch)
{
    half_patch = min(3, half_patch);

    for (int dy = 0; dy < height; dy++) {
        int fy_min = max(dy - half_patch, 0);
        int fy_max = min(dy + half_patch, height - 1);
        int fy_len = fy_max - fy_min + 1;

        for (int dx = 0; dx < width; dx++) {
            int fx_min = max(dx - half_patch, 0);
            int fx_max = min(dx + half_patch, width - 1);
            int fx_len = fx_max - fx_min + 1;

            int pixel_sums[3];
            pixel_sums[0] = pixel_sums[1] = pixel_sums[2] = 0;
            
            for (int fy = fy_min; fy <= fy_max; fy++) {
                for (int fx = fx_min; fx <= fx_max; fx++) {
                    int f = fy * width + fx;
                    int px = map[f].x;
                    int py = map[f].y;

                    float *spixel = src + get_pidx(py, px, width) * N_CHANNELS;
                    pixel_sums[0] += spixel[0];
                    pixel_sums[1] += spixel[1];
                    pixel_sums[2] += spixel[2];
                }
            }

            int num_pixels = fy_len * fx_len;

            float *dpixel = dst + get_pidx(dy, dx, width) * N_CHANNELS;
            dpixel[0] = pixel_sums[0] / num_pixels;
            dpixel[1] = pixel_sums[1] / num_pixels;
            dpixel[2] = pixel_sums[2] / num_pixels;
        }
    }
}

void patchmatch(float *src, float *dst, int height, int width, int half_patch)
{
    double t1, time_init, time_search = 0, time_map;
    map_t *curMap = (map_t *) malloc(height * width * sizeof(map_t));

    t1 = currentSeconds();
    init_random_map(dst, src, curMap, height, width, half_patch);
    time_init = currentSeconds() - t1;

    for (int i = 1; i <= NUM_ITERATIONS; i++) {
        #if DEBUG
        cout << "PATCHMATCH iteration " << i << endl;
        #endif

        t1 = currentSeconds();
        nn_search(dst, src, curMap, height, width, half_patch);
        time_search += currentSeconds() - t1;

        #if DEBUG
        if (SAVE_ITER_OUTPUT && (i % 4) == 0) {
            char fname[64];
            sprintf(fname, "../scratch/pm-iter-%i.jpg", i);
            cout << fname << endl;

            float *cur;
            clone_array(dst, &cur, height, width);
            nn_map_average(src, cur, curMap, height, width, half_patch);
            imwrite_array(fname, cur, height, width, 3);
            free(cur);
        }
        #endif
    }

    t1 = currentSeconds();
    nn_map_average(src, dst, curMap, height, width, half_patch);
    time_map = currentSeconds() - t1;

    free(curMap);

    cout << "Time init: "<< time_init << endl;
    cout << "Time search per iter: "<< (time_search / NUM_ITERATIONS) << endl;
    cout << "Time map: "<< time_map << endl;
}