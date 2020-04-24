#include <opencv2/opencv.hpp>
#include <patchmatch.h>
#include <float.h>
#include <math.h>

using namespace cv;

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

template<distance_func_t distance_func>
float patch_distance(cv::Mat &first, int fx, int fy, cv::Mat &second, int sx, int sy)
{
    float dist = 0;
    for (int j = -HALF_PATCH; j < HALF_PATCH; j++) {
        for (int i = -HALF_PATCH; i < HALF_PATCH; i++) {
            int fx1 = min(first.cols - 1, max(0, fx + i));
            int fy1 = min(first.rows - 1, max(0, fy + i));
            Vec3b fpixel = first.at<Vec3b>(fy1, fx1);

            int sx1 = min(second.cols - 1, max(0, sx + i));
            int sy1 = min(second.rows - 1, max(0, sy + i));
            Vec3b spixel = second.at<Vec3b>(sy1, sx1);

            dist += distance_func(fpixel, spixel);
        }
    }
    return dist;
}


void pick_random_pixel(int radius, int height, int width, 
    int sx, int sy, int &rx, int &ry)
{
    rx = 0;
    ry = 0; 
    bool picked = false;

    while (!picked) {
        // north, south, east or west
        int cardinal = random() % 4;

        // little switch to select a patch on the edge of radius
        switch (cardinal) {
            case 0: 
                if (sy - radius > 0) {
                    ry = sy - radius;
                    rx = (random() % (2 * HALF_PATCH + 1)) - HALF_PATCH + sx;
                    rx = min(width - 1, max(0, rx));
                    picked = true;
                }
                break;

            case 1:
                if (sx - radius > 0) {
                    ry = (random() % (2 * HALF_PATCH + 1)) - HALF_PATCH + sy;
                    ry = min(height - 1, max(0, ry));
                    rx = sx - radius;
                    picked = true;
                }
                break;

            case 2:
                if (sy + radius < height) {
                    ry = sy + radius;
                    rx = (random() % (2 * HALF_PATCH + 1)) - HALF_PATCH + sx;
                    rx = min(width - 1, max(0, rx)) ;
                    picked = true;
                }
                break;

            case 3:
                if (sx + radius < width) {
                    ry = (random() % (2 * HALF_PATCH + 1)) - HALF_PATCH + sy;
                    ry = min(height - 1, max(0, ry));
                    rx = sx + radius;
                    picked = true;
                }
                break;
        }
    }
}

template<distance_func_t distance_func>
void nn_search(cv::Mat &first, cv::Mat &second, map_t *curMap, map_t *newMap)
{
    int f, sx, sy;
    float dist = FLT_MAX;

    for (int fy = 0; fy < first.rows; fy++) {
        for (int fx = 0; fx < first.cols; fx++) {
            f = (fy * first.rows) + fx;

            sx = curMap[f].x; 
            sy = curMap[f].y; 
            dist = curMap[f].dist;

            // propagate
            if (fx > 0) {
                int nearIdx = f - 1;
                if ((curMap[nearIdx].dist < dist) && (curMap[nearIdx].x > 0)) {
                    sx = curMap[ nearIdx ].x - 1; 
                    sy = curMap[ nearIdx ].y; 
                    dist = patch_distance<distance_func>(first, fx, fy, second, sx, sy);
                }
            }

            if (fx > 0) {
                int nearIdx = f - first.rows;
                if ((curMap[nearIdx].dist < dist) && (curMap[nearIdx].y > 0)) {
                    sx = curMap[ nearIdx ].x; 
                    sy = curMap[ nearIdx ].y - 1; 
                    dist = patch_distance<distance_func>(first, fx, fy, second, sx, sy);
                }
            }

            // random search
            for (int search = RANDOM_TRIALS; search >= 0; search--) {
                int radius = search * HALF_PATCH;

                int rx, ry;
                pick_random_pixel(radius, second.rows, second.cols,
                    sx, sy, rx, ry);

                float rdist = patch_distance<distance_func>(first, fx, fy, second, rx, ry);

                if (rdist < dist) {
                    sx = rx;
                    sy = ry;
                    dist = rdist;
                }
            }
            
            newMap[f].x = sx;
            newMap[f].y = sy;
            newMap[f].dist = dist;
        }
    }
}