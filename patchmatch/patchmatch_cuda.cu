#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h>

#include "patchmatch.h"

using namespace cv;
using namespace std;

__device__ __inline__ float square(float x) { return x * x; }

__device__ __inline__ int get_max(int x,int y)
{
    if(x > y) return x;
    return y;
}

__device__ __inline__ int get_min(int x,int y)
{
    if(x < y) return x;
    return y;
}

__device__ __inline__ void init_rand(curandState *state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    curand_init(i, j, 0, state);
}

__device__ __inline__ float get_rand(curandState *state)
{
	return curand_uniform(state);
}

__device__ __inline__
float sum_squared_diff(uchar3 fpixel, uchar3 spixel)
{
    float dist = sqrt(
        square(fpixel.x - spixel.x) +
        square(fpixel.y - spixel.y) +
        square(fpixel.z - spixel.z)
    );
    return dist;
}

__device__ __inline__
float sum_absolute_diff(uchar3 fpixel, uchar3 spixel)
{
    float dist = sqrt(
        abs(fpixel.x - spixel.x) +
        abs(fpixel.y - spixel.y) +
        abs(fpixel.z - spixel.z)
    );
    return dist;
}

__device__
float patch_distance(uchar3 *first, uchar3 *second, 
    int fx, int fy, int sx, int sy, 
    int width, int height, int half_patch)
{
    float dist = 0;
    for (int j = -half_patch; j <= half_patch; j++) {
        for (int i = -half_patch; i <= half_patch; i++) {
            int fx1 = get_min(width - 1, get_max(0, fx + i));
            int fy1 = get_min(height - 1, get_max(0, fy + i));
            int f = fy1 * width + fx1;

            int sx1 = get_min(width - 1, get_max(0, sx + i));
            int sy1 = get_min(height - 1, get_max(0, sy + i));
            int s = sy1 * width + sx1;

            dist += sum_squared_diff(first[f], second[s]);
        }
    }
    return dist;
}

__device__
void init_random_map(uchar3 *first, uchar3 *second, map_t *map, 
    int width, int height, int half_patch)
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (y >= height || x >= width) return;

    curandState state;
    init_rand(&state);

    int rx = (int)(get_rand(&state) * width) % width;
    int ry = (int)(get_rand(&state) * height) % height;
    int idx = y * width + x;

    map[idx].x = rx;
    map[idx].y = ry;
    map[idx].dist = patch_distance(first, second, x, y, rx, ry, 
        width, height, half_patch);
}

__device__
void nn_search(uchar3 *first, uchar3 *second, map_t *map, 
    int width, int height, int half_patch)
{
    int fy = threadIdx.y + blockIdx.y * blockDim.y;
    int fx = threadIdx.x + blockIdx.x * blockDim.x;
    if (fy >= height || fx >= width) return;

    int search_radius = get_max(width, height);

    curandState state;
    init_rand(&state);

    int f = (fy * width) + fx;
    int best_x = map[f].x; 
    int best_y = map[f].y; 
    float best_dist = map[f].dist;

    // propagate
    if (fx > 0) {
        // find neighbor's patch
        int pf = f - 1;
        int px = map[pf].x + 1;
        int py = map[pf].y;
        
        if (px < width) { 
            float dist = patch_distance(first, second, fx, fy, px, py, 
                width, height, half_patch);
            
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
        int px = map[pf].x;
        int py = map[pf].y + 1;
        
        if (py < height) { 
            float dist = patch_distance(first, second, fx, fy, px, py, 
                width, height, half_patch);
            
            if (dist < best_dist) {
                best_x = px; 
                best_y = py;
                best_dist = dist;
            }
        }
    }

    // random search
    for (int radius = search_radius; radius >= 1; radius /= 2) {
        // pick a random pixel
        int xmin = get_max(best_x - radius, 0);
        int xmax = get_min(best_x + radius, width);
        int ymin = get_max(best_y - radius, 0);
        int ymax = get_min(best_y + radius, height);
        int xlen = (xmax - xmin);
        int ylen = (ymax - ymin);
        int rx = (int)(get_rand(&state) * xlen) % xlen + xmin;
        int ry = (int)(get_rand(&state) * ylen) % ylen + ymin;

        float dist = patch_distance(first, second, fx, fy, rx, ry, 
            width, height, half_patch);

        if (dist < best_dist) {
            best_x = rx;
            best_y = ry;
            best_dist = dist;
        }
    }
    
    map[f].x = best_x;
    map[f].y = best_y;
    map[f].dist = best_dist;
        
}

__device__
void nn_map(uchar3 *src, uchar3 *dst, map_t *map, 
    int width, int height, int half_patch)
{
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    if (dy >= height || dx >= width) return;
    int idx = dy * width + dx;

    if (map[idx].x < 0 || map[idx].x >= width ||
        map[idx].y < 0 || map[idx].y >= height) {
        return;
    }
    else {
        int midx = map[idx].y * width + map[idx].x;
        dst[idx] = src[midx];
    }  
}

__device__
void nn_map_average(uchar3 *src, uchar3 *dst, map_t *map, 
    int width, int height, int half_patch)
{
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    if (dy >= height || dx >= width) return;
    
    int fy_min = get_max(dy - half_patch, 0);
    int fy_max = get_min(dy + half_patch, height - 1);
    int fy_len = fy_max - fy_min + 1;

    int fx_min = get_max(dx - half_patch, 0);
    int fx_max = get_min(dx + half_patch, width - 1);
    int fx_len = fx_max - fx_min + 1;

    int pixel_sums[3];
    pixel_sums[0] = pixel_sums[1] = pixel_sums[2] = 0;
    
    for (int fy = fy_min; fy <= fy_max; fy++) {
        for (int fx = fx_min; fx <= fx_max; fx++) {
            int f = fy * width + fx;
            int px = map[f].x;
            int py = map[f].y;

            int p = py * width + px;
            uchar3 spixel = src[p];
            pixel_sums[0] += spixel.x;
            pixel_sums[1] += spixel.y;
            pixel_sums[2] += spixel.z;
        }
    }

    int d = dy * width + dx;
    int num_pixels = fy_len * fx_len;
    dst[d].x = pixel_sums[0] / num_pixels;
    dst[d].y = pixel_sums[1] / num_pixels;
    dst[d].z = pixel_sums[2] / num_pixels;
}

__global__ 
void init_kernel(uchar3 *dst, uchar3 *src, map_t *map, 
    int width, int height, int half_patch)
{
    dst[0].x = 30;
    dst[0].y = 45;
    dst[0].z = 100;
    printf("1\n");
}

__global__ 
void patchmatch_kernel(uchar3 *dst, uchar3 *src, map_t *map, 
    int width, int height, int half_patch)
{
    init_random_map(dst, src, map, width, height, half_patch);
    __syncthreads();
    for (int i = 1; i <= NUM_ITERATIONS; i++) {
        nn_search(dst, src, map, width, height, half_patch);
        __syncthreads();
    }
    nn_map_average(src, dst, map, width, height, half_patch);
}

void mat_to_uchar3_array(const cv::Mat &mat, uchar3 **arr_ptr)
{
    int ny = mat.rows;
    int nx = mat.cols;
    uchar3 *arr = (uchar3 *) malloc(ny * nx * sizeof(uchar3));

    int idx = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            Vec3b pixel = mat.at<Vec3b>(y, x);
            arr[idx].x = pixel[0];
            arr[idx].y = pixel[1];
            arr[idx].z = pixel[2];
            idx++;
        }
    }

    *arr_ptr = arr;
}

void uchar3_array_to_mat(uchar3 *arr, cv::Mat &mat)
{
    int ny = mat.rows;
    int nx = mat.cols;

    int idx = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            uchar3 pixel = arr[idx];
            mat.at<Vec3b>(y, x)[0] = pixel.x;
            mat.at<Vec3b>(y, x)[1] = pixel.y;
            mat.at<Vec3b>(y, x)[2] = pixel.z;
            idx++;
        }
    }
}

void print_uchar3(uchar3 *arr, int len)
{
    for (int i = 0; i < len; i++) {
        cout << (int) arr[i].x << "," << (int) arr[i].y << "," << (int) arr[i].z << endl;
    }
}

void print_map(map_t *map, int len)
{
    for (int i = 0; i < len; i++) {
        cout << map[i].x << "," << map[i].y << "," << map[i].dist << endl;
    }
}

void patchmatch(const cv::Mat &srcMat, cv::Mat &dstMat, int half_patch)
{
    if (srcMat.rows != dstMat.rows || srcMat.cols != dstMat.cols) {
        cout << "Error: size not match." << endl;
        return;
    }

    int height = dstMat.rows;
    int width = dstMat.cols;
    int len = height * width;

    uchar3 *src, *dst;
    mat_to_uchar3_array(srcMat, &src);
    mat_to_uchar3_array(dstMat, &dst);
    map_t *map = (map_t *) malloc(height * width * sizeof(map_t));

    cout << "src" << endl;
    print_uchar3(src, 10);
    cout << "dst" << endl;
    print_uchar3(dst, 10);

    uchar3 *d_src, *d_dst;
    map_t *d_map;
    cudaMalloc((void **)&d_src, len * sizeof(uchar3));
    cudaMalloc((void **)&d_dst, len * sizeof(uchar3));
    cudaMalloc((void **)&d_map, len * sizeof(map_t));

    cudaMemcpy(d_src, src, len * sizeof(uchar3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst, len * sizeof(uchar3), cudaMemcpyHostToDevice); 
    
    int blocksize = 32;
    dim3 blockDim(blocksize, blocksize, 1);
    dim3 gridDim(
        (height + blocksize - 1) / blocksize,
        (width + blocksize - 1) / blocksize, 
        1);

    init_kernel<<<blockDim, gridDim>>>(d_dst, d_src, d_map, width, height, half_patch);
    cudaDeviceSynchronize();
    
    cudaMemcpy(dst, d_dst, len * sizeof(uchar3), cudaMemcpyDeviceToHost);
    cudaMemcpy(map, d_map, len * sizeof(map_t), cudaMemcpyDeviceToHost);

    cout << "dst" << endl;
    print_uchar3(dst, 10);
    cout << "map" << endl;
    print_map(map, 10);

    patchmatch_kernel<<<blockDim, gridDim>>>(d_dst, d_src, d_map, width, height, half_patch);
    cudaDeviceSynchronize();

    cudaMemcpy(dst, d_dst, len * sizeof(uchar3), cudaMemcpyDeviceToHost);
    cudaMemcpy(map, d_map, len * sizeof(map_t), cudaMemcpyDeviceToHost);

    cout << "dst" << endl;
    print_uchar3(dst, 10);
    cout << "map" << endl;
    print_map(map, 10);

    uchar3_array_to_mat(dst, dstMat);

    cudaFree(d_map);
    cudaFree(d_src);
    cudaFree(d_dst);

    free(src);
    free(dst);
}