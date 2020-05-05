#include <stdio.h>
#include <curand_kernel.h>

#include "patchmatch.h"

using namespace std;

#define N_CHANNELS 4

__device__ __inline__ float square(float x) { return x * x; }

__device__ __inline__ int get_max(int x,int y) { return (x > y) ? x : y; }

__device__ __inline__ int get_min(int x,int y) { return (x <= y) ? x : y; }

__device__ __inline__ int get_pidx(int y, int x, int w) { return y * w + x; }

__device__ __inline__ int get_cidx(int y, int x, int w, int c) 
{ 
    return (y * w + x) * N_CHANNELS + c; 
}

__device__ __inline__ 
void init_rand(curandState *state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    curand_init(i, j, 0, state);
}

__device__ __inline__ 
float get_rand(curandState *state)
{
	return curand_uniform(state);
}

__device__ __inline__
float sum_squared_diff(float *fpixel, float *spixel)
{
    float dist = sqrt(
        square(fpixel[0] - spixel[0]) +
        square(fpixel[1] - spixel[1]) +
        square(fpixel[2] - spixel[2])
    );
    return dist;
}

__device__ __inline__
float sum_absolute_diff(float *fpixel, float *spixel)
{
    float dist = sqrt(
        abs(fpixel[0] - spixel[0]) +
        abs(fpixel[1] - spixel[1]) +
        abs(fpixel[2] - spixel[2])
    );
    return dist;
}

__device__ __inline__
float patch_distance(float *first, float *second, 
    int fx, int fy, int sx, int sy, 
    int height, int width, int half_patch)
{
    float dist = 0;
    for (int j = -half_patch; j <= half_patch; j++) {
        for (int i = -half_patch; i <= half_patch; i++) {
            int fx1 = get_min(width - 1, get_max(0, fx + i));
            int fy1 = get_min(height - 1, get_max(0, fy + i));
            float *fpixel = first + get_pidx(fy1, fx1, width) * N_CHANNELS;

            int sx1 = get_min(width - 1, get_max(0, sx + i));
            int sy1 = get_min(height - 1, get_max(0, sy + i));
            float *spixel = second + get_pidx(sy1, sx1, width) * N_CHANNELS;

            dist += sum_squared_diff(fpixel, spixel);
        }
    }
    return dist;
}

__device__
void init_random_map(float *first, float *second, map_t *map, 
    int height, int width, int half_patch)
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
        height, width, half_patch);  
}

__device__
void nn_search(float *first, float *second, map_t *curMap, 
    int height, int width, int half_patch)
{
    // int search_radius = get_min(MAX_SEARCH_RADIUS, get_min(width, height));
    int search_radius = get_max(width, height);

    curandState state;
    init_rand(&state);

    int fy = threadIdx.y + blockIdx.y * blockDim.y;
    int fx = threadIdx.x + blockIdx.x * blockDim.x;
    if (fy >= height || fx >= width) return;

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
    for (int radius = search_radius; radius >= 1; radius /= 2) {
        int xmin = get_max(best_x - radius, 0);
        int xmax = get_min(best_x + radius, width);
        int ymin = get_max(best_y - radius, 0);
        int ymax = get_min(best_y + radius, height);
        int xlen = (xmax - xmin);
        int ylen = (ymax - ymin);
        int rx = (int)(get_rand(&state) * xlen) % xlen + xmin;
        int ry = (int)(get_rand(&state) * ylen) % ylen + ymin;

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

__device__
void nn_map(float *src, float *dst, map_t *map, 
    int height, int width)
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
        int midx = get_pidx(map[idx].y, map[idx].x, width);
        dst[idx * N_CHANNELS + 0] = src[midx * N_CHANNELS + 0];
        dst[idx * N_CHANNELS + 1] = src[midx * N_CHANNELS + 1];
        dst[idx * N_CHANNELS + 2] = src[midx * N_CHANNELS + 2];
    } 
}

__device__
void nn_map_average(float *src, float *dst, map_t *map, 
    int height, int width, int half_patch)
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

__global__ void patchmatch_kernel(float *src, float *dst, map_t *curMap, 
    int height, int width, int half_patch)
{
    init_random_map(dst, src, curMap, height, width, half_patch);
    __syncthreads();

    for (int i = 1; i <= NUM_ITERATIONS; i++) {
        nn_search(dst, src, curMap, height, width, half_patch);
        __syncthreads();
    }
    nn_map_average(src, dst, curMap, height, width, half_patch);
}

__global__ void dummy_kernel(float *src, float *dst, 
    int height, int width, int half_patch)
{
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    int dx = threadIdx.x + blockIdx.x * blockDim.x;

    int idx = dy * width + dx;
    dst[idx * 4] = 0;
    dst[idx * 4 + 1] = 0;
    dst[idx * 4 + 2] = 0;
}

void patchmatch(float *src, float *dst, int height, int width, int half_patch)
{
    int n_pixels = height * width;

    float *d_src, *d_dst;
    map_t *d_map;
    cudaMalloc((void **)&d_src, n_pixels * N_CHANNELS * sizeof(float));
    cudaMalloc((void **)&d_dst, n_pixels * N_CHANNELS * sizeof(float));
    cudaMalloc((void **)&d_map, n_pixels * sizeof(map_t));

    cudaMemcpy(d_src, src, n_pixels * N_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, dst, n_pixels * N_CHANNELS * sizeof(float), cudaMemcpyHostToDevice); 
    
    int blocksize = 32;
    dim3 blockDim(blocksize, blocksize, 1);
    dim3 gridDim(
        (height + blocksize - 1) / blocksize,
        (width + blocksize - 1) / blocksize, 
        1);

    patchmatch_kernel<<<gridDim, blockDim>>>(d_src, d_dst, d_map, height, width, half_patch);
    // dummy_kernel<<<gridDim, blockDim>>>(d_src, d_dst, width, height, half_patch);
    cudaDeviceSynchronize();

    cudaMemcpy(dst, d_dst, n_pixels * N_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_map);
    cudaFree(d_src);
    cudaFree(d_dst);
}