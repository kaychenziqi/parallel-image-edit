#ifndef PATCHMATCH_H_
#define PATCHMATCH_H_

#define NUM_ITERATIONS 10
#define MAX_SEARCH_RADIUS 256
#define SAVE_ITER_OUTPUT 0

#ifndef HALF_PATCH
#define HALF_PATCH 7
#endif

// map entry type
typedef struct {
    int x;
    int y;
    float dist;
} map_t;

// distance functions
float sum_squared_diff(float *fpixel, float *spixel);
float sum_absolute_diff(float *fpixel, float *spixel);
float patch_distance(float *first, float *second, 
    int fx, int fy, int sx, int sy, 
    int height, int width, int half_patch = 1);

// intialize nearest neighbor field
void init_random_map(float *first, float *second, map_t *map, 
    int height, int width, int half_patch = 1);

// nearest neighbor field
void nn_search(float *first, float *second, map_t *curMap, 
    int height, int width, int half_patch = 1);
void nn_map(float *src, float *dst, map_t *map,
    int height, int width);
void nn_map_average(float *src, float *dst, map_t *map, 
    int height, int width, int half_patch = 1);

void patchmatch(float *src, float *dst, 
    int height, int width, int half_patch = 1);

#endif