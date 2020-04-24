#include <iostream>
#include "math.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include "omp.h"
using namespace std;
#define ITERATIONS 70000
#define THREAD_COUNT 12


enum corner_pixel {INSIDE_MASK, BOUNDRY, OUTSIDE};

void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
	if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}

void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc)
{
	if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}


void calculate_boundbox(int target_w, int target_h, int target_nc, int *boundryPixelArray, int *boundBoxMinX, int *boundBoxMinY, int *boundBoxMaxX, int *boundBoxMaxY){
    *boundBoxMaxY = INT32_MIN;
    *boundBoxMaxX = INT32_MIN;
    *boundBoxMinY = INT32_MAX;
    *boundBoxMinX = INT32_MAX;

    for(int channel = 0; channel < target_nc; channel++){
        for(int y = 0; y < target_h; y++){
            for(int x = 0; x < target_w; x++){
                int id = x + y*target_w + channel * target_w * target_h;
                if(boundryPixelArray[id]==BOUNDRY){
                    if(x<*boundBoxMinX){
                        *boundBoxMinX = x;
                    }
                    if(x>*boundBoxMaxX){
                        *boundBoxMaxX = x;
                    }
                    if(y<*boundBoxMinY){
                        *boundBoxMinY = y;
                    }
                    if(y>*boundBoxMaxY){
                        *boundBoxMaxY = y;
                    }  
                }
            }
        }
    }
}

void calculate_boundbox_omp_static(int target_w, int target_h, int target_nc, int *boundryPixelArray, int *boundBoxMinX, int *boundBoxMinY, int *boundBoxMaxX, int *boundBoxMaxY){
    *boundBoxMaxY = INT32_MIN;
    *boundBoxMaxX = INT32_MIN;
    *boundBoxMinY = INT32_MAX;
    *boundBoxMinX = INT32_MAX;

    #if OMP
    #pragma omp parallel
    #endif
    {
        int t = omp_get_thread_num();
        int x_each = target_w/4;
        int y_each = target_h/3;
        int y_begin = (t/4)*y_each;
        int y_end = y_begin+y_each;
        int x_begin = (t%4)*x_each;
        int x_end = x_begin+x_each;
        for(int channel = 0; channel < target_nc; channel++){
            for(int y = y_begin; y<y_end && y < target_h; y++){
                for(int x = x_begin; x<x_end && x < target_w; x++){
                    int id = x + y*target_w + channel * target_w * target_h;
                    if(boundryPixelArray[id]==BOUNDRY){
                        if(x<*boundBoxMinX){
                            *boundBoxMinX = x;
                        }
                        if(x>*boundBoxMaxX){
                            *boundBoxMaxX = x;
                        }
                        if(y<*boundBoxMinY){
                            *boundBoxMinY = y;
                        }
                        if(y>*boundBoxMaxY){
                            *boundBoxMaxY = y;
                        }  
                    }
                }
            }
        }
    }
}

//TODO
void calculate_boundbox_omp_dynamic(int target_w, int target_h, int target_nc, int *boundryPixelArray, int *boundBoxMinX, int *boundBoxMinY, int *boundBoxMaxX, int *boundBoxMaxY){
    *boundBoxMaxY = INT32_MIN;
    *boundBoxMaxX = INT32_MIN;
    *boundBoxMinY = INT32_MAX;
    *boundBoxMinX = INT32_MAX;

    #if OMP
    #pragma omp parallel
    #endif
    {
        int t = omp_get_thread_num();
        int x_each = target_w/4;
        int y_each = target_h/3;
        int y_begin = (t/4)*y_each;
        int y_end = y_begin+y_each;
        int x_begin = (t%4)*x_each;
        int x_end = x_begin+x_each;
        for(int channel = 0; channel < target_nc; channel++){
            for(int y = y_begin; y<y_end && y < target_h; y++){
                for(int x = x_begin; x<x_end && x < target_w; x++){
                    int id = x + y*target_w + channel * target_w * target_h;
                    if(boundryPixelArray[id]==BOUNDRY){
                        if(x<*boundBoxMinX){
                            *boundBoxMinX = x;
                        }
                        if(x>*boundBoxMaxX){
                            *boundBoxMaxX = x;
                        }
                        if(y<*boundBoxMinY){
                            *boundBoxMinY = y;
                        }
                        if(y>*boundBoxMaxY){
                            *boundBoxMaxY = y;
                        }  
                    }
                }
            }
        }
    }
}

void extract_boundary(float *maskIn, int *boundryPixelArray, int source_nchannel, int source_width, int source_height){
    for(int channel = 0; channel < source_nchannel; channel++){
        for(int y = 0; y < source_height; y++){
            for(int x = 0; x < source_width; x++){
                int id = x + y*source_width + channel * source_width * source_height;
                
                // if(x==0 && y==0 && maskIn[id]) boundryPixelArray[id]=CORNER_PIXEL_0_0;
                // else if(x==0 && y==source_height-1 && maskIn[id]) boundryPixelArray[id]=CORNER_PIXEL_0_H;
                // else if(x==source_width-1 && y==0 && maskIn[id]) boundryPixelArray[id]=CORNER_PIXEL_W_0;
                // else if(x==source_width-1 && y==source_height-1 && maskIn[id]) boundryPixelArray[id]=CORNER_PIXEL_W_H;
                // else if(x==0 && y < source_height-1 && maskIn[id]) boundryPixelArray[id]=EDGE_PIXEL_LEFT;
                // else if(x==source_width-1 && y < source_height-1 && maskIn[id]) boundryPixelArray[id]=EDGE_PIXEL_RIGHT;
                // else if(x < source_width-1 && y==0 && maskIn[id]) boundryPixelArray[id]=EDGE_PIXEL_DOWN;
                // else if(x < source_width-1 && y==source_height-1 && maskIn[id]) boundryPixelArray[id]=EDGE_PIXEL_UP;
                if(x==0 && y==0 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==0 && y==source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==source_width-1 && y==0 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==source_width-1 && y==source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==0 && y < source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==source_width-1 && y < source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x < source_width-1 && y==0 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x < source_width-1 && y==source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else{
                    int id_right = x+1 + y*source_width + channel * source_width * source_height;
                    int id_left = x-1 + y*source_width + channel * source_width * source_height;
                    int id_up = x + (y+1)*source_width + channel * source_width * source_height;
                    int id_down = x + (y-1)*source_width + channel * source_width * source_height;

                    if(maskIn[id]>=0.5&&maskIn[id_right]>=0.5&&maskIn[id_left]>=0.5&&maskIn[id_up]>=0.5&&maskIn[id_down]>=0.5){
                        boundryPixelArray[id] = INSIDE_MASK;
                    }
                    else if(maskIn[id]){
                        boundryPixelArray[id] = BOUNDRY;
                    }
                    else{
                        boundryPixelArray[id] = OUTSIDE;
                    }
                }
            }
        }
    }
}

void extract_boundary_omp_static(float *maskIn, int *boundryPixelArray, int source_nchannel, int source_width, int source_height){
    #if OMP
    #pragma omp parallel
    #endif
    {
        int t = omp_get_thread_num();
        int x_each = source_width/4;
        int y_each = source_height/3;
        int y_begin = (t/4)*y_each;
        int y_end = y_begin+y_each;
        int x_begin = (t%4)*x_each;
        int x_end = x_begin+x_each;
    
        //int tcount = omp_get_num_threads(); we use 12 threads in this experiment
        for(int channel = 0; channel < source_nchannel; channel++){
            for(int y = y_begin; y<y_end && y < source_height; y++){
                for(int x = x_begin; x<x_end && x < source_width; x++){
                    int id = x + y*source_width + channel * source_width * source_height;
                    if(x==0 && y==0 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                    }
                    else if(x==0 && y==source_height-1 && maskIn[id]){
                        boundryPixelArray[id]=OUTSIDE;
                    }
                    else if(x==source_width-1 && y==0 && maskIn[id]){
                        boundryPixelArray[id]=OUTSIDE;
                    }
                    else if(x==source_width-1 && y==source_height-1 && maskIn[id]){
                        boundryPixelArray[id]=OUTSIDE;
                    }
                    else if(x==0 && y < source_height-1 && maskIn[id]){
                        boundryPixelArray[id]=OUTSIDE;
                    }
                    else if(x==source_width-1 && y < source_height-1 && maskIn[id]){
                        boundryPixelArray[id]=OUTSIDE;
                    }
                    else if(x < source_width-1 && y==0 && maskIn[id]){
                        boundryPixelArray[id]=OUTSIDE;
                    }
                    else if(x < source_width-1 && y==source_height-1 && maskIn[id]){
                        boundryPixelArray[id]=OUTSIDE;
                    }
                    else{
                        int id_right = x+1 + y*source_width + channel * source_width * source_height;
                        int id_left = x-1 + y*source_width + channel * source_width * source_height;
                        int id_up = x + (y+1)*source_width + channel * source_width * source_height;
                        int id_down = x + (y-1)*source_width + channel * source_width * source_height;

                        if(maskIn[id]>=0.5&&maskIn[id_right]>=0.5&&maskIn[id_left]>=0.5&&maskIn[id_up]>=0.5&&maskIn[id_down]>=0.5){
                            boundryPixelArray[id] = INSIDE_MASK;
                        }
                        else if(maskIn[id]){
                            boundryPixelArray[id] = BOUNDRY;
                        }
                        else{
                            boundryPixelArray[id] = OUTSIDE;
                        }
                    }

                }
            }   
        }

    }
}

//TODO
void extract_boundary_omp_dynamic(float *maskIn, int *boundryPixelArray, int source_nchannel, int source_width, int source_height){
    for(int channel = 0; channel < source_nchannel; channel++){
        for(int y = 0; y < source_height; y++){
            for(int x = 0; x < source_width; x++){
                int id = x + y*source_width + channel * source_width * source_height;
                
                if(x==0 && y==0 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==0 && y==source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==source_width-1 && y==0 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==source_width-1 && y==source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==0 && y < source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x==source_width-1 && y < source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x < source_width-1 && y==0 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else if(x < source_width-1 && y==source_height-1 && maskIn[id]){
                    boundryPixelArray[id]=OUTSIDE;
                }
                else{
                    int id_right = x+1 + y*source_width + channel * source_width * source_height;
                    int id_left = x-1 + y*source_width + channel * source_width * source_height;
                    int id_up = x + (y+1)*source_width + channel * source_width * source_height;
                    int id_down = x + (y-1)*source_width + channel * source_width * source_height;

                    if(maskIn[id]>=0.5&&maskIn[id_right]>=0.5&&maskIn[id_left]>=0.5&&maskIn[id_up]>=0.5&&maskIn[id_down]>=0.5){
                        boundryPixelArray[id] = INSIDE_MASK;
                    }
                    else if(maskIn[id]){
                        boundryPixelArray[id] = BOUNDRY;
                    }
                    else{
                        boundryPixelArray[id] = OUTSIDE;
                    }
                }
            }
        }
    }
}

void merge_without_blend(float *srcimg, float *targetimg, float *outimg, int *boundary_array,int source_nchannel, int source_width, int source_height){
    for(int channel = 0; channel < source_nchannel; channel++){
        for(int y = 0; y < source_height; y++){
            for(int x = 0; x < source_width; x++){
                int id = x + y*source_width + channel * source_width * source_height;
                if(boundary_array[id] == INSIDE_MASK){
                    outimg[id] = targetimg[id];
                }
                else{
                    outimg[id] = srcimg[id];
                }
            }
        }
    }
}

void merge_without_blend_omp_static(float *srcimg, float *targetimg, float *outimg, int *boundary_array,int source_nchannel, int source_width, int source_height){
    #if OMP
    #pragma omp parallel
    #endif
    {
        int t = omp_get_thread_num();
        int x_each = source_width/4;
        int y_each = source_height/3;
        int y_begin = (t/4)*y_each;
        int y_end = y_begin+y_each;
        int x_begin = (t%4)*x_each;
        int x_end = x_begin+x_each;
    
        //int tcount = omp_get_num_threads(); we use 12 threads in this experiment
        for(int channel = 0; channel < source_nchannel; channel++){
            for(int y = y_begin; y<y_end && y < source_height; y++){
                for(int x = x_begin; x<x_end && x < source_width; x++){
                    int id = x + y*source_width + channel * source_width * source_height;
                    if(boundary_array[id] == INSIDE_MASK){
                        outimg[id] = targetimg[id];
                    }
                    else{
                        outimg[id] = srcimg[id];
                    }
                }
            }
        }
    }
}

void merge_without_blend_omp_dynamic(float *srcimg, float *targetimg, float *outimg, int *boundary_array,int source_nchannel, int source_width, int source_height){
    #if OMP
    #pragma omp parallel
    #endif
    {
        int t = omp_get_thread_num();
        int x_each = source_width/4;
        int y_each = source_height/3;
        int y_begin = (t/4)*y_each;
        int y_end = y_begin+y_each;
        int x_begin = (t%4)*x_each;
        int x_end = x_begin+x_each;
    
        //int tcount = omp_get_num_threads(); we use 12 threads in this experiment
        for(int channel = 0; channel < source_nchannel; channel++){
            for(int y = y_begin; y<y_end && y < source_height; y++){
                for(int x = x_begin; x<x_end && x < source_width; x++){
                    int id = x + y*source_width + channel * source_width * source_height;
                    if(boundary_array[id] == INSIDE_MASK){
                        outimg[id] = targetimg[id];
                    }
                    else{
                        outimg[id] = srcimg[id];
                    }
                }
            }
        }
    }
}

void poisson_jacobi(float *targetimg, float *outimg, 
    int *boundary_array,int c, int w, 
    int h, int boundBoxMinX, int boundBoxMaxX, 
    int boundBoxMinY, int boundBoxMaxY){
    for(int i=0; i<ITERATIONS; i++){
        //printf("%d iteration\n", i);
        for(int channel = 0; channel < c; channel++){
            for(int y = boundBoxMinY; y <= boundBoxMaxY; y++){
                for(int x = boundBoxMinX; x <= boundBoxMaxX; x++){
                    int id = x + y*w + channel * w * h;
                    int idx_nextX = x+1 + w*y +w*h*channel;
                    int idx_prevX = x-1 + w*y + w*h*channel;
                    int idx_nextY = x + w*(y+1) +w*h*channel;
                    int idx_prevY = x + w*(y-1) +w*h*channel;
                    //printf("id: %d, idx_nextX: %d, idx_prevX: %d, idx_nextY: %d, idx_prevY: %d\n", id, idx_nextX, idx_prevX, idx_nextY, idx_prevY);
                    if(boundary_array[id] == INSIDE_MASK){
                        double neighbor_target = targetimg[idx_nextY]+targetimg[idx_nextX]+targetimg[idx_prevX]+targetimg[idx_prevY];
                        double neighbor_output = outimg[idx_nextY]+outimg[idx_nextX]+outimg[idx_prevX]+outimg[idx_prevY];
                        outimg[id] = 0.25*(4*targetimg[id]-neighbor_target + neighbor_output);
                    }
                }
            }
        }
    }
}

void poisson_jacobi_omp_static(float *targetimg, float *outimg, 
    int *boundary_array,int c, int w, 
    int h, int boundBoxMinX, int boundBoxMaxX, 
    int boundBoxMinY, int boundBoxMaxY){
    
    #if OMP
    #pragma omp parallel
    #endif
    {
        int t = omp_get_thread_num();
        int x_each = (boundBoxMaxX-boundBoxMinX+1)/4;
        int y_each = (boundBoxMaxY-boundBoxMinY+1)/3;
        int y_begin = (t/4)*y_each + boundBoxMinY;
        int y_end = y_begin+y_each + boundBoxMinY;
        int x_begin = (t%4)*x_each + boundBoxMinX;
        int x_end = x_begin+x_each + boundBoxMinX;

    
        for(int i=0; i<ITERATIONS; i++){
            //printf("%d iteration\n", i);
            for(int channel = 0; channel < c; channel++){
                for(int y = y_begin; y < y_end && y<boundBoxMaxY; y++){
                    for(int x = x_begin; x < x_end && x<boundBoxMaxX; x++){
                        int id = x + y*w + channel * w * h;
                        int idx_nextX = x+1 + w*y +w*h*channel;
                        int idx_prevX = x-1 + w*y + w*h*channel;
                        int idx_nextY = x + w*(y+1) +w*h*channel;
                        int idx_prevY = x + w*(y-1) +w*h*channel;
                        //printf("id: %d, idx_nextX: %d, idx_prevX: %d, idx_nextY: %d, idx_prevY: %d\n", id, idx_nextX, idx_prevX, idx_nextY, idx_prevY);
                        if(boundary_array[id] == INSIDE_MASK){
                            double neighbor_target = targetimg[idx_nextY]+targetimg[idx_nextX]+targetimg[idx_prevX]+targetimg[idx_prevY];
                            double neighbor_output = outimg[idx_nextY]+outimg[idx_nextX]+outimg[idx_prevX]+outimg[idx_prevY];
                            outimg[id] = 0.25*(4*targetimg[id]-neighbor_target + neighbor_output);
                        }
                    }
                }
            }
            #if OMP
            #pragma omp barrier
            #endif
        }
    }
}

void poisson_jacobi_omp_dynamic(float *targetimg, float *outimg, 
    int *boundary_array,int c, int w, 
    int h, int boundBoxMinX, int boundBoxMaxX, 
    int boundBoxMinY, int boundBoxMaxY){
    
    #if OMP
    #pragma omp parallel
    #endif
    {
        int t = omp_get_thread_num();
        int x_each = (boundBoxMaxX-boundBoxMinX+1)/4;
        int y_each = (boundBoxMaxY-boundBoxMinY+1)/3;
        int y_begin = (t/4)*y_each + boundBoxMinY;
        int y_end = y_begin+y_each + boundBoxMinY;
        int x_begin = (t%4)*x_each + boundBoxMinX;
        int x_end = x_begin+x_each + boundBoxMinX;

    
        for(int i=0; i<ITERATIONS; i++){
            //printf("%d iteration\n", i);
            for(int channel = 0; channel < c; channel++){
                for(int y = y_begin; y < y_end; y++){
                    for(int x = x_begin; x < x_end; x++){
                        int id = x + y*w + channel * w * h;
                        int idx_nextX = x+1 + w*y +w*h*channel;
                        int idx_prevX = x-1 + w*y + w*h*channel;
                        int idx_nextY = x + w*(y+1) +w*h*channel;
                        int idx_prevY = x + w*(y-1) +w*h*channel;
                        //printf("id: %d, idx_nextX: %d, idx_prevX: %d, idx_nextY: %d, idx_prevY: %d\n", id, idx_nextX, idx_prevX, idx_nextY, idx_prevY);
                        if(boundary_array[id] == INSIDE_MASK){
                            double neighbor_target = targetimg[idx_nextY]+targetimg[idx_nextX]+targetimg[idx_prevX]+targetimg[idx_prevY];
                            double neighbor_output = outimg[idx_nextY]+outimg[idx_nextX]+outimg[idx_prevX]+outimg[idx_prevY];
                            outimg[id] = 0.25*(4*targetimg[id]-neighbor_target + neighbor_output);
                        }
                    }
                }
            }
            #if OMP
            #pragma omp barrier
            #endif
        }
    }
}

int main(int argc, char **argv)
{
    //int iterations=ITERATIONS;

    string source_image = "";
    string mask = "";
    string target_image = "";

    source_image = argv[1];
    cout<<" source_image   : "<<source_image<<endl;
    
    target_image = argv[2];
    cout<<" target_image   : "<<target_image<<endl;
    
    mask = argv[3];
    cout<<" Mask name   : "<<mask <<endl;

    //load source image
    cv::Mat msourceImage = cv::imread(source_image.c_str(), -1);
    if (msourceImage.data == NULL) { cerr << "ERROR: Could not load source image " << source_image << endl; return 1; }
    cv::Mat mmask = cv::imread(mask.c_str(), -1);
    if (mmask.data == NULL) { cerr << "ERROR: Could not load mask image " << mask << endl; return 1; }
    cv::Mat mtargetImage = cv::imread(target_image.c_str(), -1);
    if (mtargetImage.data == NULL) { cerr << "ERROR: Could not load  image " << mask << endl; return 1; }

    msourceImage.convertTo(msourceImage,CV_32F);
    mtargetImage.convertTo(mtargetImage,CV_32F);
    mmask.convertTo(mmask,CV_32F);

    msourceImage /= 255.f;
    mtargetImage /= 255.f;
    mmask /= 255.f;

    int source_w = msourceImage.cols;         // width
    int source_h = msourceImage.rows;         // height
    int source_nc = msourceImage.channels();  // number of channels
    cout <<endl<<" Source image   : " << source_w << " x " << source_h << " x " <<source_nc<<endl;

    int target_w = mtargetImage.cols;         // width
    int target_h = mtargetImage.rows;         // height
    int target_nc = mtargetImage.channels();  // number of channels
    cout <<endl<<" target image  : " << target_w << " x " << target_h << " x " <<target_nc<<endl;

    int mask_w = mmask.cols;         // width
    int mask_h = mmask.rows;         // height
    int mask_nc = mmask.channels();  // number of channels
    cout <<endl<<" mask          : " << mask_w << " x " << mask_h << " x " <<mask_nc<<endl;

    cv::Mat mOut_seq(target_h,target_w,mtargetImage.type());  

    float *srcimgIn  = new float[(size_t)source_w*source_h*source_nc];
    float *maskIn  = new float[(size_t)mask_w*mask_h*mask_nc];
    float *targetimgIn  = new float[(size_t)target_w*target_h*target_nc];

    convert_interleaved_to_layered (srcimgIn, (float*)msourceImage.data, source_w, source_h, source_nc);
    convert_interleaved_to_layered (maskIn, (float*)mmask.data, mask_w, mask_h, mask_nc);
    convert_interleaved_to_layered(targetimgIn, (float*)mtargetImage.data, target_w, target_h, target_nc);

    int *boundryPixelArray_seq = new int[(size_t)target_w*target_h*mOut_seq.channels()];
    float *imgOut_seq = new float[(size_t)target_w*target_h*mOut_seq.channels()];
    int *boundryPixelArray_openmp = new int[(size_t)target_w*target_h*mOut_seq.channels()];
    float *imgOut_openmp = new float[(size_t)target_w*target_h*mOut_seq.channels()];
    
    // begin sequential part clocking
    clock_t t1 = clock();
    cout<<"begin sequentail"<<endl;
    //get boundary pixel array to indicate which pixel is corner, edge, inside_mask, boundary or just outside
    extract_boundary(maskIn, boundryPixelArray_seq, source_nc, source_w, source_h);
    int boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY; 
    // calculate the bounding box for reducing unnecessary calculation
    cout<<"begin 596"<<endl;
    calculate_boundbox(target_w, target_h, target_nc, boundryPixelArray_seq, &boundBoxMinX, &boundBoxMinY, &boundBoxMaxX, &boundBoxMaxY);
    cout<<"begin 598"<<endl;
    merge_without_blend(srcimgIn, targetimgIn, imgOut_seq, boundryPixelArray_seq, source_nc, source_w, source_h);
    cout<<"begin 600"<<endl;
    poisson_jacobi(targetimgIn, imgOut_seq, boundryPixelArray_seq, source_nc, source_w, source_h, boundBoxMinX, boundBoxMaxX, boundBoxMinY, boundBoxMaxY);
    
    clock_t sequential_time = clock()-t1;
    cout << "time cost for CPU: "<<sequential_time * 1.0 / CLOCKS_PER_SEC * 1000 << endl;
    convert_layered_to_interleaved((float*)mOut_seq.data, imgOut_seq, source_w, source_h, source_nc);
    cv::imwrite("FinalImage_sequential.jpg",mOut_seq*255.f);

    /*-------openmp static---------*/
    omp_set_num_threads(THREAD_COUNT);
    clock_t t2 = clock();
    extract_boundary_omp_static(maskIn, boundryPixelArray_openmp, source_nc, source_w, source_h);
    calculate_boundbox_omp_static(target_w, target_h, target_nc, boundryPixelArray_openmp, &boundBoxMinX, &boundBoxMinY, &boundBoxMaxX, &boundBoxMaxY);
    merge_without_blend_omp_static(srcimgIn, targetimgIn, imgOut_openmp, boundryPixelArray_openmp, source_nc, source_w, source_h);
    poisson_jacobi_omp_static(targetimgIn, imgOut_openmp, boundryPixelArray_openmp, source_nc, source_w, source_h, boundBoxMinX, boundBoxMaxX, boundBoxMinY, boundBoxMaxY);
    clock_t openmp_time_static = clock()-t2;
    cout << "time cost for openmp static: "<<openmp_time_static * 1.0 / CLOCKS_PER_SEC * 1000 << endl;
    cout << "speedup for openmp static: "<<(sequential_time * 1.0 / CLOCKS_PER_SEC * 1000)/(openmp_time_static * 1.0 / CLOCKS_PER_SEC * 1000)<<endl;
    
    convert_layered_to_interleaved((float*)mOut_seq.data, imgOut_openmp, source_w, source_h, source_nc);
    cv::imwrite("FinalImage_omp_static.jpg",mOut_seq*255.f);

    /*---------openmp dynamic---------*/
    //TODO!!!!

    free(srcimgIn);
    free(maskIn);
    free(targetimgIn);
    free(boundryPixelArray_seq);
    free(imgOut_seq);
    free(boundryPixelArray_openmp);
    free(imgOut_openmp);
} 