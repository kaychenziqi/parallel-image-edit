#include <iostream>
#include "math.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
using namespace std;
#define ITERATIONS 7000

enum pixel_position{INSIDE_MASK, BOUNDRY, OUTSIDE};

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

void extract_boundary(float *maskIn, int *boundryPixelArray, int source_nchannel, int source_width, int source_height){
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

void poisson_jacobi(float *targetimg, float *outimg, 
    int *boundary_array,int c, int w, 
    int h, int boundBoxMinX, int boundBoxMaxX, 
    int boundBoxMinY, int boundBoxMaxY){
    for(int i=0; i<ITERATIONS; i++){
        for(int channel = 0; channel < c; channel++){
            for(int y = boundBoxMinY; y <= boundBoxMaxY; y++){
                for(int x = boundBoxMinX; x <= boundBoxMaxX; x++){
                    int id = x + y*w + channel * w * h;
                    int idx_nextX = x+1 + w*y +w*h*channel;
                    int idx_prevX = x-1 + w*y + w*h*channel;
                    int idx_nextY = x + w*(y+1) +w*h*channel;
                    int idx_prevY = x + w*(y-1) +w*h*channel;
                    
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




int main(int argc, char **argv)
{
    
    int iterations=ITERATIONS;

    string source_image = "";
    string mask = "";
    string target_image = "";

    bool ret;

    source_image = argv[1];
    cout<<" source_image   : "<<source_image<<endl;
    
    target_image = argv[2];
    cout<<" target_image   : "<<target_image<<endl;
    
    mask = argv[3];
    cout<<" Mask name   : "<<mask <<endl;

    // load source, mask, target image
    cv::Mat mSourceImage = cv::imread(source_image.c_str(), -1);
    if (mSourceImage.data == NULL) { cerr << "ERROR: Could not load source image " << source_image << endl; return 1; }
    cv::Mat mmask = cv::imread(mask.c_str(), -1);
    if (mmask.data == NULL) { cerr << "ERROR: Could not load mask image " << mask << endl; return 1; }
    cv::Mat mTargetImage = cv::imread(target_image.c_str(), -1);
    if (mTargetImage.data == NULL) { cerr << "ERROR: Could not load  image " << mask << endl; return 1; }

    // convert to float representation 
    mSourceImage.convertTo(mSourceImage,CV_32F);
    mTargetImage.convertTo(mTargetImage,CV_32F);
    mmask.convertTo(mmask,CV_32F);

    // convert range of each channel to [0,1] 
    mSourceImage /= 255.f;
    mTargetImage /= 255.f;
    mmask /= 255.f;

    // get source image dimensions
    int source_w = mSourceImage.cols;         // width
    int source_h = mSourceImage.rows;         // height
    int source_nc = mSourceImage.channels();  // number of channels
    cout <<endl<<" Source image   : " << source_w << " x " << source_h << " x " <<source_nc<<endl;

    // get target image dimensions
    int target_w = mTargetImage.cols;         // width
    int target_h = mTargetImage.rows;         // height
    int target_nc = mTargetImage.channels();  // number of channels
    cout <<endl<<" target image  : " << target_w << " x " << target_h << " x " <<target_nc<<endl;

    // get source image dimensions
    int mask_w = mmask.cols;         // width
    int mask_h = mmask.rows;         // height
    int mask_nc = mmask.channels();  // number of channels
    cout <<endl<<" mask          : " << mask_w << " x " << mask_h << " x " <<mask_nc<<endl;

    // Output Images
    cv::Mat mOut(target_h,target_w,mTargetImage.type());  

    float *srcimgIn  = new float[(size_t)source_w*source_h*source_nc];
    float *maskIn  = new float[(size_t)mask_w*mask_h*mask_nc];
    float *targetimgIn  = new float[(size_t)target_w*target_h*target_nc];

    convert_interleaved_to_layered (srcimgIn, (float*)mSourceImage.data, source_w, source_h, source_nc);
    convert_interleaved_to_layered (maskIn, (float*)mmask.data, mask_w, mask_h, mask_nc);
    convert_interleaved_to_layered(targetimgIn, (float*)mTargetImage.data, target_w, target_h, target_nc);

    int *boundryPixelArray = new int[(size_t)target_w*target_h*mOut.channels()];
    float *imgOut = new float[(size_t)target_w*target_h*mOut.channels()];

    int boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY; 
    
    extract_boundary(maskIn, boundryPixelArray, source_nc, source_w, source_h);
    calculate_boundbox(target_w, target_h, target_nc, boundryPixelArray, &boundBoxMinX, &boundBoxMinY, &boundBoxMaxX, &boundBoxMaxY);
    merge_without_blend(srcimgIn, targetimgIn, imgOut, boundryPixelArray, source_nc, source_w, source_h);
     
    // begin clocking
    clock_t t1 = clock();
    poisson_jacobi(targetimgIn, imgOut, boundryPixelArray, source_nc, source_w, source_h, boundBoxMinX, boundBoxMaxX, boundBoxMinY, boundBoxMaxY);

    cout << "time cost for CPU: "<<(clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << endl;
    convert_layered_to_interleaved((float*)mOut.data, imgOut, source_w, source_h, source_nc);
    cv::imwrite("FinalImage.jpg",mOut*255.f);
} 