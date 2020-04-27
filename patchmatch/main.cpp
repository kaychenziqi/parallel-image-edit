#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "util.h"
#include "patchmatch.h"

using namespace std;
using namespace cv;

void display_image(string imgfile) {
    Mat img;
    img = imread(imgfile, IMREAD_COLOR);
    imshow(imgfile, img);
}

void mat_to_uchar3_array(const cv::Mat &mat, uchar3t **arr_ptr)
{
    int ny = mat.rows;
    int nx = mat.cols;
    uchar3t *arr = (uchar3t *) malloc(ny * nx * sizeof(uchar3t));

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

void uchar3_array_to_mat(uchar3t *arr, cv::Mat &mat)
{
    int ny = mat.rows;
    int nx = mat.cols;

    int idx = 0;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            uchar3t pixel = arr[idx];
            mat.at<Vec3b>(y, x)[0] = pixel.x;
            mat.at<Vec3b>(y, x)[1] = pixel.y;
            mat.at<Vec3b>(y, x)[2] = pixel.z;
            idx++;
        }
    }
}

void do_patchmatch(string input_file, string src_file, string output_file, 
    int width, int height, int half_patch) 
{
    Mat srcMat = imread(src_file, IMREAD_COLOR);
    Mat dstMat = imread(input_file, IMREAD_COLOR);

    if (width == -1) width = dstMat.cols;
    if (height == -1) height = dstMat.rows;

    Mat srcMat2, dstMat2;
    resize(srcMat, srcMat2, Size(width, height));
    resize(dstMat, dstMat2, Size(width, height));

    #if DEBUG
    cout << "Width: " << width << endl;
    cout << "Height: " << height << endl;
    cout << "HalfPatch: " << half_patch << endl;
    #endif

    clock_t t1 = clock();
    patchmatch(srcMat2, dstMat2, half_patch);
    clock_t t2 = clock();

    Mat outputMat;
    resize(dstMat2, outputMat, Size(dstMat.cols, dstMat.rows));
    imwrite(output_file, outputMat);

    double time_elasped = (t2 - t1) * 1.0 / CLOCKS_PER_SEC;
    cout << "Time: "<< time_elasped << endl;
}

static void usage(char *name) {
    string use_string = "-s SRC_FILE -i INPUT_FILE -o OUTPUT_FILE [-w WIDTH] [-h HEIGHT] [-p HALF_PATCH]";
    cout << "Usage: " << name << " " << use_string << endl;
    exit(0);
}

int main(int argc, char** argv) {
    string input_file = "";
    string src_file = "";
    string output_file = "";
    int width = -1;
    int height = -1;
    int half_patch = 1;

    int c;
    string optstring = "s:i:o:w:h:p:";
    while ((c = getopt(argc, argv, optstring.c_str())) != -1) {
        switch(c) {
            case 's':
                src_file = optarg;
                break;
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'w':
                width = atoi(optarg);
                break;
            case 'h':
                height = atoi(optarg);
                break;
            case 'p':
                half_patch = atoi(optarg);
                break;
            default:
                printf("Unknown option '%c'\n", c);
                usage(argv[0]);
        }
    }

    if (src_file == "") {
        cout << "Missing src file" << endl;
        usage(argv[0]);
    }
    if (input_file == "") {
        cout << "Missing input file" << endl;
        usage(argv[0]);
    }
    if (output_file == "") {
        cout << "Missing output file" << endl;
        usage(argv[0]);
    }

    // display_image(src_file);
    do_patchmatch(input_file, src_file, output_file, 
        width, height, half_patch);

    return 0;
}