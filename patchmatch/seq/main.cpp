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

void do_convert(const Mat &input, Mat &output, int width, int height)
{
    Mat tmp;
    resize(input, tmp, Size(width, height));
    tmp.convertTo(output, CV_32FC3);
}

void undo_convert(const Mat &input, Mat &output, int width, int height)
{
    Mat tmp;
    input.convertTo(tmp, CV_8UC3);
    resize(tmp, output, Size(width, height));
}

void do_patchmatch(string input_file, string src_file, string output_file, 
    int width, int height, int half_patch) 
{
    Mat srcMat, srcMat2;
    Mat dstMat, dstMat2;
    Mat outputMat;
    float *src, *dst;

    srcMat = imread(src_file, IMREAD_COLOR);
    dstMat = imread(input_file, IMREAD_COLOR);

    if (width == -1) width = dstMat.cols;
    if (height == -1) height = dstMat.rows;

    #if DEBUG
    cout << "Width: " << width << endl;
    cout << "Height: " << height << endl;
    cout << "HalfPatch: " << half_patch << endl;
    #endif

    do_convert(srcMat, srcMat2, width, height);
    do_convert(dstMat, dstMat2, width, height);

    mat_to_array(srcMat2, &src);
    mat_to_array(dstMat2, &dst);

    clock_t t1 = clock();
    patchmatch(src, dst, height, width, half_patch);
    clock_t t2 = clock();

    array_to_mat(dst, dstMat2, height, width, 3);

    undo_convert(dstMat2, outputMat, dstMat.cols, dstMat.rows);
    imwrite(output_file, outputMat);

    double time_elasped = (t2 - t1) * 1.0 / CLOCKS_PER_SEC;
    cout << "Time: "<< time_elasped << endl;

    free(src);
    free(dst);
}

static void usage(char *name) {
    string use_string = "-s SRC_FILE -i INPUT_FILE -o OUTPUT_FILE ";
    use_string += "[-w WIDTH] [-h HEIGHT] [-p HALF_PATCH] [-t THREAD_COUNT]";
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