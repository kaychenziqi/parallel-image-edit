#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <opencv2/opencv.hpp>

#include "retarget.h"
#include "patchmatch.h"

#if OMP
#include <omp.h>
#else
#include "fake_omp.h"
#endif

using namespace std;
using namespace cv;

void display_image(string imgfile) {
    Mat img;
    img = imread(imgfile, IMREAD_COLOR);
    imshow(imgfile, img);
}

void do_retarget(string input_file, string output_file, int height, int width) {
    Mat src = imread(input_file, IMREAD_COLOR);
    Mat dst;
    retarget(src, dst, height, width);
    imwrite(output_file, dst);
}

void do_resize(string input_file, string output_file, int height, int width) {
    Mat src = imread(input_file, IMREAD_COLOR);
    Mat dst;
    resize(src, dst, Size(width, height));
    imwrite(output_file, dst);
}

void do_patchmatch(string input_file, string src_file, string output_file, 
    int width, int height, int half_patch) 
{
    Mat src = imread(src_file, IMREAD_COLOR);
    Mat dst = imread(input_file, IMREAD_COLOR);

    Mat src2, dst2;
    resize(src, src2, Size(width, height));
    resize(dst, dst2, Size(width, height));

    patchmatch(src2, dst2, half_patch);

    // Mat output;
    // resize(dst2, output, Size(dst.cols, dst.rows));
    imwrite(output_file, dst2);
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
    int width = 224;
    int height = 224;
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
    // do_retarget(input_file, output_file, height, width);
    // do_resize(input_file, output_file + "-resize.jpg", height, width);
    do_patchmatch(input_file, src_file, output_file, 
        width, height, half_patch);

    return 0;
}