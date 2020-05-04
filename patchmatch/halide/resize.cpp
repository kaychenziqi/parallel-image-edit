#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void do_resize(string input_file, string src_file, string output_file, 
    int width, int height, int half_patch) 
{
    Mat srcMat, srcMat2;
    Mat dstMat, dstMat2;
    Mat outputMat;

    srcMat = imread(src_file, IMREAD_COLOR);
    dstMat = imread(input_file, IMREAD_COLOR);

    if (width == -1) width = dstMat.cols;
    if (height == -1) height = dstMat.rows;

    // #if DEBUG
    cout << "Width: " << width << endl;
    cout << "Height: " << height << endl;
    cout << "HalfPatch: " << half_patch << endl;
    // #endif

    resize(srcMat, srcMat2, Size(width, height));
    resize(dstMat, dstMat2, Size(width, height));

    imwrite(input_file + "-resize.jpg", dstMat2);
    imwrite(src_file + "-resize.jpg", srcMat2);
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

    do_resize(input_file, src_file, output_file, 
        width, height, half_patch);

    return 0;
}