#include <stdio.h>
#include <stdlib.h>
#include <retarget.h>
#include <opencv2/opencv.hpp>

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

int main(int argc, char** argv) {
    if (argc < 5) {
        cout << "Usage: ./retarget <input_path> <output_path> <target_height> <target_width>" << endl;
        exit(0);
    }

    string input_file = argv[1];
    string output_file = argv[2];
    int height = atoi(argv[3]);
    int width = atoi(argv[4]);

    display_image(input_file);
    do_retarget(input_file, output_file, height, width);
    do_resize(input_file, output_file + "-resize.jpg", height, width);

    return 0;
}