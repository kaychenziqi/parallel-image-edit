#include <stdio.h>
#include <stdlib.h>
#include <patchmatch.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void display_image(string imgfile) {
    Mat img;
    img = imread(imgfile, IMREAD_COLOR);
    imshow(imgfile, img);
}

void retarget(string srcfile, string dstfile) {
    Mat img;
    img = imread(srcfile, IMREAD_COLOR);
    imshow(srcfile, img);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Too few parameters" << endl;
        exit(0);
    }

    string img1 = argv[1];
    string img2 = argv[2];

    display_image(img1);

    return 0;
}