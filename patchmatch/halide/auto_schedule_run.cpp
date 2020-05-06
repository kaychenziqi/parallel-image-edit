// Halide tutorial lesson 21: Auto-Scheduler

// Before reading this file, see lesson_21_auto_scheduler_generate.cpp

// This is the code that actually uses the Halide pipeline we've
// compiled. It does not depend on libHalide, so we won't be including
// Halide.h.
//
// Instead, it depends on the header files that lesson_21_auto_scheduler_generator produced.
#include "auto_schedule_false.h"
#include "auto_schedule_true.h"

// We'll use the Halide::Runtime::Buffer class for passing data into and out of
// the pipeline.
#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include "Halide.h"
#include "halide_image_io.h"


#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <float.h>

using namespace Halide;
using namespace Halide::Tools;

#define HALF_PATCH 7
#define NUM_ITERATIONS 10
#define RADIUS 5
#define STRIDE 1

void do_patchmatch(std::string input_file, std::string src_file, std::string output_file) 
{
    // Let's declare and initialize the input images
    Halide::Runtime::Buffer<uint8_t> dst_u8 = load_image(input_file.c_str());
    Halide::Runtime::Buffer<uint8_t> src_u8 = load_image(src_file.c_str());

    int width = dst_u8.width();
    int height = dst_u8.height();
    Halide::Runtime::Buffer<uint8_t> output(width - STRIDE, height - STRIDE, 3);

    double auto_schedule_off = Halide::Tools::benchmark(1, 2, [&]() {
        auto_schedule_false(dst_u8, src_u8, output);
    });
    printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);

    double auto_schedule_on = Halide::Tools::benchmark(1, 2, [&]() {
        auto_schedule_true(dst_u8, src_u8, output);
    });
    printf("Auto schedule: %gms\n", auto_schedule_on * 1e3);

    save_image(output, output_file.c_str());
}


int main(int argc, char** argv) {
    std::string input_file = "";
    std::string src_file = "";
    std::string output_file = "";
    int width = -1;
    int height = -1;
    int half_patch = 1;

    int c;
    std::string optstring = "s:i:o:w:h:p:";
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
                // usage(argv[0]);
        }
    }

    if (src_file == "") {
        printf("Missing src file\n");
    }
    if (input_file == "") {
        printf("Missing dst file\n");
    }
    if (output_file == "") {
        printf("Missing output file\n");
    }

    do_patchmatch(input_file, src_file, output_file);

    return 0;
}