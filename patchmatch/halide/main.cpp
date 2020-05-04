#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <float.h>

#include "Halide.h"
#include "halide_image_io.h"
#include "CycleTimer2.h"

using namespace Halide;
using namespace Halide::Tools;

#define HALF_PATCH 7
#define NUM_ITERATIONS 10
#define RADIUS 5
#define STRIDE 1

struct MapEntry {
    Expr dx;
    Expr dy;
    Expr sx;
    Expr sy;
    Expr d;

    MapEntry(Tuple t) 
    : dx(t[0]), dy(t[1]), sx(t[2]), sy(t[2]), d(t[4]) {}

    MapEntry(Expr dx_, Expr dy_, Expr sx_, Expr sy_, Expr d_) 
    : dx(dx_), dy(dy_), sx(sx_), sy(sy_), d(d_) {}

    MapEntry(FuncRef t) : MapEntry(Tuple(t)) {}

    operator Tuple() const { return {dx, dy, sx, sy, d}; }

    Expr get_dx() { return dx; }

    Expr get_dy() { return dy; }

    Expr get_sx() { return sx; }

    Expr get_sy() { return sy; }

    Expr get_d() { return d; }
};

inline Expr sum_squared_diff(Tuple a, Tuple b)
{
    return sqrt(
        (a[0] - b[0]) * (a[0] - b[0]) +
        (a[1] - b[1]) * (a[1] - b[1]) +
        (a[2] - b[2]) * (a[2] - b[2])
    );
}

Expr patch_distance(Func dst, Func src, Expr dx, Expr dy, Expr sx, Expr sy, 
    int width, int height)
{
    Expr distance(0.f);
    for (int j = -HALF_PATCH; j <= HALF_PATCH; j++) {
        for (int i = -HALF_PATCH; i <= HALF_PATCH; i++) {
            Expr dx1 = min(width - 1, max(0, dx + i));
            Expr dy1 = min(height - 1, max(0, dy + i));
            Expr sx1 = min(width - 1, max(0, sx + i));
            Expr sy1 = min(height - 1, max(0, sy + i));

            Tuple dpixel = dst(dx1, dy1);
            Tuple spixel = src(sx1, sy1);
            distance += sum_squared_diff(dpixel, spixel);
        }
    }
    return distance;
}

Tuple propagate(Tuple cur, Func dst, Func src, int width, int height)
{
    Expr dx = cur[0];
    Expr dy = cur[1];
    Expr sx = cur[2];
    Expr sy = cur[3];
    Expr d = cur[4];

    Expr left_d = patch_distance(dst, src, dx, dy, sx - STRIDE, sy, width, height);
    Expr up_d = patch_distance(dst, src, dx, dy, sx, sy - STRIDE, width, height);

    Expr new_sx = select(left_d <= d, 
        select(left_d <= up_d, sx - STRIDE, sx),
        sx);

    Expr new_sy = select(up_d <= d, 
        select(up_d <= left_d, sy - STRIDE, sy),
        sy);

    Expr new_d = select(d <= left_d, 
        select(d <= up_d, d, up_d),
        select(left_d <= up_d, left_d, up_d)
    );

    return Tuple(dx, dy, new_sx, new_sy, new_d);
}

Tuple random_search(Tuple cur, Func dst, Func src, int width, int height, int radius)
{
    Expr dx = cur[0];
    Expr dy = cur[1];
    Expr sx = cur[2];
    Expr sy = cur[3];
    Expr d = cur[4];

    Expr rand_x = random_int() % (2 * radius) - radius;
    Expr rand_y = random_int() % (2 * radius) - radius;

    Expr rx = max(0, min(sx + rand_x, width));
    Expr ry = max(0, min(sy + rand_y, height));
    Expr rd = patch_distance(dst, src, dx, dy, rx, ry, width, height);

    Expr new_sx = select(d <= rd, sx, rx);
    Expr new_sy = select(d <= rd, sy, ry);
    Expr new_d = select(d <= rd, d, rd);

    return Tuple(dx, dy, new_sx, new_sy, new_d);
}

Tuple nn_search(Tuple cur, Func dst, Func src, int width, int height)
{
    cur = propagate(cur, dst, src, width, height);
    for (int radius = RADIUS; radius >= 1; radius--) {
        cur = random_search(cur, dst, src, width, height, radius);
    }
    return cur;
}

void do_patchmatch(std::string input_file, std::string src_file, std::string output_file) 
{
    Var x("x"), y("y"), c("c");
    
    Buffer<uint8_t> dst_u8 = load_image(input_file.c_str());
    Buffer<uint8_t> src_u8 = load_image(src_file.c_str());

    int width = dst_u8.width();
    int height = dst_u8.height();

    Func dst_f("dst_f"), src_f("src_f");
    dst_f(x, y, c) = cast<float>(dst_u8(x, y, c));
    src_f(x, y, c) = cast<float>(src_u8(x, y, c));

    Func dst("dst"), src("src");
    dst(x, y) = Tuple(dst_f(x, y, 0), dst_f(x, y, 1), dst_f(x, y, 2));
    src(x, y) = Tuple(src_f(x, y, 0), src_f(x, y, 1), src_f(x, y, 2));

    Func map("map");
     
    Var t;
    map(x, y, t) = MapEntry(x, y, 
        random_int() % width, 
        random_int() % height, 
        FLT_MAX);

    RDom iter(1, NUM_ITERATIONS);
    map(x, y, iter) = nn_search(map(x, y, iter - 1), dst, src, width, height);

    Func remap("remap");
    remap(x, y, c) = src_f(
        MapEntry(map(x, y, NUM_ITERATIONS - 1)).get_sx() % width, 
        MapEntry(map(x, y, NUM_ITERATIONS - 1)).get_sy() % height, 
        c);

    Func output("output");
    output(x, y, c) = cast<uint8_t>(remap(x, y, c));

    Buffer<uint8_t> result(width - STRIDE, height - STRIDE, 3);
    result.set_min(STRIDE, STRIDE);
    output.realize(result);

    save_image(result, output_file.c_str());
    printf("Success!\n");
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
        printf("Missing src_u8 file\n");
    }
    if (input_file == "") {
        printf("Missing dst_u8 file\n");
    }
    if (output_file == "") {
        printf("Missing output file\n");
    }

    double t1 = CycleTimer::currentSeconds();
    do_patchmatch(input_file, src_file, output_file);
    double t2 = CycleTimer::currentSeconds();
    double time_elasped = (t2 - t1);
    printf("Time: %.4f\n", time_elasped);

    return 0;
}