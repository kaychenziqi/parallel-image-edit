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

#ifndef GPU_SCHEDULE
#define GPU_SCHEDULE 0
#endif

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
    Expr width, Expr height)
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

Tuple propagate(Tuple cur, Tuple left, Tuple up, Func dst, Func src, Expr width, Expr height)
{
    Expr dx = cur[0];
    Expr dy = cur[1];
    Expr sx = cur[2];
    Expr sy = cur[3];
    Expr d = cur[4];
    Expr sx_left = min(width - 1, max(0, left[2] + 1));
    Expr sy_left = left[3];
    Expr sx_up = up[2];
    Expr sy_up = min(height - 1, max(0, up[3] + 1));

    Expr left_d = patch_distance(dst, src, dx, dy, sx_left, sy_left, width, height);
    Expr up_d = patch_distance(dst, src, dx, dy, sx_up, sy_up, width, height);

    Expr new_sx = select(left_d <= d, 
        select(left_d <= up_d, sx_left, sx_up),
        select(d <= up_d, sx, sx_up));

    Expr new_sy = select(up_d <= d, 
        select(up_d <= left_d, sy_up, sy_left),
        select(d <= left_d, sy, sy_left));

    Expr new_d = select(d <= left_d, 
        select(d <= up_d, d, up_d),
        select(left_d <= up_d, left_d, up_d)
    );

    return Tuple(dx, dy, new_sx, new_sy, new_d);
}

Tuple random_search(Tuple cur, Tuple ran, Func dst, Func src, Expr width, Expr height)
{
    Expr dx = cur[0];
    Expr dy = cur[1];
    Expr sx = cur[2];
    Expr sy = cur[3];
    Expr d = cur[4];

    Expr rx = max(0, min(ran[2], width - 1));
    Expr ry = max(0, min(ran[3], height - 1));
    Expr rd = patch_distance(dst, src, dx, dy, rx, ry, width, height);

    Expr new_sx = select(d <= rd, sx, rx);
    Expr new_sy = select(d <= rd, sy, ry);
    Expr new_d = select(d <= rd, d, rd);

    return Tuple(dx, dy, new_sx, new_sy, new_d);
}

Tuple nn_search(Tuple cur, Tuple left, Tuple up,
    Tuple ran1, Tuple ran2, Tuple ran3, Tuple ran4, Tuple ran5,
    Func dst, Func src, Expr width, Expr height)
{
    cur = propagate(cur, left, up, dst, src, width, height);
    // cur = random_search(cur, ran1, dst, src, width, height);
    // cur = random_search(cur, ran2, dst, src, width, height);
    // cur = random_search(cur, ran3, dst, src, width, height);
    // cur = random_search(cur, ran4, dst, src, width, height);
    cur = random_search(cur, ran5, dst, src, width, height);
    return cur;
}

Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Target target = get_host_target();

    std::vector<Target::Feature> features_to_try;
    if (target.os == Target::Windows) {
        // Try D3D12 first; if that fails, try OpenCL.
        if (sizeof(void*) == 8) {
            // D3D12Compute support is only available on 64-bit systems at present.
            features_to_try.push_back(Target::D3D12Compute);
        }
        features_to_try.push_back(Target::OpenCL);
    } else if (target.os == Target::OSX) {
        // OS X doesn't update its OpenCL drivers, so they tend to be broken.
        // CUDA would also be a fine choice on machines with NVidia GPUs.
        features_to_try.push_back(Target::Metal);
    } else {
        features_to_try.push_back(Target::OpenCL);
    }
    // Uncomment the following lines to also try CUDA:
    // features_to_try.push_back(Target::CUDA);

    for (Target::Feature f : features_to_try) {
        Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }
}

void do_patchmatch(std::string input_file, std::string src_file, std::string output_file) 
{
    Var x("x"), y("y"), c("c");
    
    Buffer<uint8_t> dst_u8 = load_image(input_file.c_str());
    Buffer<uint8_t> src_u8 = load_image(src_file.c_str());

    int width = dst_u8.width();
    int height = dst_u8.height();

    Func dst_f("dst_f"), src_f("src_f");
    dst_f(x, y, c) = cast<float>(dst_u8(x, y, c)) / 255.f;
    src_f(x, y, c) = cast<float>(src_u8(x, y, c)) / 255.f;

    Func dst("dst"), src("src");
    dst(x, y) = Tuple(dst_f(x, y, 0), dst_f(x, y, 1), dst_f(x, y, 2));
    src(x, y) = Tuple(src_f(x, y, 0), src_f(x, y, 1), src_f(x, y, 2));

    double t1 = CycleTimer::currentSeconds();

    Func map("map");
    Var t;
    map(x, y, t) = MapEntry(x, y, 
        random_int() % width, 
        random_int() % height, 
        FLT_MAX);

    RDom r(0, width, 0, height, 1, NUM_ITERATIONS);
    map(r.x, r.y, r.z) = nn_search(
        map(r.x, r.y, r.z - 1), 
        map(r.x - 1, r.y, r.z - 1),
        map(r.x, r.y - 1, r.z - 1),
        map(r.x + (random_int() % 10 - 5), r.y + (random_int() % 10 - 5), r.z - 1),
        map(r.x + (random_int() % 18 - 9), r.y + (random_int() % 18 - 9), r.z - 1),
        map(r.x + (random_int() % 24 - 12), r.y + (random_int() % 24 - 12), r.z - 1),
        map(r.x + (random_int() % 28 - 14), r.y + (random_int() % 28 - 14), r.z - 1),
        map(r.x + (random_int() % 30 - 15), r.y + (random_int() % 30 - 15), r.z - 1),
        dst, src, width, height);

    Func remap("remap");
    remap(x, y, c) = src_f(
        MapEntry(map(x, y, NUM_ITERATIONS)).get_sx() % width, 
        MapEntry(map(x, y, NUM_ITERATIONS)).get_sy() % height, 
        c);

    Func output("output");
    output(x, y, c) = cast<uint8_t>(remap(x, y, c) * 255);

    if (GPU_SCHEDULE) {
        printf("Using GPU schedule\n");

        Var xa, ya, xb, yb;
        Var xc, yc, xd, yd;
        Var xe, ye, xf, yf;
        Expr block_size = 16;

        dst.compute_root();
        src.compute_root();

        map.compute_root();  
        map.gpu_tile(x, y, xa, ya, xb, yb, block_size, block_size);

        remap.compute_root();
        remap.gpu_tile(x, y, xc, yc, xd, yd, block_size, block_size);

        output.gpu_tile(x, y, xe, ye, xf, yf, block_size, block_size);
        output.compile_jit(find_gpu_target());
    }
    else {
        printf("Using CPU schedule\n");

        map.compute_root()
            .parallel(y)
            .parallel(x);
        output.compute_root()
            .parallel(y)
            .parallel(x);
    }

    Buffer<uint8_t> result(width, height, 3);
    result.set_min(0, 0);
    output.realize(result);

    double t2 = CycleTimer::currentSeconds();
    double time_elasped = (t2 - t1);
    printf("Time: %.4f\n", time_elasped);

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

    do_patchmatch(input_file, src_file, output_file);

    return 0;
}