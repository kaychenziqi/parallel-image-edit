// Halide tutorial lesson 21: Auto-Scheduler

// So far we have written Halide schedules by hand, but it is also possible to
// ask Halide to suggest a reasonable schedule. We call this auto-scheduling.
// This lesson demonstrates how to use the auto-scheduler to generate a
// copy-pasteable CPU schedule that can be subsequently improved upon.

// On linux or os x, you can compile and run it like so:

// g++ lesson_21_auto_scheduler_generate.cpp ../tools/GenGen.cpp -g -std=c++11 -fno-rtti -I ../include -L ../bin -lHalide -lpthread -ldl -o lesson_21_generate
// export LD_LIBRARY_PATH=../bin   # For linux
// export DYLD_LIBRARY_PATH=../bin # For OS X
// ./lesson_21_generate -o . -g auto_schedule_gen -f auto_schedule_false -e static_library,h,schedule target=host auto_schedule=false
// ./lesson_21_generate -o . -g auto_schedule_gen -f auto_schedule_true -e static_library,h,schedule target=host auto_schedule=true machine_params=32,16777216,40
// g++ lesson_21_auto_scheduler_run.cpp -std=c++11 -I ../include -I ../tools auto_schedule_false.a auto_schedule_true.a -ldl -lpthread -o lesson_21_run
// ./lesson_21_run

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_lesson_21_auto_scheduler_run
// in a shell with the current directory at the top of the halide
// source tree.

#include "Halide.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

using namespace Halide;

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


// We will define a generator to auto-schedule.
class AutoScheduled : public Halide::Generator<AutoScheduled> {
public:
    Input<Buffer<uint8_t>>  dst_u8{"dst_u8", 3};
    Input<Buffer<uint8_t>>  src_u8{"src_u8", 3};

    Output<Buffer<uint8_t>> output{"output", 3};

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

    Tuple propagate(Tuple cur, Func dst, Func src, Expr width, Expr height)
    {
        Expr dx = cur[0];
        Expr dy = cur[1];
        Expr sx = cur[2];
        Expr sy = cur[3];
        Expr d = cur[4];
        Expr sx_left = min(width - 1, max(0, sx - STRIDE));
        Expr sy_up = min(height - 1, max(0, sy - STRIDE));

        Expr left_d = patch_distance(dst, src, dx, dy, sx_left, sy, width, height);
        Expr up_d = patch_distance(dst, src, dx, dy, sx, sy_up, width, height);

        Expr new_sx = select(left_d <= d, 
            select(left_d <= up_d, sx_left, sx),
            sx);

        Expr new_sy = select(up_d <= d, 
            select(up_d <= left_d, sy_up, sy),
            sy);

        Expr new_d = select(d <= left_d, 
            select(d <= up_d, d, up_d),
            select(left_d <= up_d, left_d, up_d)
        );

        return Tuple(dx, dy, new_sx, new_sy, new_d);
    }

    Tuple random_search(Tuple cur, Func dst, Func src, Expr width, Expr height, int radius)
    {
        Expr dx = cur[0];
        Expr dy = cur[1];
        Expr sx = cur[2];
        Expr sy = cur[3];
        Expr d = cur[4];

        Expr rand_x = random_int() % (2 * radius) - radius;
        Expr rand_y = random_int() % (2 * radius) - radius;

        Expr rx = max(0, min(sx + rand_x, width - 1));
        Expr ry = max(0, min(sy + rand_y, height - 1));
        Expr rd = patch_distance(dst, src, dx, dy, rx, ry, width, height);

        Expr new_sx = select(d <= rd, sx, rx);
        Expr new_sy = select(d <= rd, sy, ry);
        Expr new_d = select(d <= rd, d, rd);

        return Tuple(dx, dy, new_sx, new_sy, new_d);
    }

    Tuple nn_search(Tuple cur, Func dst, Func src, Expr width, Expr height)
    {
        cur = propagate(cur, dst, src, width, height);
        for (int radius = RADIUS; radius >= 1; radius--) {
            cur = random_search(cur, dst, src, width, height, radius);
        }
        return cur;
    }

    void generate() {
        Expr width = dst_u8.width();
        Expr height = dst_u8.height();

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

        // Func output("output");
        output(x, y, c) = cast<uint8_t>(remap(x, y, c));
    }

    void schedule() {
        if (auto_schedule) {
            // The auto-scheduler requires estimates on all the input/output
            // sizes and parameter values in order to compare different
            // alternatives and decide on a good schedule.

            // To provide estimates (min and extent values) for each dimension
            // of the input images ('input', 'filter', and 'bias'), we use the
            // set_estimates() method. set_estimates() takes in a list of
            // (min, extent) of the corresponding dimension as arguments.
            dst_u8.set_estimates({{0, 1024}, {0, 1024}, {0, 3}});
            src_u8.set_estimates({{0, 1024}, {0, 1024}, {0, 3}});

            // To provide estimates on the parameter values, we use the
            // set_estimate() method.
            // factor.set_estimate(2.0f);

            // To provide estimates (min and extent values) for each dimension
            // of pipeline outputs, we use the set_estimates() method. set_estimates()
            // takes in a list of (min, extent) for each dimension.
            output.set_estimates({{0, 1024}, {0, 1024}, {0, 3}});

            // Technically, the estimate values can be anything, but the closer
            // they are to the actual use-case values, the better the generated
            // schedule will be.

            // To auto-schedule the the pipeline, we don't have to do anything else:
            // every Generator implicitly has a GeneratorParam named "auto_schedule";
            // if this is set to true, Halide will call auto_schedule() on all of
            // our pipeline's outputs automatically.

            // Every Generator also implicitly has a GeneratorParams named "machine_params",
            // which allows you to specify characteristics of the machine architecture
            // for the auto-scheduler; it's generally specified in your Makefile.
            // If none is specified, the default machine parameters for a generic CPU
            // architecture will be used by the auto-scheduler.

            // Let's see some arbitrary but plausible values for the machine parameters.
            //
            //      const int kParallelism = 32;
            //      const int kLastLevelCacheSize = 16 * 1024 * 1024;
            //      const int kBalance = 40;
            //      MachineParams machine_params(kParallelism, kLastLevelCacheSize, kBalance);
            //
            // The arguments to MachineParams are the maximum level of parallelism
            // available, the size of the last-level cache (in KB), and the ratio
            // between the cost of a miss at the last level cache and the cost
            // of arithmetic on the target architecture, in that order.

            // Note that when using the auto-scheduler, no schedule should have
            // been applied to the pipeline; otherwise, the auto-scheduler will
            // throw an error. The current auto-scheduler cannot handle a
            // partially-scheduled pipeline.

            // If HL_DEBUG_CODEGEN is set to 3 or greater, the schedule will be dumped
            // to stdout (along with much other information); a more useful way is
            // to add "schedule" to the -e flag to the Generator. (In CMake and Bazel,
            // this is done using the "extra_outputs" flag.)

            // The generated schedule that is dumped to file is an actual
            // Halide C++ source, which is readily copy-pasteable back into
            // this very same source file with few modifications. Programmers
            // can use this as a starting schedule and iteratively improve the
            // schedule. Note that the current auto-scheduler is only able to
            // generate CPU schedules and only does tiling, simple vectorization
            // and parallelization. It doesn't deal with line buffering, storage
            // reordering, or factoring reductions.

            // At the time of writing, the auto-scheduler will produce the
            // following schedule for the estimates and machine parameters
            // declared above when run on this pipeline:
            //
            // Var x_i("x_i");
            // Var x_i_vi("x_i_vi");
            // Var x_i_vo("x_i_vo");
            // Var x_o("x_o");
            // Var x_vi("x_vi");
            // Var x_vo("x_vo");
            // Var y_i("y_i");
            // Var y_o("y_o");
            //
            // Func Ix = pipeline.get_func(4);
            // Func Iy = pipeline.get_func(7);
            // Func gray = pipeline.get_func(3);
            // Func harris = pipeline.get_func(14);
            // Func output1 = pipeline.get_func(15);
            // Func output2 = pipeline.get_func(16);
            //
            // {
            //     Var x = Ix.args()[0];
            //     Ix
            //         .compute_at(harris, x_o)
            //         .split(x, x_vo, x_vi, 8)
            //         .vectorize(x_vi);
            // }
            // {
            //     Var x = Iy.args()[0];
            //     Iy
            //         .compute_at(harris, x_o)
            //         .split(x, x_vo, x_vi, 8)
            //         .vectorize(x_vi);
            // }
            // {
            //     Var x = gray.args()[0];
            //     gray
            //         .compute_at(harris, x_o)
            //         .split(x, x_vo, x_vi, 8)
            //         .vectorize(x_vi);
            // }
            // {
            //     Var x = harris.args()[0];
            //     Var y = harris.args()[1];
            //     harris
            //         .compute_root()
            //         .split(x, x_o, x_i, 256)
            //         .split(y, y_o, y_i, 128)
            //         .reorder(x_i, y_i, x_o, y_o)
            //         .split(x_i, x_i_vo, x_i_vi, 8)
            //         .vectorize(x_i_vi)
            //         .parallel(y_o)
            //         .parallel(x_o);
            // }
            // {
            //     Var x = output1.args()[0];
            //     Var y = output1.args()[1];
            //     output1
            //         .compute_root()
            //         .split(x, x_vo, x_vi, 8)
            //         .vectorize(x_vi)
            //         .parallel(y);
            // }
            // {
            //     Var x = output2.args()[0];
            //     Var y = output2.args()[1];
            //     output2
            //         .compute_root()
            //         .split(x, x_vo, x_vi, 8)
            //         .vectorize(x_vi)
            //         .parallel(y);
            // }

        } else {
            // This is where you would declare the schedule you have written by
            // hand or paste the schedule generated by the auto-scheduler.
            // We will use a naive schedule here to compare the performance of
            // the autoschedule with a basic schedule.
        }
    }
private:
    Var x{"x"}, y{"y"}, c{"c"};
};

// As in lesson 15, we register our generator and then compile this
// file along with tools/GenGen.cpp.
HALIDE_REGISTER_GENERATOR(AutoScheduled, auto_schedule_gen)

// After compiling this file, see how to use it in
// lesson_21_auto_scheduler_run.cpp
