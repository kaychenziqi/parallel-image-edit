#include "Halide.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

using namespace Halide;
using namespace std;
#define ITERATIONS 100
class AutoScheduled : public Halide::Generator<AutoScheduled> {
public:
    Input<Buffer<float>> image_to_blend_f{"image_to_blend_f", 3};
    Input<Buffer<float>>  target_value_f{"target_value_f", 3};

    Output<Buffer<uint8_t>> output{"output", 3};

    void generate() {
        Func poisson_jacobi;
        Func tmp_poisson_jacobi;
        cout<<"get here"<<endl;
        // tmp_poisson_jacobi(x,y,c) = image_to_blend_f(x,y,c);
        // poisson_jacobi(x,y,c) = 0.25f * target_value_f(x,y,c) + 0.25f * 
        //                         (tmp_poisson_jacobi(min(x+1,image_to_blend_f.width()-1), y, c) +
        //                          tmp_poisson_jacobi(max(x-1,0), y, c) +
        //                          tmp_poisson_jacobi(x, min(y+1, image_to_blend_f.height()-1), c) +
        //                          tmp_poisson_jacobi(x, max(y-1,0), c));
    // RDom r(1, 64, 1, 59, 0, 2, 1, ITERATIONS);
    // Var t;
    // poisson_jacobi(x,y,c,t) =  Halide::cast<float>(image_to_blend_f(x,y,c));
    // poisson_jacobi(r.x, r.y, r.z, r.w) = 0.25f * target_value_f(r.x, r.y, r.z)
    //         //  +0.25f * (poisson_jacobi(max(r.x-1,0),r.y,r.z,r.w-1)
    //         //  +poisson_jacobi(r.x,max(r.y-1,0),r.z,r.w-1)
    //         //  +poisson_jacobi(min(r.x+1,image_to_blend_f.width()-1),r.y,r.z,r.w-1)
    //         //  +poisson_jacobi(r.x,min(r.y+1, image_to_blend_f.height()-1),r.z,r.w-1));
    //         +0.25f * (poisson_jacobi(r.x-1,r.y,r.z,r.w-1)
    //          +poisson_jacobi(r.x,r.y-1,r.z,r.w-1)
    //          +poisson_jacobi(r.x+1,r.y,r.z,r.w-1)
    //          +poisson_jacobi(r.x,r.y+1,r.z,r.w-1));
    //     cout<<"get 24"<<endl;
    // //    output(x,y,c) = cast<uint8_t>(poisson_jacobi(x,y,c));
    //     output(x,y,c) = cast<uint8_t>(poisson_jacobi(x,y,c, ITERATIONS));
    // }

    RDom r(1, 64, 1, 59, 1, ITERATIONS);
    
    poisson_jacobi(x,y,c,t) =  Halide::cast<float>(image_to_blend_f(x,y,c));
    poisson_jacobi(r.x, r.y, c, r.z) = 0.25f * target_value_f(r.x, r.y, c)
            +0.25f * (poisson_jacobi(r.x-1,r.y,c,r.z-1)
             +poisson_jacobi(r.x,r.y-1,c,r.z-1)
             +poisson_jacobi(r.x+1,r.y,c,r.z-1)
             +poisson_jacobi(r.x,r.y+1,c,r.z-1));
        cout<<"get 24"<<endl;
    //    output(x,y,c) = cast<uint8_t>(poisson_jacobi(x,y,c));
        output(x,y,c) = cast<uint8_t>(poisson_jacobi(x,y,c, ITERATIONS));
    }

    void schedule() {
        if (auto_schedule) {
            image_to_blend_f.set_estimates({{0, 66}, {0, 60}, {0, 3}});
            target_value_f.set_estimates({{0, 66}, {0, 60}, {0, 3}});
            output.set_estimates({{0, 66}, {0, 60}, {0, 3}});
        }else{
            // Manual schedule without anything: 9941.38ms
            // Manual schedule with compute root and parallel c: 3322.85ms
            // Manual schedule with compute root and parallel x,y,c: 2280.74ms
            // don't know whether we can use parallel to compare with cpu
             output.compute_root()
                    .parallel(c)
                    .parallel(x)
                    .parallel(y);
        }
    }
private:
    Var x{"x"}, y{"y"}, c{"c"},t{"t"};
    Func poisson_jacobi;
};

HALIDE_REGISTER_GENERATOR(AutoScheduled, auto_schedule_gen)


