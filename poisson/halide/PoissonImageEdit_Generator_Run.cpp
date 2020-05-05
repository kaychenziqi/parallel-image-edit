#include "auto_schedule_false.h"
#include "auto_schedule_true.h"

// We'll use the Halide::Runtime::Buffer class for passing data into and out of
// the pipeline.
#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "cycletimer.h"


#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <float.h>

using namespace Halide;
using namespace Halide::Tools;
using namespace std;

#define ITERATION 100
enum position_tag {INSIDE_MASK, BOUNDRY, OUTSIDE};
int main(int argc, char** argv) {
    string source_image = "/afs/andrew.cmu.edu/usr18/yuxindin/private/15-618/parallel-image-edit/poisson/input/1/source.png";
    string mask = "/afs/andrew.cmu.edu/usr18/yuxindin/private/15-618/parallel-image-edit/poisson/input/1/mask.png";
    string target_image = "/afs/andrew.cmu.edu/usr18/yuxindin/private/15-618/parallel-image-edit/poisson/input/1/target.png";

    //source_image = argv[1];
    cout<<" source_image   : "<<source_image<<endl;
    
    //target_image = argv[2];
    cout<<" target_image   : "<<target_image<<endl;
    
    //mask = argv[3];
    cout<<" Mask name   : "<<mask <<endl;

    // make clear about the value
    int boundBoxMinX_value = 66;
    int boundBoxMinY_value = 204;
    int boundBoxMaxX_value = 132;
    int boundBoxMaxY_value = 265; 

    Buffer<uint8_t> msourceImage = load_image(source_image);
    Buffer<uint8_t> mtargetImage = load_image(target_image);
    Buffer<uint8_t> mmask = load_image(mask);

    Var x,y,c;
    Expr value_source = msourceImage(x,y,c);
    Expr value_target = mtargetImage(x,y,c);
    Expr value_mask = mmask(x,y,c);

    value_mask = cast<float>(value_mask);
    
    Func extract_boundary;
    Func real_extract_boundary;
    Func merge_without_blend;
    Func tmp_calculate_target;
    Func calculate_target;

    extract_boundary(x, y, c) = value_mask;

    real_extract_boundary(x, y, c) = select(x==0||y==0||x==mmask.width()-1||y==mmask.height()-1, OUTSIDE*1.0f,    
                                        extract_boundary(x,y,c)==255 &&  
                                        extract_boundary(Halide::min(x+1, mmask.width()-1),y,c)==255&& 
                                        extract_boundary(Halide::max(x-1, 0),y,c)==255&&
                                        extract_boundary(x,Halide::min(y+1, mmask.height()-1),c)==255&& 
                                        extract_boundary(x,Halide::max(y-1, 0),c)==255, INSIDE_MASK*1.0f,
                                        extract_boundary(x,y,c)==255, BOUNDRY*1.0f,
                                        OUTSIDE);
    Buffer<float> boundary_array = real_extract_boundary.realize(mmask.width(), mmask.height(), mmask.channels());

    merge_without_blend(x,y,c) = cast<float>(value_source);
    merge_without_blend(x,y,c) = select(boundary_array(x,y,c)==INSIDE_MASK, cast<float>(mtargetImage(x,y,c)), merge_without_blend(x,y,c));
    
    Halide::Runtime::Buffer<float> image_to_blend_f(66, 61, mmask.channels());
    image_to_blend_f.set_min(boundBoxMinX_value+1, boundBoxMinY_value+1);
    merge_without_blend.realize(image_to_blend_f);
    image_to_blend_f.set_min(0,0);


    
    cout<<"get 80"<<endl;

    tmp_calculate_target(x, y, c) = cast<float>(value_target);
    calculate_target(x, y, c) = 4 * tmp_calculate_target(x, y, c) - tmp_calculate_target(x+1, y, c)- tmp_calculate_target(x-1, y, c)- tmp_calculate_target(x, y+1, c)- tmp_calculate_target(x, y-1, c);
    
    Halide::Runtime::Buffer<float> target_value_f(66, 61, mmask.channels());
    target_value_f.set_min(boundBoxMinX_value+1, boundBoxMinY_value+1);
    calculate_target.realize(target_value_f);
    target_value_f.set_min(0,0);

    Halide::Runtime::Buffer<uint8_t> output(66, 61, 3);
    //output(x,y,c) = image_to_blend_f(x,y,c);

    cout<<"begin enter generator"<<endl;

    double auto_schedule_off;
    double total_auto_schedule_off=0;
    double t3 = currentSeconds();
    //for(int x=0; x<ITERATION; x++){
        auto_schedule_off = Halide::Tools::benchmark(2, 3, [&]() {
            auto_schedule_false(image_to_blend_f, target_value_f, output);
        });
    //     total_auto_schedule_off+=auto_schedule_off;
    //     //image_to_blend_f(x,y,c) = output(x,y,c);
    // //}
    printf("Manual schedule: %gms\n", (currentSeconds()-t3)*1000);

    double auto_schedule_on;
    double total_auto_schedule_on=0;
    t3=currentSeconds();
    //for(int x=0; x<ITERATION; x++){
        // auto_schedule_on = Halide::Tools::benchmark(2, 3, [&]() {
        //     auto_schedule_true(image_to_blend_f, target_value_f, output);
            
        // });
        // total_auto_schedule_on += auto_schedule_on;
        //image_to_blend_f(x,y,c) = output(x,y,c);
    //}
    //printf("Auto schedule: %gms\n", (currentSeconds()-t3)*1000);


    save_image(output, "finalimage.png");
}