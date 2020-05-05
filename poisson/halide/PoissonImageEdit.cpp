#include "Halide.h"
#include <stdio.h>
#include "halide_image_io.h"


using namespace Halide::Tools;
using namespace std;

#define ITERATIONS 7000
enum position_tag {INSIDE_MASK, BOUNDRY, OUTSIDE};
int main(int argc, char **argv){
    int iterations=ITERATIONS;

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

    Halide::Buffer<uint8_t> msourceImage = load_image(source_image);
    Halide::Buffer<uint8_t> mtargetImage = load_image(target_image);
    Halide::Buffer<uint8_t> mmask = load_image(mask);

    Halide::Var x, y, c;

    Halide::Expr value_source = msourceImage(x,y,c);
    Halide::Expr value_target = mtargetImage(x,y,c);
    Halide::Expr value_mask = mmask(x,y,c);

    // value_source = Halide::cast<float>(value_source);
    // value_target = Halide::cast<float>(value_target);
    // value_mask = Halide::cast<float>(value_mask);
    

    Halide::Func extract_boundary;
    Halide::Func real_extract_boundary;
    Halide::Func find_boundbox;
    Halide::Func copy_image;
    Halide::Func merge_without_blend;
    Halide::Func poisson_jacobi;
    Halide::Func calculate_target;
    Halide::Func tmp_calculate_target;

    Halide::Func clamped("clamped");

    value_mask = Halide::cast<float>(value_mask);

// I really tried....
    extract_boundary(x, y, c) = value_mask;
    // here may cause some errors!
    real_extract_boundary(x, y, c) = select(x==0||y==0||x==mmask.width()-1||y==mmask.height()-1, OUTSIDE*1.0f,    
                                        extract_boundary(x,y,c)==255 &&  
                                        extract_boundary(Halide::min(x+1, mmask.width()-1),y,c)==255&& 
                                        extract_boundary(Halide::max(x-1, 0),y,c)==255&&
                                        extract_boundary(x,Halide::min(y+1, mmask.height()-1),c)==255&& 
                                        extract_boundary(x,Halide::max(y-1, 0),c)==255, INSIDE_MASK*1.0f,
                                        extract_boundary(x,y,c)==255, BOUNDRY*1.0f,
                                        OUTSIDE);
    Halide::Buffer<float> boundary_array = real_extract_boundary.realize(mmask.width(), mmask.height(), mmask.channels());

    // Halide::Expr boundBoxMinX;
    // Halide::Expr boundBoxMinY;
    // Halide::Expr boundBoxMaxX;
    // Halide::Expr boundBoxMaxY;

    // int boundBoxMinX_value;
    // int boundBoxMinY_value;
    // int boundBoxMaxX_value;
    // int boundBoxMaxY_value;

    // // Halide::RDom r_width(0, mmask.width()-1);
    // // Halide::RDom r_height(0, mmask.height()-1);
    // // Halide::RDom r_channel(0, mmask.channels()-1);
    // Halide::RDom r(0, mmask.width()*mmask.height()-1);
    // find_boundbox() = {mmask.width()-1, mmask.height()-1, 0, 0};

    
    // //r_width = r%mmask.width()
    // //r_height = r/mmask.width()
    // Halide::Expr old_boundBoxMinX = find_boundbox()[0];
    // Halide::Expr old_boundBoxMinY = find_boundbox()[1];
    // Halide::Expr old_boundBoxMaxX = find_boundbox()[2];
    // Halide::Expr old_boundBoxMaxY = find_boundbox()[3];
    // boundBoxMinX = select(old_boundBoxMinX > r%mmask.width() && boundary_array(r%mmask.width(), r/mmask.width(), 0) == BOUNDRY, r%mmask.width(), old_boundBoxMinX);
    // boundBoxMinY = select(old_boundBoxMinY > r/mmask.width() && boundary_array(r%mmask.width(), r/mmask.width(), 0) == BOUNDRY, r/mmask.width(), old_boundBoxMinY);
    // boundBoxMaxX = select(old_boundBoxMaxX < r%mmask.width() && boundary_array(r%mmask.width(), r/mmask.width(), 0) == BOUNDRY, r%mmask.width(), old_boundBoxMaxX);
    // boundBoxMaxY = select(old_boundBoxMaxY < r/mmask.width() && boundary_array(r%mmask.width(), r/mmask.width(), 0) == BOUNDRY, r/mmask.width(), old_boundBoxMaxY);
    // find_boundbox() = {boundBoxMinX, boundBoxMinY, boundBoxMaxX, boundBoxMaxY};

    // Halide::Realization r = find_boundbox.realize();
    // Halide::Buffer<int> r0 = r[0];
    // Halide::Buffer<int> r1 = r[1];
    // Halide::Buffer<int> r2 = r[2];
    // Halide::Buffer<int> r3 = r[3];
    // boundBoxMinX_value = r0(0);
    // boundBoxMinY_value = r1(0);
    // boundBoxMaxX_value = r2(0);
    // boundBoxMaxY_value = r3(0);

    //Halide::Var x_blend,y_blend,c_blend;
    merge_without_blend(x,y,c) = value_source;
    merge_without_blend(x,y,c) = select(boundary_array(x,y,c)==INSIDE_MASK, mtargetImage(x,y,c), merge_without_blend(x,y,c));
    
    cout<<"here!"<<endl;

    tmp_calculate_target(x, y, c) = value_target;
    calculate_target(x, y, c) = 4 * tmp_calculate_target(x, y, c) - tmp_calculate_target(x+1, y, c)- tmp_calculate_target(x-1, y, c)- tmp_calculate_target(x, y+1, c)- tmp_calculate_target(x, y-1, c);
    
    //Halide::RDom r(boundBoxMinX_value, boundBoxMaxX_value, boundBoxMinY_value, boundBoxMaxY_value, 0, mmask.channels()-1,1, ITERATIONS);
    // Halide::RDom r(1, 66, 1, 61, 0, mmask.channels()-1,1, ITERATIONS);
    
    // Halide::Var e,d,f,g;
    // poisson_jacobi(e,d,f,g) =  Halide::cast<float>(merge_without_blend(e, d, f));
    // poisson_jacobi(r.x, r.y, r.z, r.w) = 0.25f * calculate_target(r.x, r.y, r.z)
    //          +0.25f * (poisson_jacobi(r.x+1,r.y,r.z,r.w-1)
    //          +poisson_jacobi(r.x-1,r.y,r.z,r.w)
    //          +poisson_jacobi(r.x,r.y+1,r.z,r.w-1)
    //          +poisson_jacobi(r.x,r.y-1,r.z,r.w));

    Halide::RDom r(1, ITERATIONS, 1, 66, 1, 61, 0, mmask.channels()-1);
    
    Halide::Var e,d,f,g;
    poisson_jacobi(e,d,f,g) =  Halide::cast<float>(merge_without_blend(d, f, g));
    poisson_jacobi(r.x, r.y, r.z, r.w) = 0.25f * calculate_target(r.y, r.z, r.w)
             +0.25f * (poisson_jacobi(r.x-1,r.y+1,r.z,r.w)
             +poisson_jacobi(r.x,r.y-1,r.z,r.w)
             +poisson_jacobi(r.x-1,r.y,r.z+1,r.w)
             +poisson_jacobi(r.x,r.y,r.z-1,r.w));


    //Halide::Var x_jacobi, y_jacobi, c_jacobi;
    // Halide::Var t;
    // Halide::Func tmp_poisson_jacobi;
    
    // poisson_jacobi(x, y, c, t) =  Halide::cast<float>(merge_without_blend(x, y, c));
    // tmp_poisson_jacobi(x, y, c, t) = poisson_jacobi(x, y, c, t);
    
    // //tmp_poisson_jacobi(x, y, c, r) = poisson_jacobi(x, y, c, r-1);
    // poisson_jacobi(x, y, c, r) = 0.25f * calculate_target(x, y, c)
    //         +0.25f * (tmp_poisson_jacobi(x+1,y,c,r)+tmp_poisson_jacobi(x-1,y,c,r)+tmp_poisson_jacobi(x,y+1,c,r)+tmp_poisson_jacobi(x,y-1,c,r));

    // poisson_jacobi(x_jacobi, y_jacobi, c_jacobi) = Halide::cast<float>(merge_without_blend(x_jacobi, y_jacobi, c_jacobi));
    // for(int i=0; i<ITERATIONS; i++){
    //     poisson_jacobi(x_jacobi, y_jacobi, c_jacobi) = 0.25f * calculate_target(x_jacobi, y_jacobi, c_jacobi)
    //         +0.25f * (poisson_jacobi(x_jacobi+1, y_jacobi, c_jacobi)+poisson_jacobi(x_jacobi-1, y_jacobi, c_jacobi)+poisson_jacobi(x_jacobi, y_jacobi+1, c_jacobi)+poisson_jacobi(x_jacobi, y_jacobi-1, c_jacobi));
    // }

    Halide::Func final_image;
    final_image(x,y,c) = Halide::cast<uint8_t>(poisson_jacobi(ITERATIONS,x,y,c));
    //Halide::Buffer<float> result = final_image.realize(boundBoxMaxX_value - boundBoxMinX_value, boundBoxMaxY_value - boundBoxMinY_value, mmask.channels());

    Halide::Buffer<uint8_t> new_area(66, 61, mmask.channels());
    new_area.set_min(boundBoxMinX_value+1, boundBoxMinY_value+1);
    final_image.realize(new_area);

    save_image(new_area, "FinalImage.png");
}