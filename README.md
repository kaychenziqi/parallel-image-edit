# Parallel Image Editing

CMU 15-618 Parallel Programming, May 2020

Author: Ziqi Chen, Yuxin Ding

## Introduction

This repository contains the implementations of two image editing algorithms, the Poisson Image Editing algorithm and the PatchMatch algorithm. 

For each algorithm, there is a sequential version and three parallel versions, written in OpenMP, CUDA and Halide.

For Halide, there are multiple implementations, with manual scheduling or auto scheduling.

## Dependencies

The project is developed on a x86 Red Hat Linux machine.

1. OpenCV 3.4.0 or above

2. OpenMP

3. CUDA 10.2

4. Halide 

    Please find a precompiled release or compile from source [(link)](https://github.com/halide/Halide).


## PatchMatch

### Overview

The paper: [PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)

The code implements a basic application of the algorithm, which reconstructs a target image using a source image. 

|Target|Source|Output|
|-------|-------|-------|
| <img src="https://github.com/kaychenziqi/parallel-image-edit/blob/master/patchmatch/img/avatar.jpg" width="256">  | <img src="https://github.com/kaychenziqi/parallel-image-edit/blob/master/patchmatch/img/monalisa.jpg" width="256"> | <img src="https://github.com/kaychenziqi/parallel-image-edit/blob/master/patchmatch/output/avatar-output.jpg" width="256"> |

### How to Run

#### Sequential, OpenMP, CUDA Versions

```
cd patchmatch/seq 
# cd patchmatch/omp
# cd patchmatch/cuda

# change input/output files in Makefile, if necessary
# ...

make
make test
```

#### Halide Version

```
cd patchmatch/halide
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HALIDE_INSTALL_PATH/bin
```

1. Auto Schedule
    ```
    make auto_all
    make auto_test
    ```

2. Manual Schedule
    ```
    make halide
    make test
    ```

## Poisson Image Editing

### Overview

The paper: [Poisson Image Editing](http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)

|Source|Target|Mask|Output|
|-------|-------|-------|-------|
| <img src="https://github.com/kaychenziqi/parallel-image-edit/blob/master/poisson/input/1/source.png" width="192">  | <img src="https://github.com/kaychenziqi/parallel-image-edit/blob/master/poisson/input/1/target.png" width="192"> | <img src="https://github.com/kaychenziqi/parallel-image-edit/blob/master/poisson/input/1/mask.png" width="192"> | <img src="https://github.com/kaychenziqi/parallel-image-edit/blob/master/poisson/output/finalimage.jpg" width="192"> |

### How to Run

#### Sequential Version
```
cd poisson/seq
cmake.
make
./PoissonImageEdit ../input/1/source.png ../input/1/target.png ../input/1/mask.png
```

#### CUDA Version
```
cd poisson/cuda
make
./PoissonImageEdit ../input/1/source.png ../input/1/target.png ../input/1/mask.png
```

#### OpenMP Version
```
cd poisson/omp
make
./PoissonImageEdit ../input/1/source.png ../input/1/target.png ../input/1/mask.png
```


#### Halide Version

1. Halide Auto Schedule

    ```
    cd poisson/halide
    make auto_gen
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HALIDE_PATH/bin
    make auto_schedule_false
    make auto_schedule_true
    make auto
    ./PoissonAuto
    ```

2. Halide Manual Schedule 

    Set `USE_GPU` in PoissonImageEdit_GPU.cpp to 0 
    ```
    cd poisson/halide
    make gpuversion
    ./PoissonImageEdit_GPU
    ```

3. Halide GPU version 

    Set `USE_GPU` in PoissonImageEdit_GPU.cpp to 1 

    ```
    cd poisson/halide
    make gpuversion
    ./PoissonImageEdit_GPU
    ```
