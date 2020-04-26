CC = g++ -m64
NVCC = nvcc
DEBUG = 1

CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OMP_FLAGS = -fopenmp -DOMP
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`
CU_LDFLAGS = -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC_FLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

LIBS = GL glut cudart
LDLIBS  := $(addprefix -l, $(LIBS))

INC_FILES = util.h patchmatch.h
CC_FILES = main.cpp patchmatch.cpp
CU_FILES = patchmatch.cu

CU_OBJDIR = objs
CU_OBJS = $(CU_OBJDIR)/main.o $(CU_OBJDIR)/patchmatch_cuda.o

LOGS=logs

default: seq

all: seq omp cuda

dirs:
	mkdir -p $(CU_OBJDIR)/

$(CU_OBJDIR)/%.o: %.cpp
	$(CC) $< $(CFLAGS) -c -o $@ $(OPENCV_FLAGS)

$(CU_OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCC_FLAGS) -c -o $@ $(OPENCV_FLAGS)

seq: $(CC_FILES) $(INC_FILES) fake_omp.h fake_omp.cpp
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) fake_omp.cpp $(LDFLAGS) $(OPENCV_FLAGS)

omp: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o PatchMatchOmp $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

cuda: dirs $(CU_OBJS)
	$(CC) $(CFLAGS) -o PatchMatchCuda $(CU_OBJS) $(LDFLAGS) $(CU_LDFLAGS) $(LDLIBS) $(OPENCV_FLAGS)

clean:
	rm -rf $(CU_OBJDIR) PatchMatchSeq PatchMatchOmp PatchMatchCuda