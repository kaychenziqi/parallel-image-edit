
EXECUTABLE := PatchMatchCuda

CXX = g++ -m64
NVCC = nvcc

CXXFLAGS = -O3 -Wall
LDFLAGS = -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`
NVCC_FLAGS = -O3 -m64 --gpu-architecture compute_61

CU_FILES   := patchmatch.cu
CU_DEPS    :=
CC_FILES   := main.cpp util.cpp cycletimer.c

OBJDIR = objs
OBJS = $(OBJDIR)/main.o $(OBJDIR)/patchmatch.o $(OBJDIR)/util.o $(OBJDIR)/cycletimer.o 


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(OPENCV_FLAGS)

$(OBJDIR)/%.o: %.cpp
	$(CC) $< $(CFLAGS) -c -o $@ $(OPENCV_FLAGS)

$(OBJDIR)/%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCC_FLAGS) -c -o $@
