
EXECUTABLE := PoissonImageEdit

CU_FILES   := PoissonImageEdit.cu

CU_DEPS    :=

CC_FILES   := 

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
OPENCVFLAGS=-L/afs/andrew.cmu.edu/usr18/yuxindin/private/15-618/opencv34/lib64 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61
NVCCFLAGS+=-DOPENCV `pkg-config opencv --cflags --libs`


OBJS=$(OBJDIR)/PoissonImageEdit.o  


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(OPENCVFLAGS)

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
