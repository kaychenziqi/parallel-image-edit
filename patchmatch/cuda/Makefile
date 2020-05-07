
EXECUTABLE := PatchMatchCuda

CC = g++ -m64
NVCC = nvcc

CXXFLAGS = -O3 -Wall
LDFLAGS = -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`
# NVCC_FLAGS = -O3 -m64 --gpu-architecture compute_61
NVCC_FLAGS = -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

CU_FILES   := patchmatch.cu
CU_DEPS    :=
CC_FILES   := main.cpp util.cpp #cycletimer.c

OBJDIR = objs
OBJS = $(OBJDIR)/main.o $(OBJDIR)/patchmatch.o $(OBJDIR)/util.o # $(OBJDIR)/cycletimer.o 

INPUT_FILE = ../img/avatar.jpg
SRC_FILE = ../img/monalisa.jpg
OUTPUT_FILE = ../output/avatar-cuda.jpg


.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

test:
	./PatchMatchCuda -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(OPENCV_FLAGS)

$(OBJDIR)/%.o: %.cpp
	$(CC) $< $(CFLAGS) -c -o $@ $(OPENCV_FLAGS)

# $(OBJDIR)/%.o: %.c
# 	$(CC) $< $(CFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCC_FLAGS) -c -o $@
