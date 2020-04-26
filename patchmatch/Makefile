DEBUG=0
CC=g++
OMP=-fopenmp -DOMP
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS= -lm
OPENCV_FLAGS=`pkg-config opencv --cflags --libs`

INC_FILES = patchmatch.h retarget.h
SRC_FILES = main.cpp patchmatch.cpp retarget.cpp

all: PatchMatchSeq PatchMatchOmp

PatchMatchSeq: $(SRC_FILES) $(INC_FILES) fake_omp.h fake_omp.cpp
	$(CC) $(CFLAGS) -o PatchMatchSeq $(SRC_FILES) fake_omp.cpp $(LDFLAGS) $(OPENCV_FLAGS)

PatchMatchOmp: $(SRC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) $(OMP) -o PatchMatchOmp $(SRC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

clean:
	rm -f PatchMatchSeq PatchMatchOmp