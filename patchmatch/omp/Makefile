CC = g++ -m64
DEBUG = 1

CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OMP_FLAGS = -fopenmp -DOMP
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`

INC_FILES = util.h patchmatch.h cycletimer.h
CC_FILES = main.cpp util.cpp patchmatch.cpp cycletimer.c

default: all

all: omp

omp: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchOmp $(CC_FILES) $(OMP_FLAGS) $(LDFLAGS) $(OPENCV_FLAGS)

test:
	./PatchMatchOmp -i ../img/Cloth2-view1.jpg -s ../img/Cloth1-view1.jpg -o ../output/Cloth2-view1-omp.jpg -p 9 -t 8

clean:
	rm -rf PatchMatchOmp