CC = g++ -m64
DEBUG = 0

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
	# ./PatchMatchOmp -i ../img/Cloth2-view1.jpg -s ../img/Cloth1-view1.jpg -o ../output/Cloth2-view1-omp.jpg -p 7 -t 8
	./PatchMatchOmp -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-omp.jpg -t 8 -p 1 
	./PatchMatchOmp -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-omp.jpg -t 8 -p 3
	./PatchMatchOmp -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-omp.jpg -t 8 -p 5
	./PatchMatchOmp -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-omp.jpg -t 8 -p 7
	./PatchMatchOmp -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-omp.jpg -t 8 -p 9

clean:
	rm -rf PatchMatchOmp