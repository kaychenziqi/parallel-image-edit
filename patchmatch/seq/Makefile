CC = g++ -m64
DEBUG = 0

CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`

INC_FILES = util.h patchmatch.h cycletimer.h
CC_FILES = main.cpp util.cpp patchmatch.cpp cycletimer.c

default: all

all: seq

seq: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

test:
	# ./PatchMatchSeq -i ../img/Cloth2-view1.jpg -s ../img/Cloth1-view1.jpg -o ../output/Cloth2-view1-seq.jpg -p 7
	./PatchMatchSeq -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-seq.jpg -p 1
	./PatchMatchSeq -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-seq.jpg -p 3
	./PatchMatchSeq -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-seq.jpg -p 5
	./PatchMatchSeq -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-seq.jpg -p 7
	./PatchMatchSeq -i ../img/avatar.jpg -s ../img/mona-lisa.jpg -o ../output/avatar-seq.jpg -p 9

clean:
	rm -rf PatchMatchSeq