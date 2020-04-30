CC = g++ -m64
DEBUG = 1

CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`

INC_FILES = util.h patchmatch.h cycletimer.h
CC_FILES = main.cpp util.cpp patchmatch.cpp cycletimer.c

default: all

all: seq

seq: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

clean:
	rm -rf PatchMatchSeq