CC = g++ -m64
DEBUG = 0

CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`

INC_FILES = util.h patchmatch.h cycletimer.h
CC_FILES = main.cpp util.cpp patchmatch.cpp cycletimer.c

INPUT_FILE = ../img/avatar.jpg
SRC_FILE = ../img/monalisa.jpg
OUTPUT_FILE = ../output/avatar-seq.jpg

default: all

all: seq

seq: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

test:
	./PatchMatchSeq -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

seq1: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=1
	./PatchMatchSeq -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 1

seq4: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=4
	./PatchMatchSeq -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 4

seq7: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=7
	./PatchMatchSeq -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

seq10: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchSeq $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=10
	./PatchMatchSeq -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 10

benchmark:
	make seq1
	make seq4
	make seq7
	make seq10

clean:
	rm -rf PatchMatchSeq