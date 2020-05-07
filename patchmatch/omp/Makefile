CC = g++ -m64
DEBUG = 0

CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OMP_FLAGS = -fopenmp -DOMP
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`

INC_FILES = util.h patchmatch.h cycletimer.h
CC_FILES = main.cpp util.cpp patchmatch.cpp cycletimer.c

INPUT_FILE = ../img/avatar.jpg
SRC_FILE = ../img/monalisa.jpg
OUTPUT_FILE = ../output/avatar-omp.jpg

default: all

all: omp

omp: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchOmp $(CC_FILES) $(OMP_FLAGS) $(LDFLAGS) $(OPENCV_FLAGS)

test:
	./PatchMatchOmp -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -t 8 -p 7

seq1: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchOmp $(CC_FILES) $(OMP_FLAGS) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=1
	./PatchMatchOmp -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -t 8 -p 1

seq4: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchOmp $(CC_FILES) $(OMP_FLAGS) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=4
	./PatchMatchOmp -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -t 8 -p 4

seq7: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchOmp $(CC_FILES) $(OMP_FLAGS) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=7
	./PatchMatchOmp -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -t 8 -p 7

seq10: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o PatchMatchOmp $(CC_FILES) $(OMP_FLAGS) $(LDFLAGS) $(OPENCV_FLAGS) -DHALF_PATCH=10
	./PatchMatchOmp -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -t 8 -p 10

benchmark:
	make seq1
	make seq4
	make seq7
	make seq10

clean:
	rm -rf PatchMatchOmp