HALIDE_INC_PATH = $(HALIDE_PATH)/include
HALIDE_LIB_PATH = $(HALIDE_PATH)/bin

INCFLAG = -I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools
LDFLAG = -L $(HALIDE_PATH)/bin -lHalide `libpng-config --cflags --ldflags` -ljpeg

EXECUTABLE = PatchMatchHalide

CC = g++ -m64
DEBUG = 0
CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`
INC_FILES = 
CC_FILES = resize.cpp

RAW_INPUT_FILE = ../img/avatar.jpg
RAW_SRC_FILE = ../img/mona-lisa.jpg
OUTPUT_FILE = ../output/avatar-halide.jpg
INPUT_FILE = $(RAW_INPUT_FILE)-resize.jpg
SRC_FILE = $(RAW_SRC_FILE)-resize.jpg

default: all

all: halide

halide:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11
	
resize_exe: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o resize $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

resize: resize_exe
	./resize -i $(RAW_INPUT_FILE) -s $(RAW_SRC_FILE) -o $(OUTPUT_FILE) -p 7

test:
	LD_LIBRARY_PATH=$(HALIDE_LIB_PATH) ./$(EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

clean:
	rm $(EXECUTABLE)