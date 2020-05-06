HALIDE_INC_PATH = $(HALIDE_PATH)/include
HALIDE_LIB_PATH = $(HALIDE_PATH)/bin

INCFLAG = -I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools
LDFLAG = -L $(HALIDE_PATH)/bin -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl

EXECUTABLE = PatchMatchHalide

CC = g++ -m64
DEBUG = 0
CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`
INC_FILES = 
CC_FILES = resize.cpp

RAW_INPUT_FILE = ../img/avatar.jpg
RAW_SRC_FILE = ../img/monalisa.jpg
OUTPUT_FILE = ../output/avatar-halide.jpg
INPUT_FILE = $(RAW_INPUT_FILE)-resize.jpg
SRC_FILE = $(RAW_SRC_FILE)-resize.jpg

AUTOGEN_EXECUTABLE = PatchmatchAutoGenerator
AUTO_LIB = auto_schedule_false.a auto_schedule_true.a
AUTO_EXECUTABLE = PatchmatchAuto

AUTOGEN_CXXFLAGS = -g -std=c++11 -fno-rtti
AUTO_CXXFLAGS = -std=c++11
AUTO_LDFLAGS = $(LDFLAG)

AUTOGEN_CC_FILES = auto_schedule_generator.cpp $(HALIDE_PATH)/tools/GenGen.cpp
AUTO_CC_FILES = auto_schedule_run.cpp

default: all

all: halide

# Usage:
# make resize

resize_exe: $(CC_FILES) $(INC_FILES)
	$(CC) $(CFLAGS) -o resize $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

resize: resize_exe
	./resize -i $(RAW_INPUT_FILE) -s $(RAW_SRC_FILE) -o $(OUTPUT_FILE) -p 7

# Usage:
# make halide
# make test

halide:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11

gpu:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11 -DGPU_SCHEDULE=1

test:
	./$(EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

# Usage:
# make auto_gen
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HALIDE_PATH/bin
# make auto_schedule_false
# make auto_schedule_true
# make auto

auto_gen:
	g++ $(AUTOGEN_CC_FILES) $(AUTOGEN_CXXFLAGS) $(INCFLAG) $(LDFLAG) -o $(AUTOGEN_EXECUTABLE)

auto_schedule_false:
	./$(AUTOGEN_EXECUTABLE) -o . -g auto_schedule_gen -f auto_schedule_false -e static_library,h,schedule target=host auto_schedule=false

auto_schedule_true:
	./$(AUTOGEN_EXECUTABLE) -o . -g auto_schedule_gen -f auto_schedule_true -e static_library,h,schedule target=host auto_schedule=true machine_params=32,16777216,40

auto: # auto_schedule_false auto_schedule_true
	g++ $(AUTO_CC_FILES) $(AUTO_CXXFLAGS) $(INCFLAG) $(AUTO_LIB) $(AUTO_LDFLAGS) -o $(AUTO_EXECUTABLE)

auto_test:
	./$(AUTO_EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

auto_all: 
	make auto_gen 
	make auto_schedule_false 
	make auto_schedule_true
	make auto

seq1:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11 -DHALF_PATCH=1
	./$(EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 1

seq4:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11 -DHALF_PATCH=4
	./$(EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 4

seq7:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11 -DHALF_PATCH=7
	./$(EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

seq10:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11 -DHALF_PATCH=10
	./$(EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 10

benchmark:
	make seq1
	make seq4
	make seq7
	make seq10


auto_run:
	make auto_schedule_false 
	make auto_schedule_true
	make auto

auto_gen1:
	g++ $(AUTOGEN_CC_FILES) $(AUTOGEN_CXXFLAGS) $(INCFLAG) $(LDFLAG) -o $(AUTOGEN_EXECUTABLE) -DHALF_PATCH=1

auto_test1:
	./$(AUTO_EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 1

auto_gen4:
	g++ $(AUTOGEN_CC_FILES) $(AUTOGEN_CXXFLAGS) $(INCFLAG) $(LDFLAG) -o $(AUTOGEN_EXECUTABLE) -DHALF_PATCH=4

auto_test4:
	./$(AUTO_EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 4

auto_gen7:
	g++ $(AUTOGEN_CC_FILES) $(AUTOGEN_CXXFLAGS) $(INCFLAG) $(LDFLAG) -o $(AUTOGEN_EXECUTABLE) -DHALF_PATCH=7

auto_test7:
	./$(AUTO_EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 7

auto_gen10:
	g++ $(AUTOGEN_CC_FILES) $(AUTOGEN_CXXFLAGS) $(INCFLAG) $(LDFLAG) -o $(AUTOGEN_EXECUTABLE) -DHALF_PATCH=10

auto_test10:
	./$(AUTO_EXECUTABLE) -i $(INPUT_FILE) -s $(SRC_FILE) -o $(OUTPUT_FILE) -p 10

auto_benchmark: 
	make auto_gen1
	make auto_run
	make auto_test1

	make auto_gen4
	make auto_run
	make auto_test4

	make auto_gen7
	make auto_run
	make auto_test7

	make auto_gen10
	make auto_run
	make auto_test10
	

clean:
	rm $(EXECUTABLE) $(AUTOGEN_EXECUTABLE) $(AUTO_EXECUTABLE)