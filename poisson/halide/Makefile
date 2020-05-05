HALIDE_INC_PATH = $(HALIDE_PATH)/include
HALIDE_LIB_PATH = $(HALIDE_PATH)/bin

INCFLAG = -I $(HALIDE_PATH)/include -I $(HALIDE_PATH)/tools
LDFLAG = -L $(HALIDE_PATH)/bin -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl

EXECUTABLE = PoissonHalide

CC = g++ -m64
DEBUG = 0
CFLAGS = -g -O3 -Wall -DDEBUG=$(DEBUG)
LDFLAGS = -lm
OPENCV_FLAGS = -DOPENCV `pkg-config opencv --cflags --libs`
INC_FILES = cycletimer.h
CC_FILES = cycletimer.c

AUTOGEN_EXECUTABLE = PoissonAutoGenerator
AUTO_LIB = auto_schedule_false.a auto_schedule_true.a
AUTO_EXECUTABLE = PoissonAuto

AUTOGEN_CXXFLAGS = -g -std=c++11 -fno-rtti
AUTO_CXXFLAGS = -std=c++11
AUTO_LDFLAGS = $(LDFLAG)

AUTOGEN_CC_FILES = PoissonImageEdit_Generator.cpp $(HALIDE_PATH)/tools/GenGen.cpp
AUTO_CC_FILES = PoissonImageEdit_Generator_Run.cpp

default: all

all: halide

# Usage:
# make resize

# resize_exe: $(CC_FILES) $(INC_FILES)
# 	$(CC) $(CFLAGS) -o resize $(CC_FILES) $(LDFLAGS) $(OPENCV_FLAGS)

# resize: resize_exe
# 	./resize -i $(RAW_INPUT_FILE) -s $(RAW_SRC_FILE) -o $(OUTPUT_FILE) -p 7

# Usage:
# make halide

halide:
	g++ main.cpp -g $(INCFLAG) $(LDFLAG) -o $(EXECUTABLE) -std=c++11

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
	g++ $(AUTO_CC_FILES) $(AUTO_CXXFLAGS) $(INCFLAG) $(AUTO_LIB) $(AUTO_LDFLAGS) $(CC_FILES) $(INC_FILES) -o $(AUTO_EXECUTABLE)

clean:
	rm $(AUTOGEN_EXECUTABLE) $(AUTO_EXECUTABLE) $(AUTO_LIB) auto_schedule_true.h auto_schedule_false.h auto_schedule_true.schedule.h auto_schedule_false.schedule.h