cmake_minimum_required (VERSION 2.8)
project (PatchMatch)
set (EXE_NAME PatchMatch)
# set (CMAKE_CXX_STANDARD 11)

# Link to opencv files.
find_package(OpenCV REQUIRED)

file (GLOB SRC_FILES src/*.cpp)

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories(inc)

# Compile
add_executable (${EXE_NAME} ${SRC_FILES})

# Link the executable to pre-built library.
target_link_libraries (${EXE_NAME} ${OpenCV_LIBS})