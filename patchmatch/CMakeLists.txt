SET(USE_OMP FALSE CACHE BOOL "USE_OMP")
message(STATUS "USE_OMP ${USE_OMP}")

cmake_minimum_required (VERSION 2.8)
project (PatchMatch)
set (EXE_SEQ PatchMatchSeq)
set (EXE_OMP PatchMatchOmp)
set (CMAKE_CXX_STANDARD 11)

# Link to opencv files.
find_package(OpenCV REQUIRED)

include_directories (${OpenCV_INCLUDE_DIRS})

file (GLOB INC_FILES inc/*.h)
file (GLOB SRC_FILES src/*.cpp)

if (${USE_OMP})
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)

        list (REMOVE_ITEM INC_FILES "inc/fake_omp.h")
        list (REMOVE_ITEM SRC_FILES "src/fake_omp.cpp")

        include (INC_FILES)

        # Compile
        add_executable (${EXE_OMP} ${SRC_FILES})

        # Link the executable to pre-built library.
        target_link_libraries (${EXE_SEQ} ${OpenCV_LIBS})

    endif ()

else ()
    include_directories (inc)

    # Compile
    add_executable (${EXE_SEQ} ${SRC_FILES})

    # Link the executable to pre-built library.
    target_link_libraries (${EXE_SEQ} ${OpenCV_LIBS})
    
endif()