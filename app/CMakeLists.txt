cmake_minimum_required(VERSION 3.4.1)

set(DIST_DIR ${CMAKE_SOURCE_DIR}/../distribution)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}    -Xlinker --no-warn-mismatch -O2 -mfpu=vfpv3-d16 -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Xlinker --no-warn-mismatch -O2 -mfpu=vfpv3-d16 -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -mfloat-abi=hard -std=gnu++11")
set(STLPORT_FORCE_REBUILD "true")

add_definitions(-DDEBUG)

add_library(lib_opencl SHARED IMPORTED)
set_target_properties( lib_opencl
                       PROPERTIES
                       IMPORTED_LOCATION
                       ${DIST_DIR}/opencl/lib/${ANDROID_ABI}/libGLES_mali_v2.so )

add_library( # Sets the name of the library.
             clnet

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ${SRC_DIR}/opencl/cl_log.cpp
             ${SRC_DIR}/opencl/cl_objects.cpp
             ${SRC_DIR}/cnpy.cpp
             ${SRC_DIR}/run_neon.cpp
             ${SRC_DIR}/clnet.cpp)

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

find_library(z-lib
        NAMES z)

target_include_directories(clnet PRIVATE ${DIST_DIR}/opencl/include)
include_directories(${SRC_DIR}/include)

target_link_libraries( # Specifies the target library.
                       clnet

                       lib_opencl
                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       ${z-lib})