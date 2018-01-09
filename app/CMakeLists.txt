cmake_minimum_required(VERSION 3.4.1)

set(DIST_DIR ${CMAKE_SOURCE_DIR}/../distribution)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__CL_ENABLE_EXCEPTIONS -DANDROID_CL -O3 -ffast-math")

add_definitions(-DDEBUG)

add_library(lib_opencl SHARED IMPORTED)
set_target_properties( lib_opencl
                       PROPERTIES
                       IMPORTED_LOCATION
                       ${DIST_DIR}/opencl/lib/${ANDROID_ABI}/libOpenCL.so )
#                       ${DIST_DIR}/opencl/lib/${ANDROID_ABI}/libGLES_mali_v2.so )

add_library( # Sets the name of the library.
             clnet

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ${SRC_DIR}/opencl/cl_log.cpp
             ${SRC_DIR}/opencl/cl_objects.cpp
             ${SRC_DIR}/layers/conv_layer.cpp
             ${SRC_DIR}/layers/fc_layer.cpp
             ${SRC_DIR}/layers/pooling_layer.cpp
             ${SRC_DIR}/layers/relu_layer.cpp
             ${SRC_DIR}/layers/softmax_layer.cpp
             ${SRC_DIR}/net.cpp
             ${SRC_DIR}/utility_cpu.cpp
             ${SRC_DIR}/utility_gpu.cpp
             ${SRC_DIR}/cnpy.cpp
             ${SRC_DIR}/clnet.cpp)

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

find_library( # Sets the name of the path variable.
              jnigraphics-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              jnigraphics )

find_library(z-lib
             NAMES z)

target_include_directories(clnet PRIVATE ${DIST_DIR}/opencl/include)
include_directories(${SRC_DIR}/include)

target_link_libraries( # Specifies the target library.
                       clnet

                       lib_opencl
                       ${log-lib}
                       ${jnigraphics-lib}
                       ${z-lib})