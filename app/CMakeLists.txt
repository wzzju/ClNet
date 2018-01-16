cmake_minimum_required(VERSION 3.4.1)

set(DIST_DIR ${CMAKE_SOURCE_DIR}/../distribution)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

add_definitions(-DDEBUG
                -DANDROID_CL
                -D__CL_ENABLE_EXCEPTIONS
                -DCL_USE_DEPRECATED_OPENCL_1_1_APIS)

add_library(lib_opencl SHARED IMPORTED)
set_target_properties( lib_opencl
                       PROPERTIES
                       IMPORTED_LOCATION
                       ${DIST_DIR}/opencl/lib/${ANDROID_ABI}/libOpenCL.so )
#                       ${DIST_DIR}/opencl/lib/${ANDROID_ABI}/libGLES_mali_v2.so )
#                       ${DIST_DIR}/opencl/lib/${ANDROID_ABI}/libGLES_mali.so )

aux_source_directory(${SRC_DIR} SOURCE_FILES)
aux_source_directory(${SRC_DIR}/opencl SOURCE_FILES_OPENCL)
aux_source_directory(${SRC_DIR}/layers SOURCE_FILES_LAYERS)
list(APPEND SOURCE_FILES ${SOURCE_FILES_OPENCL})
list(APPEND SOURCE_FILES ${SOURCE_FILES_LAYERS})

add_library( # Sets the name of the library.
             clnet

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ${SOURCE_FILES})

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