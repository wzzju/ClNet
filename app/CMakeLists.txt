cmake_minimum_required(VERSION 3.4.1)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp)

add_library( # Sets the name of the library.
             clnet-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/clnet.cpp )


find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

include_directories(${SRC_DIR}/include)

target_link_libraries( # Specifies the target library.
                       clnet-lib

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )