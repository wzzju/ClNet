//
// Created by yuchen on 18-1-15.
//

#ifndef CLNET_SPMV_H
#define CLNET_SPMV_H

#include <sstream>
#include "opencl/cl_objects.h"

#define VECTOR_SIZE 32
#define BLOCK_SIZE 128

void csrTest(cl_objects &clObject, std::stringstream &strs);

#endif //CLNET_SPMV_H
