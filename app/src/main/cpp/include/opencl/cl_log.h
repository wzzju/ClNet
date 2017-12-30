//
// Created by yuchen on 17-11-22.
//

#ifndef CLNET_CLLOG_H
#define CLNET_CLLOG_H

#include <android/log.h>
#include <CL/cl.h>

// Commonly-defined shortcuts for LogCat output from native C applications.
#define  LOG_TAG    "CLNET"

#ifdef DEBUG
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define  LOGD(...)
#define  LOGE(...)
#endif

/* This function helps to create informative messages in
 * case when OpenCL errors occur. The function returns a string
 * representation for an OpenCL error code.
 * For example, "CL_DEVICE_NOT_FOUND" instead of "-1".
 */
const char *opencl_error_to_str(cl_int error);

#define CHECK_ERRORS(ERR, FILE, LINE)                                                 \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGD                                                                          \
        (                                                                             \
            "OPENCL ERROR with the error code %s.\nIt is happened in file %s at line %d.\nExiting!\n",   \
            opencl_error_to_str(ERR), FILE, LINE                                      \
        );                                                                            \
                                                                                      \
        return;                                                                       \
    }

#define CHECK_ERRORS_WITH_RETURN(ERR, FILE, LINE)                                     \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGD                                                                          \
        (                                                                             \
            "OPENCL ERROR with the error code %s.\nIt is happened in file %s at line %d.\nExiting!\n",   \
            opencl_error_to_str(ERR), FILE, LINE                                      \
        );                                                                            \
                                                                                      \
        return ERR;                                                                   \
    }

#define CHECK_ERRORS_WITH_NULL_RETURN(ERR, FILE, LINE)                                \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGD                                                                          \
        (                                                                             \
            "OPENCL ERROR with the error code %s.\nIt is happened in file %s at line %d.\nExiting!\n",   \
            opencl_error_to_str(ERR), FILE, LINE                                      \
        );                                                                            \
                                                                                      \
        return nullptr;                                                                  \
    }

#endif //CLNET_CLLOG_H
