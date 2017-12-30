//
// Created by yuchen on 17-12-30.
//

#include <jni.h>
#include "opencl/cl_log.h"
#include "clnet.h"

JNIEXPORT void JNICALL
CLNET(runNEON)(JNIEnv *env, jobject instance) {
    LOGD("HELLO, NEON!");
}