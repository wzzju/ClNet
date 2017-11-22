//
// Created by yuchen on 17-11-21.
//

#ifndef CLNET_CLNET_H
#define CLNET_CLNET_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROGRAM_FILE "/data/user/0/io.github.wzzju.clnet/app_execdir/matvec.cl"
#define KERNEL_FUNC "matvec_mult"

#define CLNET(METHOD_NAME) \
Java_io_github_wzzju_clnet_MainActivity_##METHOD_NAME

JNIEXPORT jstring JNICALL
CLNET(testCL)(JNIEnv *env, jobject /* this */);

JNIEXPORT void JNICALL
CLNET(deviceQuery)(JNIEnv *env, jobject instance);

#ifdef __cplusplus
}
#endif

#endif //CLNET_CLNET_H
