//
// Created by yuchen on 17-11-21.
//

#ifndef CLNET_CLNET_H
#define CLNET_CLNET_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLNET(METHOD_NAME) \
Java_io_github_wzzju_clnet_MainActivity_##METHOD_NAME

JNIEXPORT void JNICALL
CLNET(initNet)(JNIEnv *env, jobject instance, jstring weightPath_,
               jstring clPath_, jboolean useGPU);

JNIEXPORT
jfloatArray JNICALL
CLNET(inference)(JNIEnv *env, jobject instance,
                 jfloatArray data_);

JNIEXPORT jstring JNICALL
CLNET(runCL)(JNIEnv *env, jobject instance, jstring path_);

JNIEXPORT void JNICALL
CLNET(runNpy)(JNIEnv *env, jobject instance, jstring dir_);

JNIEXPORT void JNICALL
CLNET(deviceQuery)(JNIEnv *env, jobject instance);

#ifdef __cplusplus
}
#endif

#endif //CLNET_CLNET_H