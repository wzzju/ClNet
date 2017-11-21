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

JNIEXPORT jstring JNICALL
CLNET(stringFromJNI)(JNIEnv *env, jobject /* this */);

#ifdef __cplusplus
}
#endif

#endif //CLNET_CLNET_H
