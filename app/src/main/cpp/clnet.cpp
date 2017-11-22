#include <jni.h>
#include <string>
#include "clnet.h"

JNIEXPORT jstring JNICALL
CLNET(testCL)(JNIEnv *env, jobject /* this */) {
    std::string hello = "\n**************************END**************************\n";

    return env->NewStringUTF(hello.c_str());
}
