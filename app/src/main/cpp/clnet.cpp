#include <jni.h>
#include <string>
#include "clnet.h"

JNIEXPORT jstring JNICALL
CLNET(stringFromJNI)(JNIEnv *env, jobject /* this */) {
    std::string hello = "Start ClNet";
    return env->NewStringUTF(hello.c_str());
}
