//
// Created by yuchen on 17-12-30.
//

#ifndef CLNET_HELPER_H
#define CLNET_HELPER_H

#include <string>
#include <chrono>
#include <iostream>
#include <vector>
#include "opencl/cl_log.h"

namespace helper {
    template<typename FuncType>
    class InnerScopeExit {
    public:
        InnerScopeExit(const FuncType _func) : func(_func) {}

        ~InnerScopeExit() { func(); }

    private:
        FuncType func;
    };

    template<typename F>
    InnerScopeExit<F> MakeScopeExit(F f) {
        return InnerScopeExit<F>(f);
    };
}
#define DO_STRING_JOIN(arg1, arg2) arg1 ## arg2
#define STRING_JOIN(arg1, arg2) DO_STRING_JOIN(arg1, arg2)// 定义DO_STRING_JOIN的原因是需要先解析__LINE__宏
// SCOPE_EXIT宏的作用：无论这个宏出现在何处，它总是在scope结束处才执行code指定的代码。
#define SCOPE_EXIT(code) auto STRING_JOIN(scope_exit_object_, __LINE__) = helper::MakeScopeExit([&](){code;});

// 计算代码在一定scope内执行所耗费的时间
class CostTimeHelper {
public:
    CostTimeHelper(const std::string &_tag) : tag(_tag) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~CostTimeHelper() {
        stop_time = std::chrono::high_resolution_clock::now();
        const auto cost_time = std::chrono::duration_cast<std::chrono::microseconds>(
                stop_time - start_time).count(); //us
        LOGD("%s cost time : %lld us.\n", tag.c_str(), cost_time);
    }

private:
    std::string tag;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
};

#endif //CLNET_HELPER_H
