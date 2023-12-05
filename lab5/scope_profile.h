
#pragma once


#include <iostream>
#include <chrono>
#include <string_view>

#include <vector>
#include <list>
#include <array>
#include <memory_resource>

class ScopeProfileLog {
public:
    explicit ScopeProfileLog(const std::string &message) {
        _message = message;
        _start = std::chrono::high_resolution_clock::now();
    }

    ~ScopeProfileLog() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - _start);
        std::cout << _message.data() << " : " << duration.count() << " (microseconds)\n";
    }

private:
    std::string _message;
    std::chrono::time_point<std::chrono::steady_clock> _start;
};

#define SCOPED_PROFILE_LOG(message) auto scopedLog = ScopeProfileLog(message);



