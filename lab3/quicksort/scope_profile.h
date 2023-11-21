
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
    explicit ScopeProfileLog(const std::string_view &message) {
        _message = message;
        _start = std::chrono::high_resolution_clock::now();
    }

    ~ScopeProfileLog() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - _start);
        std::cout << _message.data() << " : " << duration.count() << " (microseconds)\n";
    }

private:
    std::string_view _message;
    std::chrono::time_point<std::chrono::steady_clock> _start;
};

#define SCOPED_PROFILE_LOG(message) auto scopedLog = ScopeProfileLog(message);












































/*

class ScopeProfileLog {
public:
    explicit ScopeProfileLog(const std::string_view &message) {
        _message = message;
        _start = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
    }

    ~ScopeProfileLog() {
        uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
        std::cout << _message.data() << " : " << now - _start << " (milliseconds)";
    }

private:
    std::string_view _message;
    uint64_t _start;
};

#define SCOPED_PROFILE_LOG(message) auto scopedLog = ScopeProfileLog(message);

 * */





