
#pragma once


#include <iostream>
#include <chrono>
#include <string_view>

#include <vector>
#include <list>
#include <array>

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
        std::cout << _message.data() << " : " << now - _start << " (milliseconds)\n";
    }

private:
    std::string_view _message;
    uint64_t _start;
};

#define SCOPED_PROFILE_LOG(message) auto scopedLog = ScopeProfileLog(message);
