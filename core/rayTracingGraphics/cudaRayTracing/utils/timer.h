#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>

namespace cuda::rayTracing {

template <typename Clock = std::chrono::high_resolution_clock>
class Timer{
private:
    std::string name;
    const typename Clock::time_point start;
    typename Clock::time_point back;

    typename Clock::time_point measure(std::string massage, const typename Clock::time_point& back) const {
        typename Clock::time_point now = Clock::now();
        double countedTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - back).count();
        std::cout << massage << " :: " << countedTime << " ms" << std::endl;
        return now;
    }
public:
    Timer(std::string name) : name(name), start(Clock::now()), back(start) {}
    ~Timer() {
        measure(name, start);
    };

    void elapsedTime(std::string massage) {
        back = measure(massage, back);
    }
};

}
#endif // TIMER_H
