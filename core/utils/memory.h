#ifndef MOON_UTILS_MEMORY_H
#define MOON_UTILS_MEMORY_H

#include <vulkan.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace moon::utils {

class Memory {
private:
    struct Description {
        uint64_t alignment{ 0 };
        uint64_t size{ 0 };
        std::string name{};
    };

    uint64_t totalMemoryUsed{ 0 };
    std::unordered_map<VkDeviceMemory, Description> memoryMap;

    Memory() = default;
    ~Memory();
public:

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    static Memory& instance();

    VkResult allocate(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkMemoryRequirements requirements,
        VkMemoryPropertyFlags properties,
        VkDeviceMemory* memory,
        const std::string& name = "");

    void nameMemory(
        VkDeviceMemory memory,
        const std::string& name);

    void free(
        VkDeviceMemory memory);

    void status();
};

}

#endif // MOON_UTILS_MEMORY_H