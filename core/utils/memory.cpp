#include "memory.h"

#include "operations.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

namespace moon::utils {

Memory::~Memory() {
#ifndef DEBUG_PRINT_DISABLE
    status();
#endif
};

Memory& Memory::instance() {
    static Memory s;
    return s;
}

VkResult Memory::allocate(VkPhysicalDevice physicalDevice, VkDevice device, VkMemoryRequirements requirements, VkMemoryPropertyFlags properties, VkDeviceMemory* memory, const std::string& name) {
    VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = requirements.size;
        allocInfo.memoryTypeIndex = physicalDevice::findMemoryTypeIndex(physicalDevice, requirements.memoryTypeBits, properties);
    const VkResult result = CHECK(vkAllocateMemory(device, &allocInfo, nullptr, memory));
    if (result == VK_SUCCESS) {
        uint64_t align = requirements.alignment ? requirements.alignment : 1u;
        uint64_t requested = requirements.size;
        uint64_t allocated = ((requested + align - 1) / align) * align;

        std::lock_guard<std::mutex> lock(mtx);
        memoryMap[*memory] = Description{ align, allocated, name };
        totalMemoryUsed += allocated;
    }

    return result;
}

void Memory::nameMemory(VkDeviceMemory memory, const std::string& name) {
    std::lock_guard<std::mutex> lock(mtx);
    if (auto memIt = memoryMap.find(memory); memIt != memoryMap.end()) {
        auto& desc = memIt->second;
        desc.name = name;
    }
}

void Memory::free(VkDeviceMemory memory) {
    std::lock_guard<std::mutex> lock(mtx);
    if (auto memIt = memoryMap.find(memory); memIt != memoryMap.end()) {
        const auto& desc = memIt->second;
        totalMemoryUsed -= desc.size;
        memoryMap.erase(memory);
    }
}

void Memory::status() {
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto& [memory, desc] : memoryMap) {
        std::cout << std::setw(16) << (uint64_t)memory << "\t"
                  << std::setw(16) << desc.alignment << "\t"
                  << std::setw(16) << desc.size << "\t"
                  << std::setw(16) << desc.name << std::endl;
    }

    std::cout << "Total allocations : " << memoryMap.size() << std::endl;
    std::cout << "Total memory used : " << totalMemoryUsed << std::endl;
}

}