#include "memory.h"

#include "operations.h"

#include <iostream>

namespace moon::utils {

Memory::~Memory() {
#ifndef DEBUG_PRINT_DISABLE
    instance().status();
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
        memoryMap[*memory] = Description{ requirements.alignment, requirements.size, name };
        totalMemoryUsed += requirements.alignment + requirements.size;
    }

    return result;
}

void Memory::nameMemory(VkDeviceMemory memory, const std::string& name) {
    if (auto memIt = memoryMap.find(memory); memIt != memoryMap.end()) {
        auto& desc = memIt->second;
        desc.name = name;
    }
}

void Memory::free(VkDeviceMemory memory) {
    if (auto memIt = memoryMap.find(memory); memIt != memoryMap.end()) {
        const auto& desc = memIt->second;
        totalMemoryUsed -= desc.alignment + desc.size;
        memoryMap.erase(memory);
    }
}

void Memory::status() {
    for (const auto& [memory, desc] : memoryMap) {
        std::cout << std::setw(16) << memory << "\t"
                  << std::setw(16) << memoryMap[memory].alignment << "\t"
                  << std::setw(16) << memoryMap[memory].size << "\t"
                  << std::setw(16) << memoryMap[memory].name << std::endl;
    }

    std::cout << "Total allocations : " << memoryMap.size() << std::endl;
    std::cout << "Total memory used : " << totalMemoryUsed << std::endl;
}

}