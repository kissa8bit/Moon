#include "workflow.h"

#include <utils/operations.h>

namespace moon::workflows {

Workflow& Workflow::setDeviceProp(VkPhysicalDevice physical, VkDevice logical){
    physicalDevice = physical;
    device = logical;
    return *this;
}

void Workflow::update(uint32_t frameNumber) {
    if (commandBuffers[frameNumber].dropFlag()) {
        CHECK(commandBuffers[frameNumber].reset());
        CHECK(commandBuffers[frameNumber].begin());
        updateCommandBuffer(frameNumber);
        CHECK(commandBuffers[frameNumber].end());
    }
}

Workflow::operator utils::vkDefault::CommandBuffers& () {
    return commandBuffers;
}

Workflow::operator utils::vkDefault::CommandBuffers* () {
    return &commandBuffers;
}

void Workflow::raiseUpdateFlags() {
    utils::vkDefault::raiseFlags(commandBuffers);
}

} // moon::workflows
