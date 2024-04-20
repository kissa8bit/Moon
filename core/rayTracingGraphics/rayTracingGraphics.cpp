#include "rayTracingGraphics.h"

#include "swapChain.h"
#include "vkdefault.h"

#include <cstring>

void rayTracingGraphics::imageResource::create(physicalDevice phDevice, VkFormat format, VkExtent2D extent, uint32_t imageCount){
    host = new uint32_t[extent.width * extent.height];

    Buffer::create(
        phDevice.instance,
        phDevice.getLogical(),
        sizeof(uint32_t) * extent.width * extent.height,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &hostDevice.instance,
        &hostDevice.memory
        );
    vkMapMemory(phDevice.getLogical(), hostDevice.memory, 0, sizeof(uint32_t) * extent.width * extent.height, 0, &hostDevice.map);

    device.create(
        phDevice.instance,
        phDevice.getLogical(),
        format,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        extent,
        imageCount
        );
    VkSamplerCreateInfo SamplerInfo = vkDefault::samler();
    vkCreateSampler(phDevice.getLogical(), &SamplerInfo, nullptr, &device.sampler);
}

void rayTracingGraphics::imageResource::destroy(physicalDevice phDevice){
    if(host){
        delete[] host;
        host = nullptr;
    }
    hostDevice.destroy(phDevice.getLogical());
    device.deleteAttachment(phDevice.getLogical());
    device.deleteSampler(phDevice.getLogical());
}

void rayTracingGraphics::imageResource::moveFromHostToHostDevice(VkExtent2D extent){
    std::memcpy(hostDevice.map, host, sizeof(uint32_t) * extent.width * extent.height);
}

void rayTracingGraphics::imageResource::copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex){
    Texture::transitionLayout(commandBuffer, device.instances[imageIndex].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    Texture::copy(commandBuffer, hostDevice.instance, device.instances[imageIndex].image, {extent.width, extent.height, 1}, 1);
    Texture::transitionLayout(commandBuffer, device.instances[imageIndex].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
}

void rayTracingGraphics::create()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device.getLogical(), &poolInfo, nullptr, &commandPool);

    emptyTexture = createEmptyTexture(device, commandPool);

    color.create(device, format, extent, imageCount);
    bloom.create(device, format, extent, imageCount);

    imageInfo bbInfo{
        imageCount,
        format,
        extent,
        VK_SAMPLE_COUNT_1_BIT
    };

    bbGraphics.create(device.instance, device.getLogical(), bbInfo, shadersPath);

    imageInfo swapChainInfo{
        imageCount,
        format,
        swapChainKHR->getExtent(),
        VK_SAMPLE_COUNT_1_BIT
    };

    Link.setEmptyTexture(emptyTexture);
    Link.setImageCount(imageCount);
    Link.setDeviceProp(device.getLogical());
    Link.setShadersPath(shadersPath);
    Link.createDescriptorSetLayout();
    Link.createPipeline(&swapChainInfo);
    Link.createDescriptorPool();
    Link.createDescriptorSets();
    Link.updateDescriptorSets(&color.device, &bbGraphics.getAttachments());

    rayTracer.create();
}

void rayTracingGraphics::destroy() {
    if(emptyTexture){
        emptyTexture->destroy(device.getLogical());
        delete emptyTexture;
    }

    color.destroy(device);
    bloom.destroy(device);

    if(commandPool) {vkDestroyCommandPool(device.getLogical(), commandPool, nullptr); commandPool = VK_NULL_HANDLE;}
    Link.destroy();
    bbGraphics.destroy();
}

std::vector<std::vector<VkSemaphore>> rayTracingGraphics::submit(const std::vector<std::vector<VkSemaphore>>&, const std::vector<VkFence>&, uint32_t imageIndex)
{
    rayTracer.calculateImage(color.host, bloom.host);

    color.moveFromHostToHostDevice(extent);

    VkCommandBuffer commandBuffer = SingleCommandBuffer::create(device.getLogical(),commandPool);
    color.copyToDevice(commandBuffer, extent, imageIndex);
    bbGraphics.render(commandBuffer, imageIndex);
    SingleCommandBuffer::submit(device.getLogical(),device.getQueue(0,0),commandPool, &commandBuffer);

    return std::vector<std::vector<VkSemaphore>>();
}

void rayTracingGraphics::update(uint32_t imageIndex) {
    rayTracer.update();
    bbGraphics.update(imageIndex);
}