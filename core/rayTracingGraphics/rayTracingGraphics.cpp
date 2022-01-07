#include "rayTracingGraphics.h"

#include <cstring>

#include <utils/swapChain.h>
#include <utils/vkdefault.h>
#include <utils/buffer.h>
#include <utils/texture.h>

namespace moon::rayTracingGraphics {

RayTracingGraphics::RayTracingGraphics(const std::filesystem::path& shadersPath, const std::filesystem::path& workflowsShadersPath, VkExtent2D extent)
    : shadersPath(shadersPath), workflowsShadersPath(workflowsShadersPath), extent(extent) {
    setExtent(extent);
    link = std::make_unique<RayTracingLink>();
}

RayTracingGraphics::ImageResource::ImageResource(const std::string& id, const moon::utils::PhysicalDevice& phDevice, const moon::utils::vkDefault::ImageInfo& imageInfo){
    this->id = id;

    host.resize(imageInfo.Extent.width * imageInfo.Extent.height);

    hostDevice = utils::vkDefault::Buffer(
        phDevice,
        phDevice.device(),
        sizeof(uint32_t) * host.size(),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    device = utils::Attachments(
        phDevice,
        phDevice.device(),
        imageInfo,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

RayTracingGraphics::ImageResource::ImageResource(ImageResource&& other) noexcept {
    swap(other);
}

RayTracingGraphics::ImageResource& RayTracingGraphics::ImageResource::operator=(ImageResource&& other) noexcept {
    swap(other);
    return *this;
}

void RayTracingGraphics::ImageResource::swap(ImageResource& other) noexcept {
    std::swap(id, other.id);
    std::swap(host, other.host);
    std::swap(hostDevice, other.hostDevice);
    std::swap(device, other.device);
}

void RayTracingGraphics::ImageResource::moveFromHostToHostDevice(VkExtent2D extent){
    hostDevice.copy(host.data());
}

void RayTracingGraphics::ImageResource::copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, uint32_t imageIndex){
    moon::utils::texture::transitionLayout(commandBuffer, device.image(imageIndex), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    moon::utils::texture::copy(commandBuffer, hostDevice, device.image(imageIndex), {extent.width, extent.height, 1}, 1);
    moon::utils::texture::transitionLayout(commandBuffer, device.image(imageIndex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
}

void RayTracingGraphics::reset()
{
    aDatabase.destroy();
    commandPool = utils::vkDefault::CommandPool(device->device());

    emptyTexture = utils::Texture::empty(*device, commandPool);
    aDatabase.addEmptyTexture("black", &emptyTexture);

    moon::utils::vkDefault::ImageInfo imageInfo{ resourceCount, swapChainKHR->info().Format, extent, VK_SAMPLE_COUNT_1_BIT };

    color = ImageResource("color", *device, imageInfo);
    aDatabase.addAttachmentData(color.id, true, &color.device);

    bloom = ImageResource("bloom", *device, imageInfo);
    aDatabase.addAttachmentData(bloom.id, true, &bloom.device);

    bloomParams.in.bloom = bloom.id;
    bloomParams.out.bloom = "finalBloom";
    bloomParams.enable = bloomEnable;
    bloomParams.blitAttachmentsCount = 8;
    bloomParams.inputImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bloomParams.imageInfo = imageInfo;
    bloomParams.shadersPath = workflowsShadersPath;

    bloomGraph = std::make_unique<moon::workflows::BloomGraphics>(bloomParams);
    bloomGraph->setDeviceProp(*device, device->device());
    bloomGraph->create(commandPool, aDatabase);
    bloomGraph->updateDescriptors(bDatabase, aDatabase);

    moon::utils::vkDefault::ImageInfo bbInfo{ resourceCount, swapChainKHR->info().Format, extent, VK_SAMPLE_COUNT_1_BIT};
    std::string bbId = "bb";
    bbGraphics.create(*device, device->device(), bbInfo, shadersPath);
    aDatabase.addAttachmentData(bbId, bbGraphics.getEnable(), &bbGraphics.getAttachments());

    RayTracingLinkParameters linkParams;
    linkParams.in.color = color.id;
    linkParams.in.bloom = bloomParams.out.bloom;
    linkParams.in.boundingBox = bbId;
    linkParams.shadersPath = shadersPath;
    linkParams.imageInfo = utils::vkDefault::ImageInfo{ resourceCount, swapChainKHR->info().Format, swapChainKHR->info().Extent, VK_SAMPLE_COUNT_1_BIT };

    link = std::make_unique<RayTracingLink>(device->device(), linkParams, link->renderPass(), aDatabase);

    rayTracer.create();
}

utils::vkDefault::VkSemaphores RayTracingGraphics::submit(uint32_t frameIndex, const utils::vkDefault::VkSemaphores& externalSemaphore)
{
    rayTracer.calculateImage(color.host.data(), bloom.host.data());

    color.moveFromHostToHostDevice(extent);
    bloom.moveFromHostToHostDevice(extent);

    bloomGraph->update(frameIndex);

    std::vector<VkCommandBuffer> commandBuffers;
    auto& commandBuffer = commandBuffers.emplace_back(moon::utils::singleCommandBuffer::create(device->device(), commandPool));
    color.copyToDevice(commandBuffer, extent, frameIndex);
    bloom.copyToDevice(commandBuffer, extent, frameIndex);
    bbGraphics.render(commandBuffer, frameIndex);
    CHECK(vkEndCommandBuffer(commandBuffer));

    const utils::vkDefault::CommandBuffers& bloomCommandBuffers = *bloomGraph;
    commandBuffers.push_back(bloomCommandBuffers[frameIndex]);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    submitInfo.pCommandBuffers = commandBuffers.data();
    CHECK(vkQueueSubmit(device->device()(0, 0), 1, &submitInfo, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(device->device()(0, 0)));

    return {};
}

void RayTracingGraphics::update(uint32_t imageIndex) {
    rayTracer.update();
    bbGraphics.update(imageIndex);
}

void RayTracingGraphics::setEnableBoundingBox(bool enable){
    bbGraphics.setEnable(enable);
}

void RayTracingGraphics::setEnableBloom(bool enable){
    bloomEnable = enable;
}

void RayTracingGraphics::setBlitFactor(const float& blitFactor){
    bloomParams.blitFactor = blitFactor;
    bloomGraph->raiseUpdateFlags();
}

void RayTracingGraphics::setExtent(VkExtent2D extent){
    this->extent = extent;
    rayTracer.setExtent(extent.width, extent.height);
}

void RayTracingGraphics::bind(cuda::rayTracing::Object* obj) {
    rayTracer.bind(obj);
}

void RayTracingGraphics::setCamera(cuda::rayTracing::Devicep<cuda::rayTracing::Camera>* cam){
    rayTracer.setCamera(cam);
    bbGraphics.bind(cam);
}

void RayTracingGraphics::clearFrame(){
    rayTracer.clearFrame();
}

void RayTracingGraphics::buildTree(){
    rayTracer.buildTree();
}

void RayTracingGraphics::buildBoundingBoxes(bool primitive, bool tree, bool onlyLeafs){
    bbGraphics.clear();

    if(tree){
        std::stack<cuda::rayTracing::KDNode<std::vector<const cuda::rayTracing::Primitive*>::iterator>*> stack;
        stack.push(rayTracer.getTree().getRoot());
        for(;!stack.empty();){
            const auto top = stack.top();
            stack.pop();

            if(!onlyLeafs || !(top->left || top->right)){
                std::random_device device;
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                cuda::rayTracing::cbox box(top->bbox, cuda::rayTracing::vec4f(dist(device), dist(device), dist(device), 1.0f));
                bbGraphics.bind(std::move(box));
            }

            if(top->right) stack.push(top->right);
            if(top->left) stack.push(top->left);
        }
    }

    if(primitive){
        for(auto& primitive: rayTracer.getTree().storage){
            cuda::rayTracing::cbox box(primitive->bbox, cuda::rayTracing::vec4f(1.0, 0.0, 0.0, 1.0f));
            bbGraphics.bind(box);
        }
    }
}
}
