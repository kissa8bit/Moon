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

void RayTracingGraphics::ImageResource::copyToDevice(VkCommandBuffer commandBuffer, VkExtent2D extent, utils::ResourceIndex resourceIndex){
    const auto imageIndex = resourceIndex.get();
    moon::utils::texture::transitionLayout(commandBuffer, device.image(imageIndex), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 0, 1);
    moon::utils::texture::copy(commandBuffer, hostDevice, device.image(imageIndex), {extent.width, extent.height, 1}, 1);
    moon::utils::texture::transitionLayout(commandBuffer, device.image(imageIndex), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 0, 1);
}

void RayTracingGraphics::reset()
{
    aDatabase.destroy();
    commandPool = utils::vkDefault::CommandPool(device->device());
    signalSemaphores.resize(resourceCount);
    for (auto& sem : signalSemaphores) {
        sem = utils::vkDefault::Semaphore(device->device());
    }

    emptyTexture = utils::Texture::createEmpty(*device, commandPool);
    aDatabase.addEmptyTexture(utils::ImageName("black"), &emptyTexture);

    moon::utils::vkDefault::ImageInfo imageInfo{ resourceCount, swapChainKHR->info().Format, extent, VK_SAMPLE_COUNT_1_BIT };

    color = ImageResource("color", *device, imageInfo);
    aDatabase.addAttachmentData(utils::AttachmentName(color.id), &color.device);

    bloom = ImageResource("bloom", *device, imageInfo);
    aDatabase.addAttachmentData(utils::AttachmentName(bloom.id), &bloom.device);

    bloomParams.in.bloom = utils::AttachmentName(bloom.id);
    bloomParams.out.bloom = utils::AttachmentName("finalBloom");
    bloomParams.enable = bloomEnable;
    bloomParams.attachmentsCount = 8;
    bloomParams.inputImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bloomParams.imageInfo = imageInfo;
    bloomParams.shadersPath = workflowsShadersPath;

    bloomGraph = std::make_unique<moon::workflows::BloomGraphics>(bloomParams);
    bloomGraph->setDeviceProp(*device, device->device());
    bloomGraph->create(commandPool, aDatabase);
    bloomGraph->updateDescriptors(bDatabase, aDatabase);

    moon::utils::vkDefault::ImageInfo bbInfo{ resourceCount, swapChainKHR->info().Format, extent, VK_SAMPLE_COUNT_1_BIT};
    std::string bbId = "bb";
    bbGraphics.create(*device, device->device(), bbInfo, shadersPath);
    aDatabase.addAttachmentData(utils::AttachmentName(bbId), &bbGraphics.getAttachments());

    RayTracingLinkParameters linkParams;
    linkParams.in.color = utils::AttachmentName(color.id);
    linkParams.in.bloom = bloomParams.out.bloom;
    linkParams.in.boundingBox = utils::AttachmentName(bbId);
    linkParams.shadersPath = shadersPath;
    linkParams.imageInfo = utils::vkDefault::ImageInfo{ resourceCount, swapChainKHR->info().Format, swapChainKHR->info().Extent, VK_SAMPLE_COUNT_1_BIT };

    linkMember = RayTracingLink(device->device(), linkParams, pRenderPass, aDatabase);

    rayTracer.create();
}

utils::vkDefault::VkSemaphores RayTracingGraphics::submit(utils::ResourceIndex resourceIndex, const utils::vkDefault::VkSemaphores& externalSemaphore)
{
    rayTracer.calculateImage(color.host.data(), bloom.host.data());

    color.moveFromHostToHostDevice(extent);
    bloom.moveFromHostToHostDevice(extent);

    bloomGraph->update(resourceIndex);

    std::vector<VkCommandBuffer> commandBuffers;
    auto& commandBuffer = commandBuffers.emplace_back(moon::utils::singleCommandBuffer::create(device->device(), commandPool));
    color.copyToDevice(commandBuffer, extent, resourceIndex);
    bloom.copyToDevice(commandBuffer, extent, resourceIndex);
    bbGraphics.render(commandBuffer, resourceIndex);
    CHECK(vkEndCommandBuffer(commandBuffer));

    const utils::vkDefault::CommandBuffers& bloomCommandBuffers = *bloomGraph;
    commandBuffers.push_back(bloomCommandBuffers[resourceIndex.get()]);

    std::vector<VkPipelineStageFlags> waitStages(externalSemaphore.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    const VkSemaphore signalSem = signalSemaphores.at(resourceIndex.get());

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = static_cast<uint32_t>(externalSemaphore.size());
    submitInfo.pWaitSemaphores = externalSemaphore.empty() ? nullptr : externalSemaphore.data();
    submitInfo.pWaitDstStageMask = externalSemaphore.empty() ? nullptr : waitStages.data();
    submitInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    submitInfo.pCommandBuffers = commandBuffers.data();
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSem;
    CHECK(vkQueueSubmit(device->device()(0, 0), 1, &submitInfo, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(device->device()(0, 0)));

    return {signalSem};
}

void RayTracingGraphics::update(utils::ResourceIndex resourceIndex) {
    rayTracer.update();
    bbGraphics.update(resourceIndex);
}

void RayTracingGraphics::setEnableBoundingBox(bool enable){
    bbGraphics.setEnable(enable);
}

void RayTracingGraphics::setEnableBloom(bool enable){
    bloomEnable = enable;
}

void RayTracingGraphics::setBlitFactor(const float& blitFactor){
    (void)blitFactor;
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
    using namespace cuda::rayTracing;
    using NodePtr = KDNode<std::vector<const Primitive*>::iterator>*;

    bbGraphics.clear();

    std::random_device rdev;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Transform a local-space AABB to a world-space AABB via toWorld.
    auto transformBox = [](const box& local, const mat4f& M, const vec4f& col) {
        box world;
        for (int m = 0; m < 8; m++) {
            const vec4f c(
                (m & 1) ? local.max[0] : local.min[0],
                (m & 2) ? local.max[1] : local.min[1],
                (m & 4) ? local.max[2] : local.min[2],
                1.0f);
            const vec4f wc = M * c;
            world.min = min(world.min, wc);
            world.max = max(world.max, wc);
        }
        return cbox(world, col);
    };

    // Traverse a host KD-tree, emitting boxes in world space.
    using HostTree = KDTree<std::vector<const Primitive*>>;
    auto traverseTree = [&](const HostTree* hostTree, const mat4f& toWorld, const vec4f& col) {
        if (!hostTree->getRoot()) return;
        std::stack<NodePtr> stack;
        stack.push(hostTree->getRoot());
        for (; !stack.empty();) {
            NodePtr top = stack.top(); stack.pop();
            if (!onlyLeafs || !(top->left || top->right))
                bbGraphics.bind(transformBox(top->bbox, toWorld, col));
            if (top->right) stack.push(top->right);
            if (top->left)  stack.push(top->left);
        }
    };

    if (tree) {
        // TLAS: random colour per node — shows object-level bbox hierarchy.
        if (auto* tlasRoot = rayTracer.getTree().getRoot(); tlasRoot) {
            std::stack<NodePtr> stack;
            stack.push(tlasRoot);
            for (; !stack.empty();) {
                NodePtr top = stack.top(); stack.pop();
                if (!onlyLeafs || !(top->left || top->right)) {
                    bbGraphics.bind(cbox(top->bbox, vec4f(dist(rdev), dist(rdev), dist(rdev), 1.0f)));
                }
                if (top->right) stack.push(top->right);
                if (top->left)  stack.push(top->left);
            }
        }

        // Per-object BLAS: local-space KD-trees transformed to world space.
        // Each object gets its own colour so trees are visually distinct.
        for (const auto& info : rayTracer.getBLASInfos()) {
            const vec4f col(dist(rdev), dist(rdev), dist(rdev), 1.0f);
            traverseTree(info.tree, info.toWorld, col);
        }
    }

    if (primitive) {
        // Leaf-level: one box per triangle, in world space.
        for (const auto& info : rayTracer.getBLASInfos()) {
            const vec4f col(dist(rdev), dist(rdev), dist(rdev), 1.0f);
            if (!info.tree->getRoot()) continue;
            std::stack<NodePtr> stack;
            stack.push(info.tree->getRoot());
            for (; !stack.empty();) {
                NodePtr top = stack.top(); stack.pop();
                if (!top->left && !top->right) {
                    for (auto it = top->begin; it != top->end(); ++it)
                        bbGraphics.bind(transformBox((*it)->bbox, info.toWorld, col));
                }
                if (top->right) stack.push(top->right);
                if (top->left)  stack.push(top->left);
            }
        }
    }
}

void RayTracingGraphics::draw(utils::ResourceIndex resourceIndex, VkCommandBuffer commandBuffer) const {
    linkMember.draw(commandBuffer, resourceIndex);
}

}
