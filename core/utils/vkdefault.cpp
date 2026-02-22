#include "vkdefault.h"
#include "operations.h"

namespace moon::utils {

VkSamplerCreateInfo vkDefault::sampler(){
    VkSamplerCreateInfo SamplerInfo{};
        SamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        SamplerInfo.magFilter = VK_FILTER_LINEAR;
        SamplerInfo.minFilter = VK_FILTER_LINEAR;
        SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.anisotropyEnable = VK_TRUE;
        SamplerInfo.maxAnisotropy = 1.0f;
        SamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        SamplerInfo.unnormalizedCoordinates = VK_FALSE;
        SamplerInfo.compareEnable = VK_FALSE;
        SamplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        SamplerInfo.minLod = 0.0f;
        SamplerInfo.maxLod = 0.0f;
        SamplerInfo.mipLodBias = 0.0f;
    return SamplerInfo;
}

VkPipelineInputAssemblyStateCreateInfo vkDefault::inputAssembly(){
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;
    return inputAssembly;
}

VkViewport vkDefault::viewport(VkOffset2D offset, VkExtent2D extent){
    VkViewport viewport{};
        viewport.x = static_cast<float>(offset.x);
        viewport.y = static_cast<float>(offset.y);
        viewport.width  = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
    return viewport;
}

VkRect2D vkDefault::scissor(VkOffset2D offset, VkExtent2D extent){
    VkRect2D scissor{};
        scissor.offset = offset;
        scissor.extent = extent;
    return scissor;
}

VkPipelineViewportStateCreateInfo vkDefault::viewportState(const VkViewport* viewport, const VkRect2D* scissor){
    VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = scissor;
    return viewportState;
}

VkPipelineRasterizationStateCreateInfo vkDefault::rasterizationState(){
    VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
    return rasterizer;
}

VkPipelineRasterizationStateCreateInfo vkDefault::rasterizationState(VkFrontFace frontFace){
    VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = frontFace;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;
    return rasterizer;
}

VkPipelineMultisampleStateCreateInfo vkDefault::multisampleState(){
    VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;
    return multisampling;
}

VkPipelineDepthStencilStateCreateInfo vkDefault::depthStencilDisable(){
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};
    return depthStencil;
}

VkPipelineDepthStencilStateCreateInfo vkDefault::depthStencilEnable(){
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f;
        depthStencil.maxDepthBounds = 1.0f;
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {};
        depthStencil.back = {};
    return depthStencil;
}

VkPipelineColorBlendAttachmentState vkDefault::colorBlendAttachmentState(VkBool32 enable){
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = enable;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_MAX;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_MAX;
    return colorBlendAttachment;
}

VkPipelineColorBlendStateCreateInfo vkDefault::colorBlendState(uint32_t attachmentCount, const VkPipelineColorBlendAttachmentState* pAttachments){
    VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = attachmentCount;
        colorBlending.pAttachments = pAttachments;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;
    return colorBlending;
}

vkDefault::SubpassInfos vkDefault::subpassInfos(uint32_t attachmentCount) {
    vkDefault::SubpassInfos subpassInfos;
    auto& subpass = subpassInfos.emplace_back();
    subpass.out.resize(attachmentCount);
    for (uint32_t i = 0; i < attachmentCount; i++) {
        subpass.out[i] = VkAttachmentReference{i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    return subpassInfos;
}

VkPipelineVertexInputStateCreateInfo vkDefault::vertexInputState(){
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    return vertexInputInfo;
}

VkDescriptorSetLayoutBinding vkDefault::bufferVertexLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::bufferFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::imageFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::inAttachmentFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::imageComputeLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

VkDescriptorSetLayoutBinding vkDefault::bufferComputeLayoutBinding(const uint32_t& binding, const uint32_t& count){
    VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBinding.pImmutableSamplers = VK_NULL_HANDLE;
    return layoutBinding;
}

#define VKDEFAULT_MAKE_DESCRIPTOR(Name, BaseDescriptor)                                 \
	vkDefault::Name::Name(vkDefault::Name&& other) noexcept {                           \
        swap(other);                                                                    \
    }                                                                                   \
	vkDefault::Name& vkDefault::Name::operator=(vkDefault::Name&& other) noexcept {     \
        swap(other);                                                                    \
        return *this;                                                                   \
    }                                                                                   \
	vkDefault::Name::operator const BaseDescriptor&() const {                           \
        return descriptor;                                                              \
    }                                                                                   \
	vkDefault::Name::operator const BaseDescriptor*() const {                           \
        return &descriptor;                                                             \
    }                                                                                   \
    vkDefault::Name::operator const bool() const {                                      \
        return descriptor != VK_NULL_HANDLE;                                            \
    }

#define VKDEFAULT_MAKE_SWAP(Name)				                                        \
    void vkDefault::Name::swap(vkDefault::Name& other) noexcept {                       \
        uint8_t buff[sizeof(Name)];                                                     \
        std::memcpy((void*)buff, (void*)&other, sizeof(Name));                          \
        std::memcpy((void*)&other, (void*)this, sizeof(Name));                          \
        std::memcpy((void*)this, (void*)buff, sizeof(Name));                            \
    }

template<typename Descriptor>
Descriptor release(Descriptor& descriptor) {
    Descriptor temp = descriptor;
    descriptor = VK_NULL_HANDLE;
    return temp;
}

vkDefault::Pipeline::Pipeline(VkDevice device, const std::vector<VkGraphicsPipelineCreateInfo>& graphicsPipelineCreateInfos) : device(device) {
    CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, static_cast<uint32_t>(graphicsPipelineCreateInfos.size()), graphicsPipelineCreateInfos.data(), nullptr, &descriptor));
}

vkDefault::Pipeline::Pipeline(VkDevice device, const VkComputePipelineCreateInfo& computePipelineCreateInfo) : device(device) {
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &descriptor));
}

vkDefault::Pipeline::~Pipeline() {
    if (descriptor) vkDestroyPipeline(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Pipeline)
VKDEFAULT_MAKE_DESCRIPTOR(Pipeline, VkPipeline)

vkDefault::PipelineLayout::PipelineLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
    const std::vector<VkPushConstantRange>& pushConstantRange) : device(device)
{
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
        pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
        pipelineLayoutCreateInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRange.size());
        pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRange.data();
    CHECK(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &descriptor));
}

vkDefault::PipelineLayout::~PipelineLayout() {
    if (descriptor) vkDestroyPipelineLayout(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(PipelineLayout)
VKDEFAULT_MAKE_DESCRIPTOR(PipelineLayout, VkPipelineLayout)

vkDefault::DescriptorSetLayout::DescriptorSetLayout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings) : bindings(bindings), device(device){
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
    CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptor));
}

vkDefault::DescriptorSetLayout::~DescriptorSetLayout() {
    bindings.clear();
    if (descriptor) vkDestroyDescriptorSetLayout(device, release(descriptor), nullptr);
}

void vkDefault::DescriptorSetLayout::swap(DescriptorSetLayout& other) noexcept {
    std::swap(bindings, other.bindings);
    std::swap(descriptor, other.descriptor);
    std::swap(device, other.device);
}

VKDEFAULT_MAKE_DESCRIPTOR(DescriptorSetLayout, VkDescriptorSetLayout)

vkDefault::ShaderModule::ShaderModule(VkDevice device, const std::filesystem::path& shaderPath) :
    shaderModule(moon::utils::shaderModule::create(device, moon::utils::shaderModule::readFile(shaderPath))), device(device)
{}

vkDefault::ShaderModule::~ShaderModule() {
    if (shaderModule) {
        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule = VK_NULL_HANDLE;
    }
}

vkDefault::ShaderModule::operator const VkShaderModule& () const {
    return shaderModule;
}

vkDefault::FragmentShaderModule::~FragmentShaderModule() {
    pipelineShaderStageCreateInfo = VkPipelineShaderStageCreateInfo{};
    specializationInfo = VkSpecializationInfo{};
}

vkDefault::FragmentShaderModule::FragmentShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specInfo) :
    ShaderModule(device, shaderPath), specializationInfo(specInfo) {
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    pipelineShaderStageCreateInfo.module = shaderModule;
    pipelineShaderStageCreateInfo.pName = "main";
    pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
}

vkDefault::FragmentShaderModule::operator const VkPipelineShaderStageCreateInfo& () const {
    return pipelineShaderStageCreateInfo;
}

vkDefault::VertrxShaderModule::~VertrxShaderModule() {
    pipelineShaderStageCreateInfo = VkPipelineShaderStageCreateInfo{};
    specializationInfo = VkSpecializationInfo{};
}

vkDefault::VertrxShaderModule::VertrxShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specInfo) :
    ShaderModule(device, shaderPath), specializationInfo(specInfo) {
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    pipelineShaderStageCreateInfo.module = shaderModule;
    pipelineShaderStageCreateInfo.pName = "main";
    pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
}

vkDefault::VertrxShaderModule::operator const VkPipelineShaderStageCreateInfo& () const {
    return pipelineShaderStageCreateInfo;
}

vkDefault::ComputeShaderModule::~ComputeShaderModule() {
    pipelineShaderStageCreateInfo = VkPipelineShaderStageCreateInfo{};
}

vkDefault::ComputeShaderModule::ComputeShaderModule(VkDevice device, const std::filesystem::path& shaderPath) :
    ShaderModule(device, shaderPath) {
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = shaderModule;
    pipelineShaderStageCreateInfo.pName = "main";
}

vkDefault::ComputeShaderModule::operator const VkPipelineShaderStageCreateInfo& () const {
    return pipelineShaderStageCreateInfo;
}

vkDefault::RenderPass::RenderPass(VkDevice device, const AttachmentDescriptions& attachments, const vkDefault::SubpassInfos& subpassInfos, const SubpassDependencies& dependencies) : device(device) {
    utils::vkDefault::RenderPass::SubpassDescriptions subpasses;
    for (const auto& subpassInfo : subpassInfos) {
        auto& subpass = subpasses.emplace_back();
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<uint32_t>(subpassInfo.out.size());
        subpass.pColorAttachments = subpassInfo.out.data();
        subpass.inputAttachmentCount = static_cast<uint32_t>(subpassInfo.in.size());
        subpass.pInputAttachments = subpassInfo.in.data();
        subpass.pDepthStencilAttachment = subpassInfo.depth.data();
    }

    VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
        renderPassInfo.pSubpasses = subpasses.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

    CHECK(vkCreateRenderPass(device, &renderPassInfo, nullptr, &descriptor));
}

vkDefault::RenderPass::~RenderPass() {
    if (descriptor) vkDestroyRenderPass(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(RenderPass)
VKDEFAULT_MAKE_DESCRIPTOR(RenderPass, VkRenderPass)

vkDefault::Framebuffer::Framebuffer(VkDevice device, const VkFramebufferCreateInfo& framebufferInfo) : device(device) {
    CHECK(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &descriptor));
}

vkDefault::Framebuffer::~Framebuffer() {
    if (descriptor) vkDestroyFramebuffer(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Framebuffer)
VKDEFAULT_MAKE_DESCRIPTOR(Framebuffer, VkFramebuffer)

vkDefault::Instance::~Instance() {
    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

vkDefault::Instance::Instance(const VkInstanceCreateInfo& createInfo) {
    CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
}

vkDefault::Instance::operator const VkInstance& () const {
    return instance;
}

vkDefault::DebugUtilsMessenger::~DebugUtilsMessenger() {
    if (debugUtilsMessenger) {
        moon::utils::validationLayer::destroyDebugUtilsMessengerEXT(instance, debugUtilsMessenger, nullptr);
        debugUtilsMessenger = VK_NULL_HANDLE;
    }
}

vkDefault::DebugUtilsMessenger::DebugUtilsMessenger(const VkInstance& instance) : instance(instance) {
    moon::utils::validationLayer::setupDebugMessenger(instance, &debugUtilsMessenger);
}

vkDefault::DebugUtilsMessenger::operator const VkDebugUtilsMessengerEXT& () const {
    return debugUtilsMessenger;
}

vkDefault::Surface::~Surface() {
    if (surface) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }
}

vkDefault::Surface::Surface(const VkInstance& instance, Window* window) : instance(instance) {
    if (window) {
        CHECK(window->createSurface(instance, &surface));
    }
}

vkDefault::Surface::operator const VkSurfaceKHR& () const {
    return surface;
}

vkDefault::Semaphore::Semaphore(const VkDevice& device) : device(device) {
    VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    CHECK(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &descriptor));
}

vkDefault::Semaphore::~Semaphore() {
    if(descriptor) vkDestroySemaphore(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Semaphore)
VKDEFAULT_MAKE_DESCRIPTOR(Semaphore, VkSemaphore)

vkDefault::Fence::Fence(const VkDevice& device) : device(device) {
    VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    CHECK(vkCreateFence(device, &fenceInfo, nullptr, &descriptor));
}

vkDefault::Fence::~Fence() {
    if (descriptor) vkDestroyFence(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Fence)
VKDEFAULT_MAKE_DESCRIPTOR(Fence, VkFence)

vkDefault::Sampler::Sampler(const VkDevice& device, const VkSamplerCreateInfo& samplerInfo) : device(device) {
    CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &descriptor));
}

vkDefault::Sampler::~Sampler() {
    if (descriptor) vkDestroySampler(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(Sampler)
VKDEFAULT_MAKE_DESCRIPTOR(Sampler, VkSampler)

vkDefault::DescriptorPool::DescriptorPool(const VkDevice& device, const VkDescriptorPoolCreateInfo& poolInfo) : device(device) {
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptor));
}

vkDefault::DescriptorPool::DescriptorPool(const VkDevice& device, const std::vector<const vkDefault::DescriptorSetLayout*>& descriptorSetLayouts, const uint32_t descriptorsCount) : device(device) {
    std::unordered_map<VkDescriptorType, uint32_t> aggregated;
    for (const vkDefault::DescriptorSetLayout* descriptorSetLayout : descriptorSetLayouts) {
        for (const VkDescriptorSetLayoutBinding& binding : descriptorSetLayout->bindings) {
            aggregated[binding.descriptorType] += static_cast<uint32_t>(binding.descriptorCount) * descriptorsCount;
        }
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    uint32_t maxSets = descriptorsCount * static_cast<uint32_t>(descriptorSetLayouts.size());
    poolSizes.reserve(aggregated.size());
    for (const auto& [type, count] : aggregated) {
        VkDescriptorPoolSize descriptorPoolSize{};
        descriptorPoolSize.type = type;
        descriptorPoolSize.descriptorCount = count;
        poolSizes.push_back(descriptorPoolSize);
    }

    VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = maxSets;
    CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptor));
}

vkDefault::DescriptorSets vkDefault::DescriptorPool::allocateDescriptorSets(const vkDefault::DescriptorSetLayout& descriptorSetLayout, const uint32_t& descriptorSetCount) {
    vkDefault::DescriptorSets descriptorSets(descriptorSetCount);
    std::vector<VkDescriptorSetLayout> layouts(descriptorSetCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptor;
        allocInfo.descriptorSetCount = descriptorSetCount;
        allocInfo.pSetLayouts = layouts.data();
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()));
    return descriptorSets;
}

VkDescriptorSet vkDefault::DescriptorPool::allocateDescriptorSet(const vkDefault::DescriptorSetLayout& descriptorSetLayout) {
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptor;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = descriptorSetLayout;
    CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    return descriptorSet;
}

vkDefault::DescriptorPool::~DescriptorPool() {
    if (descriptor) vkDestroyDescriptorPool(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(DescriptorPool)
VKDEFAULT_MAKE_DESCRIPTOR(DescriptorPool, VkDescriptorPool)

vkDefault::ImageView::ImageView(
    const VkDevice& device,
    const VkImage& image,
    VkImageViewType type,
    VkFormat format,
    VkImageAspectFlags aspectFlags,
    uint32_t mipLevels,
    uint32_t baseArrayLayer,
    uint32_t layerCount) : device(device) {
    CHECK(utils::texture::createView(device, type, format, aspectFlags, mipLevels, baseArrayLayer, layerCount, image, &descriptor));
}

vkDefault::ImageView::~ImageView() {
    if (descriptor) vkDestroyImageView(device, release(descriptor), nullptr);
}

VKDEFAULT_MAKE_SWAP(ImageView)
VKDEFAULT_MAKE_DESCRIPTOR(ImageView, VkImageView)

vkDefault::Image::Image(
    VkPhysicalDevice                physicalDevice,
    VkDevice                        device,
    VkImageCreateFlags              flags,
    VkExtent3D                      extent,
    uint32_t                        arrayLayers,
    uint32_t                        mipLevels,
    VkSampleCountFlagBits           numSamples,
    VkFormat                        format,
    VkImageLayout                   layout,
    VkImageUsageFlags               usage,
    VkMemoryPropertyFlags           properties) : device(device) {
    CHECK(utils::texture::create(physicalDevice, device, flags, extent, arrayLayers, mipLevels, numSamples, format, layout, usage, properties, &descriptor, &memory));
}

vkDefault::Image::~Image() {
    utils::texture::destroy(device, descriptor, memory);
}

vkDefault::Image::operator const VkDeviceMemory& () const {
    return memory;
}

vkDefault::Image::operator const VkDeviceMemory* () const {
    return &memory;
}

VKDEFAULT_MAKE_SWAP(Image)
VKDEFAULT_MAKE_DESCRIPTOR(Image, VkImage)

vkDefault::Buffer::~Buffer() {
    utils::buffer::destroy(device, descriptor, memory);
}

vkDefault::Buffer::Buffer(
    VkPhysicalDevice                physicalDevice,
    VkDevice                        device,
    VkDeviceSize                    size,
    VkBufferUsageFlags              usage,
    VkMemoryPropertyFlags           properties) : memorysize(size), device(device) {
    CHECK(utils::buffer::create(physicalDevice, device, size, usage, properties, &descriptor, &memory));
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT || properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        CHECK(vkMapMemory(device, memory, 0, size, 0, &memorymap));
    }
}

void vkDefault::Buffer::copy(const void* data, VkDeviceSize offset, VkDeviceSize size) {
    if (size == 0) {
        return;
    }

    CHECK_M(memorymap != nullptr, std::string("[ vkDefault::Buffer::copy(offset) ] buffer is not host visible or not mapped"));

    if (size == maxsize) {
        size = memorysize;
    }

    CHECK_M(offset + size <= memorysize, std::string("[ vkDefault::Buffer::copy(offset) ] out of bounds"));

    if (memorymap) {
        auto dst = static_cast<uint8_t*>(memorymap) + offset;
        std::memcpy(dst, data, size);
    }
}

size_t vkDefault::Buffer::size() const {
    return memorysize;
}

void vkDefault::Buffer::raiseFlag() {
    updateFlag = true;
}

bool vkDefault::Buffer::dropFlag() {
    bool temp = updateFlag;
    updateFlag = false;
    return temp;
}

void* &vkDefault::Buffer::map() {
    return memorymap;
}

vkDefault::Buffer::operator const VkDeviceMemory& () const {
    return memory;
}

vkDefault::Buffer::operator const VkDeviceMemory* () const {
    return &memory;
}


VkDescriptorBufferInfo vkDefault::Buffer::descriptorBufferInfo() const {
    return VkDescriptorBufferInfo{ descriptor, 0, size() };
}

VKDEFAULT_MAKE_SWAP(Buffer)
VKDEFAULT_MAKE_DESCRIPTOR(Buffer, VkBuffer)

VkResult vkDefault::SwapchainKHR::reset(const VkDevice& logical, const utils::vkDefault::ImageInfo& info, const utils::swapChain::SupportDetails& supportDetails, const std::vector<uint32_t>& queueFamilyIndices, VkSurfaceKHR surface, VkSurfaceFormatKHR surfaceFormat) {
    if (descriptor) vkDestroySwapchainKHR(device, release(descriptor), nullptr);
    device = logical;
    imageInfo = info;

    VkSwapchainCreateInfoKHR createInfo {};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageInfo.Count;
        createInfo.imageFormat = imageInfo.Format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = imageInfo.Extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        createInfo.imageSharingMode = queueFamilyIndices.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
        createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
        createInfo.preTransform = supportDetails.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = swapChain::queryingPresentMode(supportDetails.presentModes);
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
    return vkCreateSwapchainKHR(device, &createInfo, nullptr, &descriptor);
}

vkDefault::SwapchainKHR::~SwapchainKHR() {
    if (descriptor) vkDestroySwapchainKHR(device, release(descriptor), nullptr);
}

std::vector<VkImage> vkDefault::SwapchainKHR::images() const {
    uint32_t count = imageInfo.Count;
    std::vector<VkImage> result(count);
    CHECK(vkGetSwapchainImagesKHR(device, descriptor, &count, result.data()));
    return result;
}

vkDefault::SwapchainKHR::operator const VkSwapchainKHR& () const {
    return descriptor;
}

vkDefault::SwapchainKHR::operator const VkSwapchainKHR* () const {
    return &descriptor;
}

vkDefault::SwapchainKHR::operator const bool() const {
    return descriptor != VK_NULL_HANDLE;
}

vkDefault::CommandPool::CommandPool(const VkDevice& device) : device(device) {
    VkCommandPoolCreateInfo poolInfo {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &descriptor));
}

void vkDefault::CommandPool::swap(CommandPool& other) noexcept {
    std::swap(device, other.device);
    std::swap(descriptor, other.descriptor);
    std::swap(commandBuffers, other.commandBuffers);
}

vkDefault::CommandBuffers vkDefault::CommandPool::allocateCommandBuffers(const uint32_t& commandBuffersCount) const {
    vkDefault::CommandBuffers commandBuffers(commandBuffersCount);
    for (auto& commandBuffer: commandBuffers) {
        commandBuffer = vkDefault::CommandBuffer(device, descriptor);
    }
    return commandBuffers;
}

vkDefault::CommandPool::~CommandPool() {
    if (descriptor) vkDestroyCommandPool(device, release(descriptor), nullptr);
    for (auto& commandBuffer: commandBuffers) {
        vkFreeCommandBuffers(device, descriptor, 1, &commandBuffer);
    }
    commandBuffers.clear();
}

VKDEFAULT_MAKE_DESCRIPTOR(CommandPool, VkCommandPool)

vkDefault::CommandBuffer::CommandBuffer(const VkDevice& device, VkCommandPool commandPool) : commandPool(commandPool), device(device) {
    VkCommandBufferAllocateInfo allocInfo {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    CHECK(vkAllocateCommandBuffers(device, &allocInfo, &descriptor));
}

vkDefault::CommandBuffer::~CommandBuffer() {
    descriptor = VK_NULL_HANDLE;
}

VkResult vkDefault::CommandBuffer::reset() const {
    return vkResetCommandBuffer(descriptor, 0);
}

VkResult vkDefault::CommandBuffer::begin() const {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;
    return vkBeginCommandBuffer(descriptor, &beginInfo);
}

VkResult vkDefault::CommandBuffer::end() const {
    return vkEndCommandBuffer(descriptor);
}

void vkDefault::CommandBuffer::raiseFlag() {
    updateFlag = true;
}

bool vkDefault::CommandBuffer::dropFlag() {
    bool temp = updateFlag;
    updateFlag = false;
    return temp;
}

VKDEFAULT_MAKE_SWAP(CommandBuffer)
VKDEFAULT_MAKE_DESCRIPTOR(CommandBuffer, VkCommandBuffer)

}
