#ifndef MOON_UTILS_VKDEFAULT_H
#define MOON_UTILS_VKDEFAULT_H

#include <vulkan.h>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <list>
#include <map>

#include "operations.h"
#include "queueFamily.h"

namespace moon::utils::vkDefault {

struct ImageInfo {
	uint32_t                Count{ 0 };
	VkFormat                Format{ VK_FORMAT_UNDEFINED };
	VkExtent2D              Extent{ 0, 0 };
	VkSampleCountFlagBits   Samples{ VK_SAMPLE_COUNT_1_BIT };
};

using Paths = std::vector<std::filesystem::path>;

struct SubpassInfo {
	std::vector<VkAttachmentReference> out;
	std::vector<VkAttachmentReference> in;
	std::vector<VkAttachmentReference> depth;
};

using SubpassInfos = std::vector<SubpassInfo>;

template<typename Buffers>
void raiseFlags(Buffers& buffers) {
	for (auto& buffer : buffers) buffer.raiseFlag();
}

VkSamplerCreateInfo sampler();
VkPipelineVertexInputStateCreateInfo vertexInputState();
VkViewport viewport(VkOffset2D offset, VkExtent2D extent);
VkRect2D scissor(VkOffset2D offset, VkExtent2D extent);
VkPipelineViewportStateCreateInfo viewportState(const VkViewport* viewport, const VkRect2D* scissor);
VkPipelineInputAssemblyStateCreateInfo inputAssembly();
VkPipelineRasterizationStateCreateInfo rasterizationState();
VkPipelineRasterizationStateCreateInfo rasterizationState(VkFrontFace frontFace);
VkPipelineMultisampleStateCreateInfo multisampleState();
VkPipelineDepthStencilStateCreateInfo depthStencilDisable();
VkPipelineDepthStencilStateCreateInfo depthStencilEnable();
VkPipelineColorBlendAttachmentState colorBlendAttachmentState(VkBool32 enable);
VkPipelineColorBlendStateCreateInfo colorBlendState(uint32_t attachmentCount, const VkPipelineColorBlendAttachmentState* pAttachments);
SubpassInfos subpassInfos(uint32_t attachmentCount = 1);
VkDescriptorSetLayoutBinding bufferVertexLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding bufferFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding imageFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);
VkDescriptorSetLayoutBinding inAttachmentFragmentLayoutBinding(const uint32_t& binding, const uint32_t& count);

#define VKDEFAULT_INIT_DESCRIPTOR(Name, BaseDescriptor)	\
private:												\
	BaseDescriptor descriptor{ VK_NULL_HANDLE };		\
	VkDevice device{ VK_NULL_HANDLE };					\
public:													\
	~Name();											\
	Name() noexcept = default;							\
	Name(const Name& other) = delete;					\
	Name& operator=(const Name& other) = delete;		\
	Name(Name&& other) noexcept;						\
	Name& operator=(Name&& other) noexcept;				\
	void swap(Name& other) noexcept;					\
	operator const BaseDescriptor&() const;				\
	operator const BaseDescriptor*() const;				\
	operator const bool() const;

using MaskType = uint8_t;

class Pipeline {
	VKDEFAULT_INIT_DESCRIPTOR(Pipeline, VkPipeline)

public:
	Pipeline(VkDevice device, const std::vector<VkGraphicsPipelineCreateInfo>& graphicsPipelineCreateInfos);
};

using PipelineMap = std::unordered_map<MaskType, utils::vkDefault::Pipeline>;

class PipelineLayout {
	VKDEFAULT_INIT_DESCRIPTOR(PipelineLayout, VkPipelineLayout)

public:
	PipelineLayout(
		VkDevice device,
		const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
		const std::vector<VkPushConstantRange>& pushConstantRange = {});
};

using PipelineLayoutMap = std::unordered_map<MaskType, utils::vkDefault::PipelineLayout>;

class DescriptorSetLayout {
	VKDEFAULT_INIT_DESCRIPTOR(DescriptorSetLayout, VkDescriptorSetLayout)

public:
	std::vector<VkDescriptorSetLayoutBinding> bindings;

	DescriptorSetLayout(VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings);
};

using DescriptorSetLayoutMap = std::unordered_map<MaskType, utils::vkDefault::DescriptorSetLayout>;

class ShaderModule {
protected:
	VkShaderModule shaderModule{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };

public:
	virtual ~ShaderModule();
	ShaderModule() = default;
	ShaderModule(const ShaderModule&) = delete;
	ShaderModule& operator=(const ShaderModule&) = delete;
	ShaderModule(ShaderModule&& other) noexcept { swap(other);}
	ShaderModule& operator=(ShaderModule&& other) noexcept { swap(other); return *this;}
	void swap(ShaderModule& other) noexcept {
		std::swap(shaderModule, other.shaderModule);
		std::swap(device, other.device);
	}

	ShaderModule(VkDevice device, const std::filesystem::path& shaderPath);
	operator const VkShaderModule& () const;
};

class FragmentShaderModule : public ShaderModule {
private:
	VkSpecializationInfo specializationInfo{};
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{};

public:
	~FragmentShaderModule();
	FragmentShaderModule() = default;
	FragmentShaderModule(const FragmentShaderModule&) = delete;
	FragmentShaderModule& operator=(const FragmentShaderModule&) = delete;
	FragmentShaderModule(FragmentShaderModule&& other) noexcept { swap(other); }
	FragmentShaderModule& operator=(FragmentShaderModule&& other) noexcept { swap(other); return *this; }
	void swap(FragmentShaderModule& other) noexcept {
		std::swap(specializationInfo, other.specializationInfo);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		ShaderModule::swap(other);
	}

	FragmentShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo = VkSpecializationInfo{});
	operator const VkPipelineShaderStageCreateInfo& () const;
};

class VertrxShaderModule : public ShaderModule {
private:
	VkSpecializationInfo specializationInfo{};
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{};

public:
	~VertrxShaderModule();
	VertrxShaderModule() = default;
	VertrxShaderModule(const VertrxShaderModule&) = delete;
	VertrxShaderModule& operator=(const VertrxShaderModule&) = delete;
	VertrxShaderModule(VertrxShaderModule&& other) noexcept { swap(other); }
	VertrxShaderModule& operator=(VertrxShaderModule&& other) noexcept { swap(other); return *this; }
	void swap(VertrxShaderModule& other) noexcept {
		std::swap(specializationInfo, other.specializationInfo);
		std::swap(pipelineShaderStageCreateInfo, other.pipelineShaderStageCreateInfo);
		ShaderModule::swap(other);
	}

	VertrxShaderModule(VkDevice device, const std::filesystem::path& shaderPath, const VkSpecializationInfo& specializationInfo = VkSpecializationInfo{});
	operator const VkPipelineShaderStageCreateInfo& () const;
};

class RenderPass {
	VKDEFAULT_INIT_DESCRIPTOR(RenderPass, VkRenderPass)

public:
	using AttachmentDescriptions = std::vector<VkAttachmentDescription>;
	using SubpassDescriptions = std::vector<VkSubpassDescription>;
	using SubpassDependencies = std::vector<VkSubpassDependency>;

	RenderPass(VkDevice device, const AttachmentDescriptions& attachments, const SubpassInfos& subpassInfos, const SubpassDependencies& dependencies);
};

class Framebuffer {
	VKDEFAULT_INIT_DESCRIPTOR(Framebuffer, VkFramebuffer)

public:
	Framebuffer(VkDevice device, const VkFramebufferCreateInfo& framebufferInfo);
};

using Framebuffers = std::vector<Framebuffer>;

class Instance {
private:
	VkInstance instance{ VK_NULL_HANDLE };

public:
	~Instance();
	Instance() = default;
	Instance(const Instance&) = delete;
	Instance& operator=(const Instance&) = delete;
	Instance(Instance&& other) noexcept { std::swap(instance, other.instance); }
	Instance& operator=(Instance&& other) noexcept { std::swap(instance, other.instance); return *this; }

	Instance(const VkInstanceCreateInfo& createInfo);
	operator const VkInstance& () const;
};

class DebugUtilsMessenger {
private:
	VkDebugUtilsMessengerEXT debugUtilsMessenger{ VK_NULL_HANDLE };
	VkInstance instance{ VK_NULL_HANDLE };

public:
	~DebugUtilsMessenger();
	DebugUtilsMessenger() = default;
	DebugUtilsMessenger(const DebugUtilsMessenger&) = delete;
	DebugUtilsMessenger& operator=(const DebugUtilsMessenger&) = delete;
	DebugUtilsMessenger(DebugUtilsMessenger&& other) noexcept { swap(other); }
	DebugUtilsMessenger& operator=(DebugUtilsMessenger&& other) noexcept { swap(other); return *this; }
	void swap(DebugUtilsMessenger& other) noexcept {
		std::swap(debugUtilsMessenger, other.debugUtilsMessenger);
		std::swap(instance, other.instance);
	}

	DebugUtilsMessenger(const VkInstance& createInfo);
	operator const VkDebugUtilsMessengerEXT& () const;
};

class Surface {
private:
	VkSurfaceKHR surface{ VK_NULL_HANDLE };
	VkInstance instance{ VK_NULL_HANDLE };

public:
	~Surface();
	Surface() = default;
	Surface(const Surface&) = delete;
	Surface& operator=(const Surface&) = delete;
	Surface(Surface&& other) { swap(other); }
	Surface& operator=(Surface&& other) { swap(other); return *this; }
	void swap(Surface& other) {
		std::swap(surface, other.surface);
		std::swap(instance, other.instance);
	}

	Surface(const VkInstance& instance, Window* window);
	operator const VkSurfaceKHR& () const;
};

class Semaphore {
private:
	VKDEFAULT_INIT_DESCRIPTOR(Semaphore, VkSemaphore)

public:
	Semaphore(const VkDevice& device);
};

using Semaphores = std::vector<Semaphore>;
using VkSemaphores = std::vector<VkSemaphore>;

class Fence {
	VKDEFAULT_INIT_DESCRIPTOR(Fence, VkFence)

public:
	Fence(const VkDevice& device);
};

using Fences = std::vector<Fence>;

class Sampler {
	VKDEFAULT_INIT_DESCRIPTOR(Sampler, VkSampler)

public:
	Sampler(const VkDevice& device, const VkSamplerCreateInfo& samplerInfo);
};

using DescriptorSets = std::vector<VkDescriptorSet>;

class DescriptorPool {
	VKDEFAULT_INIT_DESCRIPTOR(DescriptorPool, VkDescriptorPool)

public:
	DescriptorPool(const VkDevice& device, const std::vector<const vkDefault::DescriptorSetLayout*>& descriptorSetLayouts, const uint32_t descriptorsCount);
	DescriptorPool(const VkDevice& device, const VkDescriptorPoolCreateInfo& poolInfo);
	DescriptorSets allocateDescriptorSets(const vkDefault::DescriptorSetLayout& descriptorSetLayout, const uint32_t& descriptorSetCount);
	VkDescriptorSet allocateDescriptorSet(const vkDefault::DescriptorSetLayout& descriptorSetLayout);
};

class ImageView {
	VKDEFAULT_INIT_DESCRIPTOR(ImageView, VkImageView)

public:
	ImageView(
		const VkDevice& device,
		const VkImage& image,
		VkImageViewType type,
		VkFormat format,
		VkImageAspectFlags aspectFlags,
		uint32_t mipLevels,
		uint32_t baseArrayLayer,
		uint32_t layerCount);
};

class Image {
private:
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	VKDEFAULT_INIT_DESCRIPTOR(Image, VkImage)

public:
	Image(
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
		VkMemoryPropertyFlags           properties);

	operator const VkDeviceMemory& () const;
	operator const VkDeviceMemory* () const;
};

class Buffer {
private:
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	bool updateFlag{ true };
	void* memorymap{ nullptr };
	size_t memorysize{ 0 };

	VKDEFAULT_INIT_DESCRIPTOR(Buffer, VkBuffer);

public:
	Buffer(
		VkPhysicalDevice                physicalDevice,
		VkDevice                        device,
		VkDeviceSize                    size,
		VkBufferUsageFlags              usage,
		VkMemoryPropertyFlags           properties);

	void copy(const void* data);
	void copy(const void* data, uint32_t offset, uint32_t size);
	size_t size() const;
	void* &map();
	void raiseFlag();
	bool dropFlag();

	VkDescriptorBufferInfo descriptorBufferInfo() const;

	operator const VkDeviceMemory& () const;
	operator const VkDeviceMemory* () const;
};

class SwapchainKHR {
private:
	utils::vkDefault::ImageInfo imageInfo;
	VkSwapchainKHR descriptor{ VK_NULL_HANDLE };
	VkDevice device{ VK_NULL_HANDLE };

public:
	~SwapchainKHR();
	SwapchainKHR() noexcept = default;
	SwapchainKHR(const SwapchainKHR&) = delete;
	SwapchainKHR& operator=(const SwapchainKHR&) = delete;
	SwapchainKHR(SwapchainKHR&&) = delete;
	SwapchainKHR& operator=(SwapchainKHR&&) = delete;

	operator const VkSwapchainKHR& () const;
	operator const VkSwapchainKHR* () const;
	operator const bool() const;

	VkResult reset(
		const VkDevice& device,
		const utils::vkDefault::ImageInfo& imageInfo,
		const utils::swapChain::SupportDetails& supportDetails,
		const std::vector<uint32_t>& queueFamilyIndices,
		VkSurfaceKHR surface,
		VkSurfaceFormatKHR surfaceFormat);

	std::vector<VkImage> images() const;
};

class CommandBuffer {
private:
	VkCommandPool commandPool{ VK_NULL_HANDLE };
	bool updateFlag{true};
	VKDEFAULT_INIT_DESCRIPTOR(CommandBuffer, VkCommandBuffer)

public:
	CommandBuffer(const VkDevice& device, VkCommandPool commandPool);
	VkResult reset() const;
	VkResult begin() const;
	VkResult end() const;
	void raiseFlag();
	bool dropFlag();
};

using CommandBuffers = std::vector<CommandBuffer>;

class CommandPool {
private:
	std::list<VkCommandBuffer> commandBuffers;
	VKDEFAULT_INIT_DESCRIPTOR(CommandPool, VkCommandPool)

public:
	CommandPool(const VkDevice& device);
	CommandBuffers allocateCommandBuffers(const uint32_t& commandBuffersCount) const;
};

}
#endif // MOON_UTILS_VKDEFAULT_H
