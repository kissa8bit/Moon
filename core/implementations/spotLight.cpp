#include "spotLight.h"

namespace moon::implementations {

interfaces::LightType toLightType(SpotLight::Type type)
{
	moon::interfaces::LightType res{};
	switch (type)
	{
	case SpotLight::Type::circle:
		res = moon::interfaces::LightType::spotCircle;
		break;
	case SpotLight::Type::square:
		res = moon::interfaces::LightType::spotSquare;
		break;
	default:
		CHECK_M(false, "unknown SpotLight::Type");
		break;
	}
	return res;
}

SpotLight::SpotLight(Type type)
	: Light(interfaces::LightMask(toLightType(type))), uniformBuffer(&hostBuffer, sizeof(hostBuffer))
{}

void SpotLight::setTexture(const std::filesystem::path& path) {
	texturePath = path;
}

void SpotLight::create(const utils::PhysicalDevice& device, VkCommandPool commandPool, uint32_t imageCount) {
	uniformBuffer = utils::UniformBuffer(device, imageCount, uniformBuffer.host, uniformBuffer.size);

	VkCommandBuffer commandBuffer = utils::singleCommandBuffer::create(device.device(), commandPool);
	texture = texturePath.empty() ? utils::Texture::createEmpty(device, commandBuffer) : utils::Texture(texturePath, device, device.device(), commandBuffer);
	CHECK(utils::singleCommandBuffer::submit(device.device(), device.device()(0, 0), commandPool, &commandBuffer));
	texture.destroyCache();

	createDescriptors(device, imageCount);
}

void SpotLight::render(
	uint32_t frameNumber,
	VkCommandBuffer commandBuffer,
	const utils::vkDefault::DescriptorSets& descriptorSet,
	VkPipelineLayout pipelineLayout,
	VkPipeline pipeline)
{
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	utils::vkDefault::DescriptorSets descriptors = descriptorSet;
	descriptors.push_back(descriptorSets[frameNumber]);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, static_cast<uint32_t>(descriptors.size()), descriptors.data(), 0, nullptr);
	vkCmdDraw(commandBuffer, 18, 1, 0, 0);
}

void SpotLight::update(uint32_t frameNumber, VkCommandBuffer commandBuffer) {
	uniformBuffer.update(frameNumber, commandBuffer);
}

void SpotLight::createDescriptors(const utils::PhysicalDevice& device, uint32_t imageCount) {
	descriptorSetLayout = interfaces::Light::createDescriptorSetLayout(device.device());
	descriptorPool = utils::vkDefault::DescriptorPool(device.device(), { &descriptorSetLayout }, imageCount);
	descriptorSets = descriptorPool.allocateDescriptorSets(descriptorSetLayout, imageCount);

	for (size_t i = 0; i < imageCount; i++) {
		utils::descriptorSet::Writes writes;
		WRITE_DESCRIPTOR(writes, descriptorSets.at(i), uniformBuffer.device.at(i).descriptorBufferInfo());
		WRITE_DESCRIPTOR(writes, descriptorSets.at(i), texture.descriptorImageInfo());
		utils::descriptorSet::update(device.device(), writes);
	}
}

utils::Buffers& SpotLight::buffers() {
	return uniformBuffer.device;
}

SpotLight::Buffer& SpotLight::buffer(bool markDirty) {
	if (markDirty) {
		utils::vkDefault::raiseFlags(uniformBuffer.device);
	}
	return hostBuffer;
}

const SpotLight::Buffer& SpotLight::buffer() const {
	return hostBuffer;
}

void SpotLight::setTransformation(const math::mat4& transformation) {
	buffer(true).view = transpose(inverse(transformation));
}

} // moon::implementations
