#ifndef CUDA_TEXTURE_H
#define CUDA_TEXTURE_H

#include <filesystem>
#include "utils/buffer.h"

namespace cuda::rayTracing {

struct Texture
{
    std::filesystem::path path;
    Buffer<uint8_t> buffer;
    cudaTextureObject_t object{0};

    Texture() = default;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) = default;
    Texture& operator=(Texture&&) = default;
    Texture(const std::filesystem::path& path);

    void create(const std::filesystem::path& texturePath = std::filesystem::path());
};

}
#endif // CUDA_TEXTURE_H
