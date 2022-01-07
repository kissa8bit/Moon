#include "texture.h"

#include <stb_image.h>

namespace cuda::rayTracing {

Texture::Texture(const std::filesystem::path& path) : path(path) {
    create(path);
}

void Texture::create(const std::filesystem::path& texturePath){
    path = texturePath.empty() ? path : texturePath;

    int texWidth = 0, texHeight = 0, texChannels = 0;
    uint8_t* datd = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    buffer = Buffer<uint8_t>(4 * texWidth * texHeight, datd);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = buffer.get();
    resDesc.res.pitch2D.width = texWidth;
    resDesc.res.pitch2D.height = texHeight;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
    resDesc.res.pitch2D.pitchInBytes = 4 * texWidth;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = true;
    texDesc.readMode = cudaTextureReadMode::cudaReadModeElementType;
    texDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
    checkCudaErrors(cudaCreateTextureObject(&object, &resDesc, &texDesc, NULL));
    checkCudaErrors(cudaGetLastError());
}

}
