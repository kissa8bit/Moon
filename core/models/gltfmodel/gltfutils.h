#ifndef GLTFMODEL_UTILS_H
#define GLTFMODEL_UTILS_H

#include <vector>

#include "vector.h"
#include "matrix.h"
#include "quaternion.h"

namespace moon::models {

template <typename T1, size_t n, typename T2>
void convert(math::Vector<T1, n>& dst,const std::vector<T2>& src) {
    if (src.size() == n) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<T1>(src[i]);
        }
    }
}

template <typename T>
math::Vector<float, 3> toVector3f(const std::vector<T>& src) {
    math::Vector<float, 3> dst;
    convert(dst, src);
    return dst;
}

template <typename T1, typename T2>
void convert(math::Quaternion<T1>& dst, const std::vector<T2>& src) {
    if (src.size() == 4) {
        dst = moon::math::Quaternion<T1>(
            static_cast<T1>(src[3]), static_cast<T1>(src[0]), static_cast<T1>(src[1]), static_cast<T1>(src[2])
        );
    }
}

template <typename T1, size_t n, size_t m, typename T2>
void convert(math::Matrix<T1, n, m>& dst, const std::vector<T2>& src) {
    if (src.size() == n * m) {
        const T2* data = src.data();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                dst[i][j] = static_cast<T1>(data[m * i + j]);
            }
        }
    }
}

inline VkSamplerAddressMode getVkWrapMode(int32_t wrapMode) {
    switch (wrapMode) {
    case 10497:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 33071:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 33648:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    };
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

inline VkFilter getVkFilterMode(int32_t filterMode) {
    switch (filterMode) {
    case 9728:
    case 9984:
    case 9985:
        return VK_FILTER_NEAREST;
    case 9729:
    case 9986:
    case 9987:
        return VK_FILTER_LINEAR;
    }
    return VK_FILTER_LINEAR;
}

}

#endif