#include "operations.h"

#include <fstream>
#include <iostream>

#include "math/vec4.h"

namespace cuda::rayTracing {

void debug::check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
    }
}

namespace cuda::Image {
    void outPPM(void* frameBuffer, size_t width, size_t height, const std::string& filename) {
        vec4f* hostFrameBuffer = new vec4f[width * height];
        cudaMemcpy(hostFrameBuffer, frameBuffer, width * height * sizeof(vec4f), cudaMemcpyDeviceToHost);

        std::ofstream image(filename);
        image << "P3\n" << width << " " << height << "\n255\n";
        for (size_t j = 0; j < height; j++) {
            for (size_t i = 0; i < width; i++) {
                size_t pixel_index = j * width + (width - 1 - i);
                image   << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].r()) << " "
                        << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].g()) << " "
                        << static_cast<uint32_t>(255.99f * hostFrameBuffer[pixel_index].b()) << "\n";
            }
        }
        delete[] hostFrameBuffer;
    }

    void outPGM(void* frameBuffer, size_t width, size_t height, const std::string& filename) {
        vec4f* hostFrameBuffer = new vec4f[width * height];
        cudaMemcpy(hostFrameBuffer, frameBuffer, width * height * sizeof(vec4f), cudaMemcpyDeviceToHost);

        std::ofstream image(filename);
        image << "P2\n" << width << " " << height << "\n255\n";
        for (size_t j = 0; j < height; j++) {
            for (size_t i = 0; i < width; i++) {
                size_t pixel_index = j * width + (width - 1 - i);
                image << static_cast<uint32_t>(255.99f * (hostFrameBuffer[pixel_index].r() + hostFrameBuffer[pixel_index].g() + hostFrameBuffer[pixel_index].b()) / 3) << "\n";
            }
        }
        delete[] hostFrameBuffer;
    }
}

    }
