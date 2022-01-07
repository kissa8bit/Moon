#ifndef BUFFERH
#define BUFFERH

#include "operations.h"
#include "devicep.h"

namespace cuda::rayTracing {

template <typename type>
class Buffer
{
private:
    Devicep<type> memory;
    size_t size{ 0 };

public:
    Buffer() {};
    ~Buffer() = default;
    Buffer(const size_t& size, const type* mem = nullptr)
        : memory(size), size(size) {
        if(mem){
            cudaMemcpy(memory.get(), mem, size * sizeof(type), cudaMemcpyHostToDevice);
            checkCudaErrors(cudaGetLastError());
        }
    }

    Buffer(const Buffer& other) = delete;
    Buffer& operator=(const Buffer& other) = delete;

    Buffer(Buffer&& other) = default;
    Buffer& operator=(Buffer&& other) = default;

    type* get() { return memory;}
    size_t getSize() { return size;}
    type* release() { return memory.release();}
};

}
#endif // !BUFFERH
