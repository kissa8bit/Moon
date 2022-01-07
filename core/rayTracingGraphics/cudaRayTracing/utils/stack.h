#ifndef STACK_H
#define STACK_H

#include <stddef.h>

namespace cuda::rayTracing {

template <typename type, size_t container_capacity = 20>
class Stack {
private:
    size_t container_size{0};
    type static_storage[container_capacity];

public:
    __host__ __device__ Stack(){}
    __host__ __device__ ~Stack(){}
    __host__ __device__ Stack(const type& data){ push(data); }
    __host__ __device__ constexpr size_t capacity() const { return container_capacity; }
    __host__ __device__ size_t size() const { return container_size; }
    __host__ __device__ bool empty() const { return container_size == 0; }

    __host__ __device__ bool push(const type& data){
        if(container_size >= container_capacity){
            return false;
        }

        static_storage[container_size] = data;
        container_size++;
        return true;
    }

    __host__ __device__ type& top() {
        return static_storage[container_size - 1];
    }

    __host__ __device__ const type& top() const {
        return static_storage[container_size - 1];
    }

    __host__ __device__ bool pop(){
        if(container_size == 0){
            return false;
        }
        container_size--;
        return true;
    }
};

}
#endif // STACK_H
