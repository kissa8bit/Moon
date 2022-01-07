#ifndef MAT4_H
#define MAT4_H

#include "vec4.h"

namespace cuda::rayTracing {

template<typename T>
class mat4
{
private:
    vec4<T> row[4];

public:
    __host__ __device__ mat4() {}

    __host__ __device__ mat4(const T& v) {
        for(size_t i = 0; i < 4; i++) row[i] = vec4<T>(v);
    }

    __host__ __device__ mat4(const vec4<T>& row0, const vec4<T>& row1, const vec4<T>& row2, const vec4<T>& row3){
        row[0] = row0; row[1] = row1; row[2] = row2; row[3] = row3;
    }

    __host__ __device__ const mat4<T>& operator+() const { return *this; }
    __host__ __device__ mat4<T> operator-() const { return mat4<T>(-row[0], -row[1], -row[2], -row[3]); }
    __host__ __device__ vec4<T> operator[](int i) const { return row[i]; }
    __host__ __device__ vec4<T>& operator[](int i) { return row[i]; };

    __host__ __device__ mat4<T>& operator+=(const mat4<T>& m2){
        for(size_t i = 0; i < 4; i++){
            row[i] += m2.row[i];
        }
        return *this;
    }

    __host__ __device__ mat4<T>& operator-=(const mat4<T>& m2){
        for(size_t i = 0; i < 4; i++){
            row[i] -= m2.row[i];
        }
        return *this;
    }

    __host__ __device__ mat4<T>& operator*=(const mat4<T>& m2){
        for(int i = 0; i < 4; i++){
            vec4<T> copy = row[i];
            for(int j = 0; j < 4; j++){
                row[i][j] = dot(copy, vec4<T>(m2[0][j], m2[1][j], m2[2][j], m2[3][j]));
            }
        }
        return *this;
    }

    __host__ __device__ mat4<T>& operator*=(const T t){
        for(size_t i = 0; i < 4; i++){
            row[i] *= t;
        }
        return *this;
    }

    static mat4<T> identity(){
        mat4<T> m(0);
        for(int i = 0; i < 4; i++){
            m[i][i] = T(1);
        }
        return m;
    }
};

template<typename T>
__host__ __device__ mat4<T> operator+(const mat4<T>& m1, const mat4<T>& m2){
    mat4<T> copy(m1);
    return copy += m2;
}

template<typename T>
__host__ __device__ mat4<T> operator-(const mat4<T>& m1, const mat4<T>& m2){
    mat4<T> copy(m1);
    return copy -= m2;
}

template<typename T>
__host__ __device__ mat4<T> operator*(const mat4<T>& m1, const mat4<T>& m2){
    mat4<T> copy(m1);
    return copy *= m2;
}

template<typename T>
__host__ __device__ mat4<T> operator*(const mat4<T>& m1, const T t){
    mat4<T> copy(m1);
    return copy *= t;
}

template<typename T>
__host__ __device__ mat4<T> operator*(const T t, const mat4<T>& m1){
    return m1 * t;
}

template<typename T>
__host__ __device__ vec4<T> operator*(const mat4<T>& m, const vec4<T>& v){
    vec4<T> res;
    for(size_t i = 0; i < 4; i++){
        res[i] = dot(m[i], v);
    }
    return res;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const mat4<T>& m) {
    for(size_t i = 0; i < 4; i++){
        os << m[i] << '\n';
    }
    return os;
}

template<typename T>
__host__ __device__ mat4<T> trans(vec4<T> t){
    auto m = mat4<T>::identity();
    m[0][3] = t[0];
    m[1][3] = t[1];
    m[2][3] = t[2];
    return m;
}

template<typename T>
__host__ __device__ mat4<T> scale(vec4<T> s){
    auto m = mat4<T>::identity();
    m[0][0] = s[0];
    m[1][1] = s[1];
    m[2][2] = s[2];
    return m;
}

using mat4f = mat4<float>;
}
#endif // MAT4_H
