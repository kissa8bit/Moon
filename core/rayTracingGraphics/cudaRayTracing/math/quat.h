#ifndef QUAT_H
#define QUAT_H

#include "vec4.h"
#include "mat4.h"

namespace cuda::rayTracing {

template<typename T>
class quat
{
private:
    T e[4];

public:
    __host__ __device__ quat() {}
    __host__ __device__ ~quat() {}
    __host__ __device__ quat(const T& s, const T& x, const T& y, const T& z) { e[0] = s; e[1] = x; e[2] = y; e[3] = z;}
    __host__ __device__ quat(const T& angle, const vec4<T>& axis) {
        float s = std::cos(angle*T(0.5));
        vec4f ax = std::sin(angle*T(0.5)) * normal(vec4f(axis[0], axis[1], axis[2], 0.0));
        *this = quat(s, ax[0], ax[1], ax[2]);
    }

    __host__ __device__ T operator[](int i) const { return e[i]; }
    __host__ __device__ T& operator[](int i) { return e[i]; };
    __host__ __device__ T s() const {return e[0];}
    __host__ __device__ T x() const {return e[1];}
    __host__ __device__ T y() const {return e[2];}
    __host__ __device__ T z() const {return e[3];}
    __host__ __device__ vec4<T> vec() const {return vec4<T>(e[1], e[2], e[3], T(0));}

    __host__ __device__ bool operator==(const quat& other) const {
        return e[0] == other.e[0] && e[1] == other.e[1] && e[2] == other.e[2] && e[3] == other.e[3];
    }

    __host__ __device__ bool operator!=(const quat& other) const {
        return !(*this == other);
    }

    __host__ __device__ quat& operator+=(const quat& other){
        e[0] += other.e[0];
        e[1] += other.e[1];
        e[2] += other.e[2];
        e[3] += other.e[3];
        return *this;
    }

    __host__ __device__ quat& operator-=(const quat& other){
        e[0] -= other.e[0];
        e[1] -= other.e[1];
        e[2] -= other.e[2];
        e[3] -= other.e[3];
        return *this;
    }

    __host__ __device__ quat operator+(const quat& other) const {
        quat copy(*this);
        return copy += other;
    }

    __host__ __device__ quat operator-(const quat& other) const {
        quat copy(*this);
        return copy -= other;
    }

    __host__ __device__ quat& operator*=(const quat& other) {
        T s = e[0];
        e[0] = s * other.e[0] - (e[1] * other.e[1] + e[2] * other.e[2] + e[3] * other.e[3]);
        e[1] = s * other.e[1] + other.e[0] * e[1] + (e[2] * other.e[3] - e[3] * other.e[2]);
        e[2] = s * other.e[2] + other.e[0] * e[2] + (e[3] * other.e[1] - e[1] * other.e[3]);
        e[3] = s * other.e[3] + other.e[0] * e[3] + (e[1] * other.e[2] - e[2] * other.e[1]);
        return *this;
    }

    __host__ __device__ quat& operator*=(const T& c) {
        e[0] *= c;
        e[1] *= c;
        e[2] *= c;
        e[3] *= c;
        return *this;
    }

    __host__ __device__ quat operator*(const quat& other) const {
        quat copy(*this);
        return copy *= other;
    }

    __host__ __device__ inline T norm2() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3];
    }

    __host__ __device__ quat& normalize() {
        T norm = T(1) / std::sqrt(norm2());
        return *this *= norm;
    }

    __host__ __device__ quat& conjugate() {
        e[1] = -e[1];
        e[2] = -e[2];
        e[3] = -e[3];
        return *this;
    }

    __host__ __device__ quat& invert() {
        T norm = T(1) / norm2();
        return this->conjugate() *= norm;
    }
};

template<typename T>
std::ostream& operator<< (std::ostream& out, const quat<T>& quat) {
    out << quat[0] << '\t' << quat[1] << '\t' << quat[2] << '\t' << quat[3];
    return out;
}

template<typename T>
__host__ __device__ quat<T> operator*(const T& c, const quat<T>& q) {
    quat<T> copy(q);
    return copy *= c;
}

template<typename T>
__host__ __device__ quat<T> operator*(const quat<T>& quat, const T& c) {
    return c * quat;
}

template<typename T>
__host__ __device__ quat<T> normalize(const quat<T>& q) {
    quat<T> copy(q);
    return copy.normalize();
}

template<typename T>
__host__ __device__ quat<T> conjugate(const quat<T>& q){
    quat<T> copy(q);
    return copy.conjugate();
}

template<typename T>
__host__ __device__ quat<T> invert(const quat<T>& q) {
    quat<T> copy(q);
    return copy.invert();
}

template<typename T>
quat<T> toQuat(const mat4<T>& m){
    T s = std::sqrt(T(1)+ m[0][0] + m[1][1] + m[2][2]) / T(2);
    return quat<T>{s,
        (m[1][0] - m[0][1]) / (T(4) * s),
        (m[0][2] - m[2][0]) / (T(4) * s),
        (m[2][1] - m[1][2]) / (T(4) * s)
    };
}

template<typename T>
mat4<T> toMat(const quat<T>& q){
    mat4<T> m = mat4<T>::identity();
    m[0][0] -= T(2) * (q[2] * q[2] + q[3] * q[3]);  m[0][1]  = T(2) * (q[1] * q[2] - q[3] * q[0]);   m[0][2]  = T(2) * (q[1] * q[3] + q[2] * q[0]);
    m[1][0]  = T(2) * (q[1] * q[2] + q[3] * q[0]);  m[1][1] -= T(2) * (q[1] * q[1] + q[3] * q[3]);   m[1][2]  = T(2) * (q[2] * q[3] - q[1] * q[0]);
    m[2][0]  = T(2) * (q[1] * q[3] - q[2] * q[0]);  m[2][1]  = T(2) * (q[2] * q[3] + q[1] * q[0]);   m[2][2] -= T(2) * (q[1] * q[1] + q[2] * q[2]);
    return m;
}

using quatf = quat<float>;
}

#endif // QUAT_H
