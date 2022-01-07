#ifndef QUAT2_H
#define QUAT2_H

#include "quat.h"

namespace cuda::rayTracing {

template<typename T>
class quat2
{
private:
    quat<T> p;
    quat<T> q;

public:
    __host__ __device__ quat2() {}
    __host__ __device__ quat2(const quat<T>& p, const quat<T>& q) : p(p), q(q) {}
    __host__ __device__ ~quat2() {}

    __host__ __device__ quat<T> rotation() const {return p;}
    __host__ __device__ quat<T> translation() const {
        quat<T> copy(p);
        return T(2) * q * copy.conjugate();
    }

    __host__ __device__ quat<T> q1() const {return p;}
    __host__ __device__ quat<T> q2() const {return q;}

    __host__ __device__ bool operator==(const quat2& other) const {return p == other.p && q == other.q;}
    __host__ __device__ bool operator!=(const quat2& other) const {return !(*this == other);}

    __host__ __device__ quat2& operator+=(const quat2& other) {
        p += other.p;
        q += other.q;
        return *this;
    }

    __host__ __device__ quat2& operator-=(const quat2& other) {
        p -= other.p;
        q -= other.q;
        return *this;
    }

    __host__ __device__ quat2& operator*=(const quat2& other) {
        const auto pcopy = p;
        p *= other.p;
        q = pcopy * other.q + q * other.p;
        return *this;
    }

    __host__ __device__ quat2& operator*=(const T& c) {
        p *= c;
        q *= c;
        return *this;
    }

    __host__ __device__ quat2 operator+(const quat2& other) const {
        quat2 copy(*this);
        return copy += other;
    }

    __host__ __device__ quat2 operator-(const quat2& other) const {
        quat2 copy(*this);
        return copy -= other;
    }

    __host__ __device__ quat2 operator*(const quat2& other) const {
        quat2 copy(*this);
        return copy *= other;
    }

    T norm2() const {
        return p.norm2();
    }

    T anticommutator () const {
        return p.s() * q.s() + p.x() * q.x() + p.y() * q.y() + p.z() * q.z();
    }

    __host__ __device__ quat2& normalize()
    {
        T norma = T(1) / std::sqrt(p.norm2());
        return *this *= norma;
    }

    __host__ __device__ quat2& conjugate() {
        p.conjugate();
        q.conjugate() *= T(-1);
        return *this;
    }

    __host__ __device__ quat2& invert() {
        p.invert();
        q = p * q * p;
        return *this;
    }
};

template<typename T>
std::ostream& operator<< (std::ostream& out, const quat2<T>& quat) {
    out << quat.q1() << "\t\t" << quat.q2();
    return out;
}

template<typename T>
__host__ __device__ quat2<T> convert(const quat<T>& rotation, const quat<T>& translation){
    return quat2<T>(rotation, T(0.5) * translation * rotation);
}

template<typename T>
__host__ __device__ quat2<T> operator*(const T& c, const quat2<T>& quat) {
    quat2<T> copy(quat);
    return copy *= c;
}

template<typename T>
__host__ __device__ quat2<T> operator*(const quat2<T>& quat, const T& c) {
    return c * quat;
}

template<typename T>
__host__ __device__ quat2<T> normalize(const quat2<T>& quat) {
    quat2<T> copy(quat);
    return copy.normalize();
}

template<typename T>
__host__ __device__ quat2<T> conjugate(const quat2<T>& quat) {
    quat2<T> copy(quat);
    return copy.conjugate();
}

template<typename T>
__host__ __device__ quat2<T> invert(const quat2<T>& quat) {
    quat2<T> copy(quat);
    return copy.invert();
}

template<typename T>
mat4<T> toMat(const quat2<T>& q){
    quat<T> translation = q.translation();
    mat4<T> R = toMat(q.rotation());

    mat4<T> SE3 = mat4<T>::identity();
    SE3[0][0] = R[0][0];    SE3[0][1] = R[0][1];    SE3[0][2] = R[0][2];    SE3[0][3] = translation.vector().x;
    SE3[1][0] = R[1][0];    SE3[1][1] = R[1][1];    SE3[1][2] = R[1][2];    SE3[1][3] = translation.vector().y;
    SE3[2][0] = R[2][0];    SE3[2][1] = R[2][1];    SE3[2][2] = R[2][2];    SE3[2][3] = translation.vector().z;
    return SE3;
}

template<typename T>
quat2<T> toQuat2(const mat4<T>& SE3){
    mat4<T> R = mat4<T>::identity();
    R[0][0] = SE3[0][0];    R[0][1] = SE3[0][1];    R[0][2] = SE3[0][2];
    R[1][0] = SE3[1][0];    R[1][1] = SE3[1][1];    R[1][2] = SE3[1][2];
    R[2][0] = SE3[2][0];    R[2][1] = SE3[2][1];    R[2][2] = SE3[2][2];
    quat<T> rotatrion = toQuat(R);
    quat<T> translation = quat<T>(T(0), SE3[0][3], SE3[1][3], SE3[2][3]);
    return convert(rotatrion, translation);
}

using quat2f = quat2<float>;
}
#endif // QUAT2_H
