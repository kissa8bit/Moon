#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>

#undef max

namespace moon::math {

template<typename type, uint32_t n>
class BaseVector
{
protected:
    BaseVector<type, n - 1> vec;
    type s{type(0)};

public:
    BaseVector() = default;
    BaseVector(const BaseVector<type, n - 1>& other);
    BaseVector(const BaseVector<type, n>& other);
    BaseVector(const BaseVector<type, n - 1>& other, const type& s);
    BaseVector<type, n>& operator=(const BaseVector<type, n>& other);

    type& operator[](uint32_t i);
    const type& operator[](uint32_t i) const;
    uint32_t size() const;
    const BaseVector<type, n - 1>& dvec() const;

    bool operator==(const BaseVector<type, n>& other) const;
    bool operator!=(const BaseVector<type, n>& other) const;

    BaseVector<type,n> operator+(const BaseVector<type, n>& other) const;
    BaseVector<type,n> operator-(const BaseVector<type, n>& other) const;
    BaseVector<type,n> operator*(const BaseVector<type, n>& other) const;
    BaseVector<type,n> operator/(const type& c) const;

    BaseVector<type,n>& operator+=(const BaseVector<type, n>& other);
    BaseVector<type,n>& operator-=(const BaseVector<type, n>& other);
    BaseVector<type,n>& operator*=(const BaseVector<type, n>& other);

    BaseVector<type,n>& operator+=(const type& c);
    BaseVector<type,n>& operator-=(const type& c);
    BaseVector<type,n>& operator*=(const type& c);
    BaseVector<type,n>& operator/=(const type& c);

    template<typename T, uint32_t N> friend BaseVector<T,N> operator+(const T& c, const BaseVector<T,N>& other);
    template<typename T, uint32_t N> friend BaseVector<T,N> operator-(const T& c, const BaseVector<T,N>& other);
    template<typename T, uint32_t N> friend BaseVector<T,N> operator*(const T& c, const BaseVector<T,N>& other);

    template<typename T, uint32_t N> friend BaseVector<T,N> operator+(const BaseVector<T,N>& other, const T& c);
    template<typename T, uint32_t N> friend BaseVector<T,N> operator-(const BaseVector<T,N>& other, const T& c);
    template<typename T, uint32_t N> friend BaseVector<T,N> operator*(const BaseVector<T,N>& other, const T& c);

    BaseVector<type,n>& normalize();
    template<typename T, uint32_t N> friend BaseVector<T,N> normalize(const BaseVector<T,N>& other);

    template<typename T, uint32_t N> friend T dot(const BaseVector<T,N>& left, const BaseVector<T,N>& right);

    template<typename T, uint32_t N> friend T maximum(const BaseVector<T,N>& other);

    template<typename T, uint32_t N> friend T minimum(const BaseVector<T, N>& other);

    template<typename T, uint32_t N> friend BaseVector<T, N> maximum(const BaseVector<T, N>& left, const BaseVector<T, N>& right);

    template<typename T, uint32_t N> friend BaseVector<T, N> minimum(const BaseVector<T, N>& left, const BaseVector<T, N>& right);

    template<typename T, uint32_t N> friend T maxAbs(const BaseVector<T,N>& other);

    template<typename T, uint32_t N> friend BaseVector<T,N> maxAbs(const BaseVector<T,N>& left, const BaseVector<T,N>& right);

    template<typename T, uint32_t N> friend std::ostream& operator<<(std::ostream& out, const BaseVector<T,N>& other);
};

template<typename type, uint32_t n>
BaseVector<type,n>::BaseVector(const BaseVector<type, n - 1>& other)
    : vec(other) {}

template<typename type, uint32_t n>
BaseVector<type,n>::BaseVector(const BaseVector<type, n>& other)
    : vec(other.vec), s(other.s) {}

template<typename type, uint32_t n>
BaseVector<type,n>::BaseVector(const BaseVector<type, n - 1>& other, const type& s)
    : vec(other), s(s) {}

template<typename type, uint32_t n>
BaseVector<type, n>& BaseVector<type,n>::operator=(const BaseVector<type,n>& other){
    vec = other.vec;
    s = other.s;
    return *this;
}

template<typename type, uint32_t n>
type& BaseVector<type,n>::operator[](uint32_t i) {
    return i >= n - 1 ? s : vec[i];
}

template<typename type, uint32_t n>
const type& BaseVector<type,n>::operator[](uint32_t i) const{
    return i >= n - 1 ? s : vec[i];
}

template<typename type, uint32_t n>
uint32_t BaseVector<type,n>::size() const {
    return vec.size() + 1;
}

template<typename type, uint32_t n>
const BaseVector<type, n - 1>& BaseVector<type,n>::dvec() const {
    return vec;
}

template<typename type, uint32_t n>
bool BaseVector<type,n>::operator==(const BaseVector<type,n>& other) const {
    return vec == other.vec && s == other.s;
}

template<typename type, uint32_t n>
bool BaseVector<type,n>::operator!=(const BaseVector<type,n>& other) const {
    return !(*this == other);
}

template<typename type, uint32_t n>
BaseVector<type,n> BaseVector<type,n>::operator+(const BaseVector<type,n>& other) const {
    return BaseVector<type,n>(vec + other.vec, s + other.s);
}

template<typename type, uint32_t n>
BaseVector<type,n> BaseVector<type,n>::operator-(const BaseVector<type,n>& other) const {
    return BaseVector<type,n>(vec - other.vec, s - other.s);
}

template<typename type, uint32_t n>
BaseVector<type,n> BaseVector<type,n>::operator*(const BaseVector<type,n>& other) const {
    return BaseVector<type,n>(vec * other.vec, s * other.s);
}

template<typename type, uint32_t n>
BaseVector<type,n> BaseVector<type,n>::operator/(const type& c) const {
    return BaseVector<type,n>(vec / c, s / c);
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::operator+=(const BaseVector<type,n>& other) {
    vec += other.vec; s += other.s;
    return *this;
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::operator-=(const BaseVector<type,n>& other) {
    vec -= other.vec; s -= other.s;
    return *this;
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::operator*=(const BaseVector<type,n>& other) {
    vec *= other.vec; s *= other.s;
    return *this;
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::operator+=(const type& c) {
    vec += c; s += c;
    return *this;
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::operator-=(const type& c) {
    vec -= c; s -= c;
    return *this;
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::operator*=(const type& c) {
    vec *= c; s *= c;
    return *this;
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::operator/=(const type& c) {
    vec /= c; s /= c;
    return *this;
}

template<typename T, uint32_t N> BaseVector<T,N> operator+(const T& c, const BaseVector<T, N>& other) {
    return BaseVector<T,N>(c + other.vec, c + other.s);
}

template<typename T, uint32_t N> BaseVector<T,N> operator-(const T& c, const BaseVector<T, N>& other) {
    return BaseVector<T,N>(c - other.vec, c - other.s);
}

template<typename T, uint32_t N> BaseVector<T,N> operator*(const T& c, const BaseVector<T, N>& other) {
    return BaseVector<T,N>(c * other.vec, c * other.s);
}

template<typename T, uint32_t N> BaseVector<T,N> operator+(const BaseVector<T, N>& other, const T& c) {
    return BaseVector<T,N>(other.vec + c, other.s + c);
}

template<typename T, uint32_t N> BaseVector<T,N> operator-(const BaseVector<T, N>& other, const T& c) {
    return BaseVector<T,N>(other.vec - c, other.s - c);
}

template<typename T, uint32_t N> BaseVector<T,N> operator*(const BaseVector<T, N>& other, const T& c) {
    return BaseVector<T,N>(other.vec * c, other.s * c);
}

template<typename T, uint32_t N> T dot(const BaseVector<T, N>& left, const BaseVector<T, N>& right) {
    return dot(left.vec, right.vec) + left.s * right.s;
}

template<typename T, uint32_t N> T maximum(const BaseVector<T,N>& other) {
    return std::max(maximum(other.vec),other.s);
}

template<typename T, uint32_t N> T minimum(const BaseVector<T, N>& other) {
    return std::min(minimum(other.vec), other.s);
}

template<typename T, uint32_t N> BaseVector<T,N> maximum(const BaseVector<T,N>& left, const BaseVector<T,N>& right) {
    return BaseVector<T, N>(maximum(left.vec, right.vec), std::max(left.s, right.s));
}

template<typename T, uint32_t N> BaseVector<T, N> minimum(const BaseVector<T, N>& left, const BaseVector<T, N>& right) {
    return BaseVector<T, N>(minimum(left.vec, right.vec), std::min(left.s, right.s));
}

template<typename T, uint32_t N> T maxAbs(const BaseVector<T,N>& other) {
    return std::max(maxAbs(other.vec),std::abs(other.s));
}

template<typename T, uint32_t N> BaseVector<T,N> maxAbs(const BaseVector<T,N>& left, const BaseVector<T,N>& right) {
    return BaseVector<T,N>(maxAbs(left.vec, right.vec),std::max(std::abs(left.s),std::abs(right.s)));
}

template<typename type, uint32_t n>
BaseVector<type,n>& BaseVector<type,n>::normalize(){
    type norma = type(1) / std::sqrt(dot(*this, *this));
    return *this *= norma;
}

template<typename T, uint32_t N> BaseVector<T,N> normalize(const BaseVector<T, N>& other) {
    T norma = T(1) / std::sqrt(dot(other, other));
    return other * norma;
}

template<typename T, uint32_t N> std::ostream& operator<<(std::ostream& out, const BaseVector<T, N>& other){
    out << other.vec << '\t' << other.s;
    return out;
}

template<typename type>
class BaseVector<type, 2>
{
protected:
    type x0{type(0)}, x1{type(0)};

public:
    BaseVector() = default;
    BaseVector(const BaseVector<type, 2>& other);
    BaseVector<type, 2>& operator=(const BaseVector<type, 2>& other);
    BaseVector(const type& x0, const type& x1);
    BaseVector(const type& x);

    type& operator[](uint32_t i);
    const type& operator[](uint32_t i) const;
    uint32_t size() const;

    bool operator==(const BaseVector<type, 2>& other) const;
    bool operator!=(const BaseVector<type, 2>& other) const;

    BaseVector<type,2> operator+(const BaseVector<type, 2>& other) const;
    BaseVector<type,2> operator-(const BaseVector<type, 2>& other) const;
    BaseVector<type,2> operator*(const BaseVector<type, 2>& other) const;
    BaseVector<type,2> operator/(const type& c) const;

    BaseVector<type,2>& operator+=(const BaseVector<type, 2>& other);
    BaseVector<type,2>& operator-=(const BaseVector<type, 2>& other);
    BaseVector<type,2>& operator*=(const BaseVector<type, 2>& other);

    BaseVector<type,2>& operator+=(const type& c);
    BaseVector<type,2>& operator-=(const type& c);
    BaseVector<type,2>& operator*=(const type& c);
    BaseVector<type,2>& operator/=(const type& c);

    template<typename T> friend BaseVector<T,2> operator+(const T& c, const BaseVector<T, 2>& other);
    template<typename T> friend BaseVector<T,2> operator-(const T& c, const BaseVector<T, 2>& other);
    template<typename T> friend BaseVector<T,2> operator*(const T& c, const BaseVector<T, 2>& other);

    template<typename T> friend BaseVector<T,2> operator+(const BaseVector<T, 2>& other, const T& c);
    template<typename T> friend BaseVector<T,2> operator-(const BaseVector<T, 2>& other, const T& c);
    template<typename T> friend BaseVector<T,2> operator*(const BaseVector<T, 2>& other, const T& c);

    BaseVector<type,2>& normalize();
    template<typename T> friend BaseVector<T,2> normalize(const BaseVector<T, 2>& other);

    template<typename T> friend T dot(const BaseVector<T, 2>& left, const BaseVector<T, 2>& right);

    template<typename T> friend T maximum(const BaseVector<T,2>& other);

    template<typename T> friend T minimum(const BaseVector<T, 2>& other);

    template<typename T> friend BaseVector<T, 2> maximum(const BaseVector<T, 2>& left, const BaseVector<T, 2>& right);

    template<typename T> friend BaseVector<T, 2> minimum(const BaseVector<T, 2>& left, const BaseVector<T, 2>& right);

    template<typename T> friend T maxAbs(const BaseVector<T,2>& other);

    template<typename T> friend BaseVector<T,2> maxAbs(const BaseVector<T,2>& left, const BaseVector<T,2>& right);

    template<typename T> friend std::ostream& operator<<(std::ostream& out, const BaseVector<T, 2>& other);
};

template<typename type>
BaseVector<type,2>::BaseVector(const BaseVector<type, 2>& other)
    : x0(other.x0), x1(other.x1) {}

template<typename type>
BaseVector<type, 2>& BaseVector<type,2>::operator=(const BaseVector<type, 2>& other){
    x0 = other.x0;
    x1 = other.x1;
    return *this;
}

template<typename type>
BaseVector<type,2>::BaseVector(const type& x0, const type& x1)
    : x0(x0), x1(x1) {}

template<typename type>
BaseVector<type,2>::BaseVector(const type& x)
    : x0(x), x1(x) {}

template<typename type>
type& BaseVector<type,2>::operator[](uint32_t i) {
    return i >= 1 ? x1 : x0;
}

template<typename type>
const type& BaseVector<type,2>::operator[](uint32_t i) const {
    return i >= 1 ? x1 : x0;
}

template<typename type>
uint32_t BaseVector<type,2>::size() const {
    return 2;
}

template<typename type>
bool BaseVector<type,2>::operator==(const BaseVector<type, 2>& other) const {
    return x0 == other.x0 && x1 == other.x1;
}

template<typename type>
bool BaseVector<type,2>::operator!=(const BaseVector<type,2>& other) const {
    return !(*this == other);
}

template<typename type>
BaseVector<type,2> BaseVector<type,2>::operator+(const BaseVector<type,2>& other) const {
    return BaseVector<type,2>(x0 + other.x0, x1 + other.x1);
}

template<typename type>
BaseVector<type,2> BaseVector<type,2>::operator-(const BaseVector<type,2>& other) const {
    return BaseVector<type,2>(x0 - other.x0, x1 - other.x1);
}

template<typename type>
BaseVector<type,2> BaseVector<type,2>::operator*(const BaseVector<type,2>& other) const {
    return BaseVector<type,2>(x0 * other.x0, x1 * other.x1);
}

template<typename type>
BaseVector<type,2> BaseVector<type,2>::operator/(const type& c) const {
    return BaseVector<type,2>(x0 / c, x1 / c);
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::operator+=(const BaseVector<type,2>& other) {
    x0 += other.x0; x1 += other.x1;
    return *this;
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::operator-=(const BaseVector<type,2>& other) {
    x0 -= other.x0; x1 -= other.x1;
    return *this;
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::operator*=(const BaseVector<type,2>& other) {
    x0 *= other.x0; x1 *= other.x1;
    return *this;
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::operator+=(const type& c) {
    x0 += c; x1 += c;
    return *this;
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::operator-=(const type& c) {
    x0 -= c; x1 -= c;
    return *this;
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::operator*=(const type& c) {
    x0 *= c; x1 *= c;
    return *this;
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::operator/=(const type& c) {
    x0 /= c; x1 /= c;
    return *this;
}

template<typename T> BaseVector<T,2> operator+(const T& c, const BaseVector<T,2>& other) {
    return BaseVector<T,2>(c + other.x0, c + other.x1);
}

template<typename T> BaseVector<T,2> operator-(const T& c, const BaseVector<T,2>& other) {
    return BaseVector<T,2>(c - other.x0, c - other.x1);
}

template<typename T> BaseVector<T,2> operator*(const T& c, const BaseVector<T,2>& other) {
    return BaseVector<T,2>(c * other.x0, c * other.x1);
}

template<typename T> BaseVector<T,2> operator+(const BaseVector<T,2>& other, const T& c) {
    return BaseVector<T,2>(other.x0 + c, other.x1 + c);
}

template<typename T> BaseVector<T,2> operator-(const BaseVector<T,2>& other, const T& c) {
    return BaseVector<T,2>(other.x0 - c, other.x1 - c);
}

template<typename T> BaseVector<T,2> operator*(const BaseVector<T,2>& other, const T& c) {
    return BaseVector<T,2>(other.x0 * c, other.x1 * c);
}

template<typename type>
BaseVector<type,2>& BaseVector<type,2>::normalize(){
    type n = type(1) / std::sqrt(x0 * x0 + x1 * x1);
    return *this *= n;
}

template<typename T> BaseVector<T,2> normalize(const BaseVector<T,2>& other) {
    T n = T(1) / std::sqrt(other.x0 * other.x0 + other.x1 * other.x1);
    return other * n;
}

template<typename T> T dot(const BaseVector<T,2>& left, const BaseVector<T,2>& right){
    return left.x0 * right.x0 + left.x1 * right.x1;
}

template<typename T> T maximum(const BaseVector<T,2>& other) {
    return std::max(other.x0, other.x1);
}

template<typename T> T minimum(const BaseVector<T, 2>& other) {
    return std::min(other.x0, other.x1);
}

template<typename T> BaseVector<T,2> maximum(const BaseVector<T,2>& left, const BaseVector<T,2>& right) {
    return BaseVector<T, 2>(std::max(left.x0, right.x0), std::max(left.x1, right.x1));
}

template<typename T> BaseVector<T, 2> minimum(const BaseVector<T, 2>& left, const BaseVector<T, 2>& right) {
    return BaseVector<T, 2>(std::min(left.x0, right.x0), std::min(left.x1, right.x1));
}

template<typename T> T maxAbs(const BaseVector<T,2>& other) {
    return std::max(std::abs(other.x0), std::abs(other.x1));
}

template<typename T> BaseVector<T,2> maxAbs(const BaseVector<T,2>& left, const BaseVector<T,2>& right) {
    return BaseVector<T,2>(std::max(std::abs(left.x0), std::abs(right.x0)), std::max(std::abs(left.x1), std::abs(right.x1)));
}

template<typename T> std::ostream& operator<<(std::ostream& out, const BaseVector<T,2>& other) {
    out << other.x0 << '\t' << other.x1;
    return out;
}

template<typename type, uint32_t n> class Vector;

template<typename type>
class Vector<type, 2> : public BaseVector<type, 2>
{
public:
    Vector() : BaseVector<type, 2>() {}
    Vector(const BaseVector<type, 2>& other) : BaseVector<type, 2>(other) {}
    Vector(const Vector<type, 2>& other) : BaseVector<type, 2>(other.x0, other.x1) {}
    Vector(const type& x0, const type& x1) {
        this->x0 = x0;
        this->x1 = x1;
    }
    Vector(const type& x) : BaseVector<type, 2>(x) {}
    Vector<type, 2>& operator=(const Vector<type, 2> other) {
        this->x0 = other.x0;
        this->x1 = other.x1;
        return *this;
    }
};

template<typename type>
class Vector<type, 3> : public BaseVector<type, 3>
{
public:
    Vector() : BaseVector<type, 3>() {}
    Vector(const BaseVector<type, 2>& other, const type& s) : BaseVector<type, 3>(other, s) {}
    Vector(const BaseVector<type, 3>& other) : BaseVector<type, 3>(other) {}
    Vector(const Vector<type, 3>& other) : BaseVector<type, 3>(other.vec, other.s) {}
    Vector(const type& x0, const type& x1, const type& s) {
        this->vec = BaseVector<type, 2>(x0, x1);
        this->s = s;
    }
    Vector(const type& x) {
        this->vec[0] = x;
        this->vec[1] = x;
        this->s = x;
    }
    Vector<type,3>& operator=(const Vector<type,3> other) {
        this->vec = other.vec;
        this->s = other.s;
        return *this;
    }

    template<typename T> friend Vector<T,3> cross(const Vector<T, 3>& left, const Vector<T, 3>& right);
};

template<typename type>
class Vector<type, 4> : public BaseVector<type, 4>
{
public:
    Vector() : BaseVector<type,4>() {}
    Vector(const BaseVector<type, 3>& other, const type& s) : BaseVector<type, 4>(other, s) {}
    Vector(const BaseVector<type, 4>& other) : BaseVector<type, 4>(other) {}
    Vector(const Vector<type, 4>& other) : BaseVector<type,4>(other.vec, other.s) {}
    Vector(const type& x0, const type& x1, const type& x2, const type& s) {
        this->vec = BaseVector<type, 3>({x0, x1}, x2);
        this->s = s;
    }
    Vector(const type& x) {
        this->vec[0] = x;
        this->vec[1] = x;
        this->vec[2] = x;
        this->s = x;
    }
    Vector<type,4>& operator=(const Vector<type,4> other) {
        this->vec = other.vec;
        this->s = other.s;
        return *this;
    }
};

template<typename T> Vector<T,3> cross(const Vector<T, 3>& left, const Vector<T, 3>& right){
    return Vector<T,3>(
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]
    );
}

template<typename T> Vector<T,3> cross(const BaseVector<T, 3>& left, const BaseVector<T, 3>& right){
    return Vector<T,3>(
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]
    );
}

template<typename T, uint32_t N> Vector<T,N> mix(const Vector<T,N>& left, const Vector<T,N>& right, T s){
    return left + T(s) * (right - left);
}

extern template class Vector<float, 2>;
extern template class Vector<double, 2>;

extern template class Vector<float, 3>;
extern template class Vector<double, 3>;

extern template class Vector<float, 4>;
extern template class Vector<double, 4>;

}
#endif // VECTOR_H
