#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace moon::math {

#define BASE_VEC_PTR_SHIFT(i) *((type*)this + i)

#define BASE_VECTOR_CTR_GENERATOR                                                                   \
    BaseVector() = default;                                                                         \
    BaseVector(const Reduced& v) : x0(v) {}                                                         \
    BaseVector(const BaseVector& v) : x0(v.x0), x1(v.x1) {}                                         \
    BaseVector(const Reduced& v, const type& s) : x0(v), x1(s) {}                                   \
    BaseVector& operator=(const BaseVector& v) { x0 = v.x0; x1 = v.x1; return *this; }

#define BASE_VECTOR_OPERATOR_GENERATOR(op)                                                          \
    BaseVector& operator op=(const type& c) { x0 op= c; x1 op= c; return *this; }                   \
    BaseVector& operator op=(const BaseVector& v) { x0 op= v.x0; x1 op= v.x1; return *this; }       \
    BaseVector operator op(const type& c) const { return BaseVector(*this) op= c; }                 \
    BaseVector operator op(const BaseVector& v) const { return BaseVector(*this) op= v; }           \
    friend BaseVector operator op(const type& c, const BaseVector& v) { return v op c; }

#define BASE_VECTOR_NORM_GENERATOR                                                                  \
    type norm() const { return std::sqrt(this->dot(*this)); }                                       \
    BaseVector& normalize() { return *this /= norm(); }                                             \
    BaseVector normalized() const { return BaseVector(*this).normalize(); }

#define BASE_VECTOR_UNARY_OPERATOR_GENERATOR                                                        \
    BaseVector operator+() { return *this; }                                                        \
    BaseVector operator-() { return *this * type(-1); }

#define BASE_VECTOR_COMPARISON_OPERATOR_GENERATOR                                                   \
    bool operator==(const BaseVector& v) const { return x0 == v.x0 && x1 == v.x1; }                 \
    bool operator!=(const BaseVector& v) const { return !(*this == v); }

#define BASE_VECTOR_FUNCTIONS_GENERATOR                                                             \
    VEC_N_TEMP T dot(const VEC_N& l, const VEC_N& r) { return l.dot(r); }                           \
    VEC_N_TEMP T norm(const VEC_N& l) { return l.norm(); }                                          \
    VEC_N_TEMP VEC_N& normalize(VEC_N& l) { return l.normalize(); }                                 \
    VEC_N_TEMP VEC_N normalized(const VEC_N& l) { return l.normalized(); }                          \
    VEC_N_TEMP std::ostream& operator<<(std::ostream& out, const VEC_N& v) {                        \
        return out << v.dvec() << '\t' << v.last();                                                 \
    }

#define BASE_VECTOR_GENERATOR                                                                       \
    BASE_VECTOR_CTR_GENERATOR                                                                       \
    BASE_VECTOR_COMPARISON_OPERATOR_GENERATOR                                                       \
    BASE_VECTOR_UNARY_OPERATOR_GENERATOR                                                            \
    BASE_VECTOR_OPERATOR_GENERATOR(+)                                                               \
    BASE_VECTOR_OPERATOR_GENERATOR(-)                                                               \
    BASE_VECTOR_OPERATOR_GENERATOR(*)                                                               \
    BASE_VECTOR_OPERATOR_GENERATOR(/)                                                               \
    BASE_VECTOR_NORM_GENERATOR

#define VEC(N) BaseVector<T, N>

template<typename type, uint32_t n>
class BaseVector
{
public:
    using Reduced = BaseVector<type, n - 1>;

protected:
    Reduced x0;
    type x1{type(0)};

public:
    type& operator[](uint32_t i) { return BASE_VEC_PTR_SHIFT(i); }
    const type& operator[](uint32_t i) const { return BASE_VEC_PTR_SHIFT(i); }
    constexpr uint32_t size() const { return n; }
    const Reduced& dvec() const { return x0; }
    const type& last() const { return x1; }

    BASE_VECTOR_GENERATOR

    type dot(const BaseVector& v) const { return x0.dot(v.x0) + x1 * v.x1;}
    type maxValue() const { return std::max(x0.maxValue(), x1); }
    type minValue() const { return std::min(x0.minValue(), x1); }
};

#define VEC_N VEC(N)
#define VEC_N_TEMP template<typename T, uint32_t N>

BASE_VECTOR_FUNCTIONS_GENERATOR
VEC_N_TEMP VEC_N max(const VEC_N& l, const VEC_N& r) { return VEC_N(max(l.dvec(), r.dvec()), std::max(l.last(), r.last())); }
VEC_N_TEMP VEC_N min(const VEC_N& l, const VEC_N& r) { return VEC_N(min(l.dvec(), r.dvec()), std::min(l.last(), r.last())); }

#undef VEC_N
#undef VEC_N_TEMP

template<typename type>
class BaseVector<type, 2>
{
public:
    using Reduced = type;

protected:
    type x0{type(0)}, x1{type(0)};

public:
    type& operator[](uint32_t i) { return BASE_VEC_PTR_SHIFT(i); }
    const type& operator[](uint32_t i) const { return BASE_VEC_PTR_SHIFT(i); }
    constexpr uint32_t size() const { return 2; }
    const Reduced& dvec() const { return x0; }
    const type& last() const { return x1; }

    BASE_VECTOR_GENERATOR

    type dot(const BaseVector& v) const { return x0 * v.x0 + x1 * v.x1; }
    type maxValue() const { return std::max(x0, x1);}
    type minValue() const { return std::min(x0, x1);}
};

#define VEC_N VEC(2)
#define VEC_N_TEMP template<typename T>

BASE_VECTOR_FUNCTIONS_GENERATOR
VEC_N_TEMP VEC_N max(const VEC_N& l, const VEC_N& r) { return VEC_N(std::max(l.dvec(), r.dvec()), std::max(l.last(), r.last())); }
VEC_N_TEMP VEC_N min(const VEC_N& l, const VEC_N& r) { return VEC_N(std::min(l.dvec(), r.dvec()), std::min(l.last(), r.last())); }

#undef VEC_N
#undef VEC_N_TEMP

#undef BASE_VECTOR_GENERATOR
#undef BASE_VECTOR_FUNCTIONS_GENERATOR
#undef BASE_VECTOR_COMPARISON_OPERATOR_GENERATOR
#undef BASE_VECTOR_UNARY_OPERATOR_GENERATOR
#undef BASE_VECTOR_NORM_GENERATOR
#undef BASE_VECTOR_OPERATOR_GENERATOR
#undef BASE_VECTOR_CTR_GENERATOR
#undef BASE_VEC_PTR_SHIFT

template<typename type, uint32_t n> class Vector;

#define VECTOR_CTR_GENERATOR(n)                                                                         \
    Vector() = default;                                                                                 \
    Vector(const Reduced& v, const type& s) : BaseVector<type, n>(v, s) {}                              \
    Vector(const type& x) : BaseVector<type, n>(Reduced(x), x) {}                                       \
    Vector(const Vector& v) : BaseVector<type, n>(v.x0, v.x1) {}                                        \
    Vector& operator=(const Vector& v) { x0 = v.dvec(); x1 = v.last(); return *this; }                  \
    Vector(const BaseVector<type, n>& v) : BaseVector<type, n>(v) {}                                    \
    Vector& operator=(const BaseVector<type, n>& v) { x0 = v.dvec(); x1 = v.last(); return *this; }

template<typename type>
class Vector<type, 2> : public BaseVector<type, 2>
{
public:
    using Reduced = type;

    VECTOR_CTR_GENERATOR(2)
};

template<typename type>
class Vector<type, 3> : public BaseVector<type, 3>
{
public:
    using Reduced = Vector<type, 2>;

    VECTOR_CTR_GENERATOR(3)

    Vector(const type& x0, const type& x1, const type& x2) : BaseVector<type, 3>(Reduced(x0, x1), x2) {}
};

template<typename type>
class Vector<type, 4> : public BaseVector<type, 4>
{
public:
    using Reduced = Vector<type, 3>;

    VECTOR_CTR_GENERATOR(4)

    Vector(const type& x0, const type& x1, const type& x2, const type& x3) : BaseVector<type, 4>(Reduced(x0, x1, x2), x3) {}
};

#undef VECTOR_CTR_GENERATOR

template<typename T>
VEC(3) cross(const VEC(3)& l, const VEC(3)& r){
    return VEC(3)({l[1] * r[2] - l[2] * r[1], l[2] * r[0] - l[0] * r[2]}, l[0] * r[1] - l[1] * r[0]);
}

template<typename T, uint32_t N>
VEC(N) mix(const VEC(N)& l, const VEC(N)& r, const T& t){
    return l + t * (r - l);
}

#undef VEC

#define VEC_EXT_TMEP(n) extern template class Vector<float, n>; extern template class Vector<double, n>;

VEC_EXT_TMEP(2)
VEC_EXT_TMEP(3)
VEC_EXT_TMEP(4)
#undef VEC_EXT_TMEP

}
#endif // VECTOR_H
