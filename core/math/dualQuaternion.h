#ifndef DUALQUATERNION_H
#define DUALQUATERNION_H

#include "quaternion.h"

namespace moon::math {

#define DQUAT_TEMP template<typename type>
#define QUAT Quaternion<type>
#define DQUAT DualQuaternion<type>
#define MAT_(n) Matrix<type, n, n>
#define MAT_4 MAT_(4)
#define VEC_(n) Vector<type, n>
#define VEC_3 VEC_(3)
#define VEC_4 VEC_(4)

#define DQUAT_LIN_OPERATOR_GENERATOR(op)                                                                    \
    DualQuaternion& operator op= (const DualQuaternion& dq) { p op= dq.p; q op= dq.q; return *this; }       \
    DualQuaternion operator op (const DualQuaternion& dq) const { return DualQuaternion(*this) op= dq; }

#define DQUAT_SCAL_OPERATOR_GENERATOR(op)                                                                   \
    DualQuaternion& operator op= (const type& c) { p op= c; q op= c; return *this; }                        \
    DualQuaternion operator op (const type& c) const {  return DualQuaternion(*this) op= c; }               \
    friend DualQuaternion operator op (const type& c, const DualQuaternion& q) { return q op c; }

DQUAT_TEMP class DualQuaternion
{
private:
    QUAT p;
    QUAT q;

public:
    DualQuaternion() = default;
    DualQuaternion(const QUAT& p, const QUAT& q) : p(p), q(q) {}
    DualQuaternion(const DualQuaternion& dq) : p(dq.p), q(dq.q) {}
    DualQuaternion& operator=(const DualQuaternion& dq) { p = dq.p; q = dq.q; return *this; }

    QUAT rotation() const { return p; }
    QUAT translation() const { return type(2) * q * p.conjugated(); }

    bool operator==(const DualQuaternion& dq) const { return p == dq.p && q == dq.q; }
    bool operator!=(const DualQuaternion& dq) const { return !(*this == dq); }

    type norm() const { return p.norm(); }
    DualQuaternion& normalize() {
        if (p.vec4().dot(q.vec4()) == 0.0f) { const auto n = norm(); p /= n; q /= n; } return *this;
    }
    DualQuaternion normalized() const { return DualQuaternion(*this).normalize(); }
    DualQuaternion& conjugate() { p.conjugate(); q.conjugate() *= type(-1); return *this; }
    DualQuaternion conjugated() const { return DualQuaternion(*this).conjugate(); }
    DualQuaternion& invert() { p.invert(); q = type(-1) * p * q * p; return *this; }
    DualQuaternion inverted() const { return DualQuaternion(*this).invert(); }

    DQUAT_LIN_OPERATOR_GENERATOR(+)
    DQUAT_LIN_OPERATOR_GENERATOR(-)

    DQUAT_SCAL_OPERATOR_GENERATOR(+)
    DQUAT_SCAL_OPERATOR_GENERATOR(-)
    DQUAT_SCAL_OPERATOR_GENERATOR(*)

    DualQuaternion operator*(const DualQuaternion& dq) const { return DualQuaternion(p * dq.p, p * dq.q + q * dq.p); }
    DualQuaternion& operator*=(const DualQuaternion& dq) { return *this = *this * dq; }
};

DQUAT_TEMP std::ostream& operator<< (std::ostream& out, const DQUAT& dq) {
    return out << dq.p << "\t" << dq.q;
}

DQUAT_TEMP DQUAT convert(const QUAT& rotation, const QUAT& translation) {
    return DQUAT(rotation, type(0.5) * translation * rotation);
}

DQUAT_TEMP MAT_4 convert(const DQUAT& quat) {
    const auto r = quat.rotation();
    const auto [x, y, z] = quat.translation().xyz();
    const auto R = convert(r);
    return MAT_4(VEC_4(R[0], x), VEC_4(R[1], y), VEC_4(R[2], z), VEC_4(VEC_3(type(0)), type(1)));
}

#undef DQUAT_TEMP
#undef QUAT
#undef DQUAT
#undef MAT_
#undef MAT_4
#undef VEC_
#undef VEC_3
#undef VEC_4
#undef DQUAT_LIN_OPERATOR_GENERATOR
#undef DQUAT_SCAL_OPERATOR_GENERATOR

extern template class DualQuaternion<float>;
extern template class DualQuaternion<double>;

}
#endif // DUALQUATERNION_H
