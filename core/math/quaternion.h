#ifndef QUATERNION_H
#define QUATERNION_H

#include "matrix.h"

#include <array>

namespace moon::math {

#define VEC_(n) Vector<type, n>
#define VEC_3 VEC_(3)
#define VEC_4 VEC_(4)
#define MAT_(n) Matrix<type, n, n>
#define MAT_3 MAT_(3)
#define MAT_4 MAT_(4)
#define QUAT_TEMP template<typename type>
#define QUAT Quaternion<type>

#define QUAT_LIN_OPERATOR_GENERATOR(op)                                                     \
    Quaternion& operator op= (const Quaternion& q) { data op= q.data; return *this; }       \
    Quaternion operator op (const Quaternion& q) const { return Quaternion(*this) op= q; }

#define QUAT_SCAL_OPERATOR_GENERATOR(op)                                                    \
    Quaternion& operator op= (const type& c) { data op= c; return *this; }                  \
    Quaternion operator op (const type& c) const {  return Quaternion(*this) op= c; }       \
    friend Quaternion operator op (const type& c, const Quaternion& q) { return q op c; }

QUAT_TEMP class Quaternion
{
private:
    union { VEC_4 data; struct { type s, x, y, z;}; };

public:
    Quaternion() : s(type(1)), x(type(0)), y(type(0)), z(type(0)) {}
    Quaternion(const type& s, const type& x, const type& y, const type& z) : s(s), x(x), y(y), z(z) {}
    Quaternion(const type& s, const VEC_3& v) : s(s), x(v[0]), y(v[1]), z(v[2]) {}
    Quaternion(const Quaternion& q) : data(q.data) {}
    Quaternion(const VEC_4& v) : data(v) {}
    Quaternion& operator=(const Quaternion& q) { data = q.data; return *this; }

    type re() const { return s; }
    VEC_3 im() const { return VEC_3(x, y, z); }
    const VEC_4& vec4() const { return data; }
    std::array<type, 3> xyz() const { return std::array<type, 3>{x, y, z}; }
    std::array<type, 4> sxyz() const { return std::array<type, 4>{s,x,y,z}; }

    bool operator==(const Quaternion& q) const { return data == q.data; }
    bool operator!=(const Quaternion& q) const { return !(*this == q); }

    QUAT_LIN_OPERATOR_GENERATOR(+)
    QUAT_LIN_OPERATOR_GENERATOR(-)
    QUAT_SCAL_OPERATOR_GENERATOR(+)
    QUAT_SCAL_OPERATOR_GENERATOR(-)
    QUAT_SCAL_OPERATOR_GENERATOR(*)
    QUAT_SCAL_OPERATOR_GENERATOR(/)

    Quaternion& operator*=(const Quaternion& q) { return *this = *this * q; }
    Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            s * q.s - (x * q.x + y * q.y + z * q.z),
            s * q.x + x * q.s + (y * q.z - z * q.y),
            s * q.y + y * q.s + (z * q.x - x * q.z),
            s * q.z + z * q.s + (x * q.y - y * q.x));
    }

    type norm() const { return data.norm(); }
    Quaternion& normalize() { data.normalize(); return *this; }
    Quaternion normalized() const { return Quaternion(*this).normalize(); }
    Quaternion& conjugate() { x = -x; y = -y; z = -z; return *this; }
    Quaternion conjugated() const { return Quaternion(*this).conjugate(); }
    Quaternion& invert() { conjugate(); data *= type(1) / dot(data, data); return *this; }
    Quaternion inverted() const { return Quaternion(*this).invert(); }
};

QUAT_TEMP QUAT normalize(const QUAT& q) { return q.normalized(); }
QUAT_TEMP QUAT conjugate(const QUAT& q) { return q.conjugated(); }
QUAT_TEMP QUAT invert(const QUAT& q) { return q.inverted(); }

QUAT_TEMP QUAT slerp(const QUAT& x, const QUAT& y, const type& t) {
    const type cosTheta = dot(x.vec4(), y.vec4());
    QUAT z = cosTheta < type(0) ? type(-1) * y : y;

    if (cosTheta > static_cast<type>(1) - std::numeric_limits<type>::epsilon()) {
        return normalize(QUAT(mix(x.vec4(), z.vec4(), t)));
    }

    const type angle = std::acos(cosTheta);
    return normalize(std::sin((static_cast<type>(1) - t) * angle) / std::sin(angle) * x + std::sin(t * angle) / std::sin(angle) * z);
}

QUAT_TEMP std::ostream& operator<< (std::ostream& out, const QUAT& q) {
    return out << q.vec4();
}

QUAT_TEMP QUAT convert(const MAT_3& m) {
    // Shepperd method: numerically stable for all rotation angles.
    // Picks the branch with the largest denominator to avoid division by ~0.
    const type trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > type(0)) {
        const type t = std::sqrt(trace + type(1)) * type(2);
        return QUAT(type(0.25) * t, (m[2][1] - m[1][2]) / t,
                     (m[0][2] - m[2][0]) / t, (m[1][0] - m[0][1]) / t);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const type t = std::sqrt(type(1) + m[0][0] - m[1][1] - m[2][2]) * type(2);
        return QUAT((m[2][1] - m[1][2]) / t, type(0.25) * t,
                     (m[0][1] + m[1][0]) / t, (m[0][2] + m[2][0]) / t);
    } else if (m[1][1] > m[2][2]) {
        const type t = std::sqrt(type(1) - m[0][0] + m[1][1] - m[2][2]) * type(2);
        return QUAT((m[0][2] - m[2][0]) / t, (m[0][1] + m[1][0]) / t,
                     type(0.25) * t, (m[1][2] + m[2][1]) / t);
    } else {
        const type t = std::sqrt(type(1) - m[0][0] - m[1][1] + m[2][2]) * type(2);
        return QUAT((m[1][0] - m[0][1]) / t, (m[0][2] + m[2][0]) / t,
                     (m[1][2] + m[2][1]) / t, type(0.25) * t);
    }
}

QUAT_TEMP MAT_3 convert(const QUAT& q) {
    const auto [s, x, y, z] = q.sxyz();
    MAT_3 R(
        VEC_3(- y * y - z * z, x * y - z * s, x * z + y * s),
        VEC_3(x * y + z * s, - x * x - z * z, y * z - x * s),
        VEC_3(x * z - y * s, y * z + x * s, - x * x - y * y)
    );
    return MAT_3::identity() + type(2) * R;
}

QUAT_TEMP MAT_4 convert4x4(const QUAT& q) {
    const MAT_3 R = convert(q);
    return MAT_4(VEC_4(R[0], type(0)), VEC_4(R[1], type(0)), VEC_4(R[2], type(0)), VEC_4(VEC_3(type(0)), type(1)));
}

QUAT_TEMP QUAT convert(const type& yaw, const type& pitch, const type& roll)
{
    type cosy = std::cos(yaw   * type(0.5)), siny = std::sin(yaw   * type(0.5));
    type cosp = std::cos(pitch * type(0.5)), sinp = std::sin(pitch * type(0.5));
    type cosr = std::cos(roll  * type(0.5)), sinr = std::sin(roll  * type(0.5));

    return QUAT(cosy * cosp * cosr + siny * sinp * sinr, sinr * cosp * cosy - cosr * sinp * siny,
        cosr * sinp * cosy + sinr * cosp * siny, cosr * cosp * siny - sinr * sinp * cosy);
}

QUAT_TEMP QUAT convert(const type& angle, const VEC_3& axis) {
    type a05 = angle * type(0.5); return QUAT(std::cos(a05), std::sin(a05) * axis);
}

QUAT_TEMP VEC_3 convertToEulerAngles(const QUAT& q) {
    const auto [s, x, y, z] = q.sxyz();
    return VEC_3(
        std::atan((s * x + y * z) * type(2) / (type(1) - (x * x + y * y) * type(2))),
        std::asin((s * y - x * z) * type(2)),
        std::atan((s * z + y * x) * type(2) / (type(1) - (z * z + y * y) * type(2)))
    );
}

QUAT_TEMP QUAT convertToAnglesAndAxis(const QUAT& q) {
    return QUAT(std::acos(q.im()) * type(2), q.im().normalized());
}

QUAT_TEMP MAT_4 rotate(const QUAT& qu){ return convert4x4(qu); }

extern template class Quaternion<float>;
extern template class Quaternion<double>;

#undef VEC_
#undef VEC_3
#undef VEC_4
#undef MAT_
#undef MAT_3
#undef MAT_4
#undef QUAT_TEMP
#undef QUAT
#undef QUAT_LIN_OPERATOR_GENERATOR
#undef QUAT_SCAL_OPERATOR_GENERATOR

}
#endif // QUATERNION_H
