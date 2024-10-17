#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include "vector.h"
#include "matrix.h"
#include "quaternion.h"
#include "dualQuaternion.h"

namespace moon::math {

template<typename type> type radians(const type& angle) {
    return type(M_PI) * angle / type(180);
}

using vec2 = Vector<float, 2>;
using vec3 = Vector<float, 3>;
using vec4 = Vector<float, 4>;

using vec2d = Vector<double, 2>;
using vec3d = Vector<double, 3>;
using vec4d = Vector<double, 4>;

using vec2u = Vector<uint32_t, 2>;
using vec3u = Vector<uint32_t, 3>;
using vec4u = Vector<uint32_t, 4>;

using mat2 = Matrix<float, 2, 2>;
using mat3 = Matrix<float, 3, 3>;
using mat4 = Matrix<float, 4, 4>;

using mat2d = Matrix<double, 2, 2>;
using mat3d = Matrix<double, 3, 3>;
using mat4d = Matrix<double, 4, 4>;

using quat = Quaternion<float>;
using quatd = Quaternion<double>;

}

#endif // !LINEAR_ALGEBRA_H