#ifndef MATRIX_H
#define MATRIX_H

#include <limits>
#include <cstdarg>
#define _USE_MATH_DEFINES
#include <math.h>

#include "vector.h"

namespace moon::math {

#define for_(ind, range) for (uint32_t ind = 0; ind < range; ind++)
#define for_i_n for_(i, n)
#define for_j_m for_(j, m)
#define MAT_TEMP_N_M template<typename type, uint32_t n, uint32_t m>
#define MAT_(ROWS, COLS) BaseMatrix<type, ROWS, COLS>
#define MAT_N_M MAT_(n, m)
#define MAT_M_N MAT_(m, n)
#define MAT_EXPR(T, exp) T res{}; exp; return res;

#define MAT_SCAL_OPERATOR_GENERATOR(op)                                                             \
    BaseMatrix& operator op= (const type& c) { for_i_n mat[i] op= c; return *this; }                \
    BaseMatrix operator op (const type& c) const { return BaseMatrix(*this) op= c; }                \
    friend BaseMatrix operator op (const type& c, const BaseMatrix& a) { return a op c; }

#define MAT_LIN_OPERATOR_GENERATOR(op)                                                              \
    BaseMatrix& operator op= (const BaseMatrix& a) { for_i_n mat[i] op= a.mat[i]; return *this;}    \
    BaseMatrix operator op (const BaseMatrix& a) const { return BaseMatrix(*this) op= a;}

MAT_TEMP_N_M class BaseMatrix {
public:
    using Transposed = MAT_M_N;
    using Reduced = BaseMatrix<type, n - 1, m - 1>;
    using RowVector = BaseVector<type, m>;
    using ColVector = BaseVector<type, n>;

protected:
    RowVector mat[n];

public:
    BaseMatrix() = default;
    BaseMatrix(const BaseMatrix& a) { copy(a); }
    BaseMatrix& operator=(const BaseMatrix& a) { copy(a); return *this; }
    void copy(const BaseMatrix& a) { for_i_n mat[i] = a.mat[i]; }

    RowVector& operator[](uint32_t i) { return mat[i]; }
    const RowVector& operator[](uint32_t i) const { return mat[i]; }

    bool operator!=(const BaseMatrix& a) const { MAT_EXPR(bool, for_i_n res |= mat[i] != a.mat[i]) }
    bool operator==(const BaseMatrix& a) const { return !(*this != a); }

    BaseMatrix operator+() const { return *this; }
    BaseMatrix operator-() const { MAT_EXPR(BaseMatrix, for_i_n res[i] = -mat[i]) }

    ColVector operator*(const RowVector& rowVector) const {
        MAT_EXPR(ColVector, for_i_n res[i] = mat[i].dot(rowVector))
    }

    Transposed transpose() const {
        MAT_EXPR(Transposed, for_i_n for_j_m res[j][i] = mat[i][j])
    }

    Reduced extract(uint32_t idel, uint32_t jdel) const {
        if(idel > n - 1 || jdel > m - 1) return Reduced();
        MAT_EXPR(Reduced,
            uint32_t i = 0; for_(ibase, n) { if (ibase == idel) continue;
                uint32_t j = 0; for_(jbase, m) { if (jbase == jdel) continue;
                    res[i][j] = mat[ibase][jbase]; j++; } i++;})
    }

    MAT_SCAL_OPERATOR_GENERATOR(+)
    MAT_SCAL_OPERATOR_GENERATOR(-)
    MAT_SCAL_OPERATOR_GENERATOR(*)
    MAT_SCAL_OPERATOR_GENERATOR(/)

    MAT_LIN_OPERATOR_GENERATOR(+)
    MAT_LIN_OPERATOR_GENERATOR(-)
};

MAT_TEMP_N_M std::ostream& operator<<(std::ostream& out, const MAT_N_M& a) {
    for_i_n std::cout << a[i] << '\n'; return out;
}

MAT_TEMP_N_M MAT_M_N transpose(const MAT_N_M& a) {
    return a.transpose();
}

template<typename type, uint32_t n, uint32_t w, uint32_t m>
MAT_N_M operator*(const MAT_(n, w)& l, const MAT_(w, m)& r) {
    MAT_EXPR(MAT_N_M, for_(i, n) for_(k, m) for_(j, w) res[i][k] += l[i][j] * r[j][k])
}

#undef MAT_N_M
#undef MAT_M_N
#undef MAT_SCAL_OPERATOR_GENERATOR
#undef MAT_LIN_OPERATOR_GENERATOR

MAT_TEMP_N_M class Matrix;

#define MAT_N_N MAT_(n, n)
#define MAT_TEMP_N template<typename type, uint32_t n>

MAT_TEMP_N class Matrix<type, n, n> : public MAT_N_N
{
public:
    using RowVector = BaseVector<type, n>;

    Matrix() = default;
    Matrix(const type& mii) { for_i_n mat[i][i] = mii; }
    Matrix(const RowVector vi, ...) {
        std::va_list args; va_start(args, vi); for_i_n mat[i] = (i == 0 ? vi : va_arg(args, RowVector)); va_end(args);
    }
    Matrix(const MAT_N_N& a) : MAT_N_N(a) {}
    Matrix& operator=(const MAT_N_N& a) { this->copy(a); return *this; }

    static Matrix identity() {return Matrix(type(1));}
};

#define MAT_TEMP template<typename type>
#define MAT_sign(i) ((i) % 2 ? type(-1) : type(1))

MAT_TEMP type det(const MAT_(2,2)& m) {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

MAT_TEMP_N type det(const MAT_N_N& m) {
    MAT_EXPR(type, for_i_n res += MAT_sign(i) * m[0][i] * det(m.extract(0,i)))
}

MAT_TEMP_N MAT_N_N inverse(const MAT_N_N& m) {
    MAT_EXPR(MAT_N_N, const auto invd = type(1) / det(m); for_i_n for_(j, n) res[j][i] = invd * MAT_sign(i + j) * det(m.extract(i,j)))
}

#define MAT_4 Matrix<type, 4, 4>
#define VEC_3 Vector<type, 3>

MAT_TEMP MAT_4 translate(const VEC_3& v) {
    MAT_4 m{1.0f}; for_(i, 3) m[i][3] += v[i]; return m;
}

MAT_TEMP MAT_4 scale(const VEC_3& v) {
    MAT_4 m{1.0f}; for_(i, 3) m[i][i] *= v[i]; return m;
}

MAT_TEMP MAT_4 perspective(const type& fovy, const type& aspect, const type& n = std::numeric_limits<type>::min(), const type& f = std::numeric_limits<type>::max()) {
    MAT_4 m;
    m[1][1] = - type(1) / std::tan(fovy / type(2));
    m[0][0] = - m[1][1] / aspect;
    m[2][2] = f == std::numeric_limits<type>::max() ? - type(1) : (f + n) / (n - f);
    m[2][3] = type(2) * n * (f == std::numeric_limits<type>::max() ? - type(1) : f / (n - f));
    m[3][2] = - type(1);

    return m;
}

MAT_TEMP MAT_4 orthographic(const type left, const type right, const type bottom, const type top, const type n, const type f) {
    MAT_4 m;
    m[0][0] = type(2) / (right - left);
    m[1][1] = - type(2) / (top - bottom);
    m[2][2] = - type(2) / (f - n);
    m[3][3] = type(1);

    m[0][3] = - (right + left) / (right - left);
    m[1][3] = - (top + bottom) / (top - bottom);
    m[2][3] = - (f + n) / (f - n);

    return m;
}

MAT_TEMP MAT_4 orthographic(const type& width, const type& height, const type& n, const type& f) {
    return orthographic(- width / type(2), width / type(2), - height / type(2), height / type(2), n, f);
}

#define MAT_EXT_TMEP(n) extern template class Matrix<float, n, n>; extern template class Matrix<double, n, n>;

MAT_EXT_TMEP(2)
MAT_EXT_TMEP(3)
MAT_EXT_TMEP(4)
#undef MAT_EXT_TMEP

#undef for_
#undef for_i_n
#undef for_j_m
#undef MAT_
#undef MAT_N_N
#undef MAT_TEMP_N_M
#undef MAT_TEMP_N
#undef MAT_TEMP
#undef MAT_EXPR
#undef MAT_sign
#undef MAT_4
#undef VEC_3

}
#endif // MATRIX_H
