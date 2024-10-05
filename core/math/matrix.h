#ifndef MATRIX_H
#define MATRIX_H

#include <fstream>
#include <limits>
#define _USE_MATH_DEFINES
#include <math.h>

#include "vector.h"

namespace moon::math {

template<typename type, uint32_t n, uint32_t m>
class BaseMatrix {
protected:
    BaseVector<type, m> mat[n];

public:
    BaseMatrix() = default;
    BaseMatrix(const BaseMatrix<type,n,m>& other);
    BaseMatrix<type,n,m>& operator=(const BaseMatrix<type,n,m>& other);

    BaseVector<type, m>& operator[](uint32_t i);
    const BaseVector<type, m>& operator[](uint32_t i) const;

    bool operator==(const BaseMatrix<type, n, m>& other) const;
    bool operator!=(const BaseMatrix<type, n, m>& other) const;

    BaseMatrix<type, n, m> operator+(const BaseMatrix<type, n, m>& other) const;
    BaseMatrix<type, n, m> operator-(const BaseMatrix<type, n, m>& other) const;
    BaseVector<type, n> operator*(const BaseVector<type, m>& other) const;
    BaseMatrix<type, n, m> operator/(const type& c) const;

    BaseMatrix<type, n, m>& operator+=(const BaseMatrix<type, n, m>& other);
    BaseMatrix<type, n, m>& operator-=(const BaseMatrix<type, n, m>& other);

    BaseMatrix<type, n, m>& operator+=(const type& c);
    BaseMatrix<type, n, m>& operator-=(const type& c);
    BaseMatrix<type, n, m>& operator*=(const type& c);
    BaseMatrix<type, n, m>& operator/=(const type& c);

    BaseMatrix<type, n-1, m-1> extract(uint32_t i, uint32_t j) const;

    template<typename T, uint32_t M, uint32_t N> friend BaseMatrix<T,N,M> transpose(const BaseMatrix<T,N,M>& other);

    template<typename T, uint32_t N, uint32_t M> friend BaseMatrix<T,N,M> operator+(const T& c, const BaseMatrix<T,N,M>& other);
    template<typename T, uint32_t N, uint32_t M> friend BaseMatrix<T,N,M> operator-(const T& c, const BaseMatrix<T,N,M>& other);
    template<typename T, uint32_t N, uint32_t M> friend BaseMatrix<T,N,M> operator*(const T& c, const BaseMatrix<T,N,M>& other);

    template<typename T, uint32_t N, uint32_t M> friend BaseMatrix<T,N,M> operator+(const BaseMatrix<T,N,M>& other, const T& c);
    template<typename T, uint32_t N, uint32_t M> friend BaseMatrix<T,N,M> operator-(const BaseMatrix<T,N,M>& other, const T& c);
    template<typename T, uint32_t N, uint32_t M> friend BaseMatrix<T,N,M> operator*(const BaseMatrix<T,N,M>& other, const T& c);

    template<typename T, uint32_t N, uint32_t M, uint32_t K> friend BaseMatrix<T,N,K> operator*(const BaseMatrix<T,N,M>& left, const BaseMatrix<T,M,K>& right);

    template<typename T, uint32_t N, uint32_t M> friend std::ostream& operator<<(std::ostream& out, const BaseMatrix<T,N,M>& other);
};

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type,n,m>::BaseMatrix(const BaseMatrix<type,n,m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] = other.mat[i];
    }
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type,n,m>& BaseMatrix<type,n,m>::operator=(const BaseMatrix<type,n,m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] = other.mat[i];
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
BaseVector<type, m>& BaseMatrix<type,n,m>::operator[](uint32_t i) {
    return mat[i];
}

template<typename type, uint32_t n, uint32_t m>
const BaseVector<type, m>& BaseMatrix<type,n,m>::operator[](uint32_t i) const {
    return mat[i];
}

template<typename type, uint32_t n, uint32_t m>
bool BaseMatrix<type,n,m>::operator==(const BaseMatrix<type, n, m>& other) const {
    bool result = true;
    for(uint32_t i = 0; i < n; i++) {
        result &= (mat[i] == other.mat[i]);
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
bool BaseMatrix<type,n,m>::operator!=(const BaseMatrix<type, n, m>& other) const {
    return !(*this == other);
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m> BaseMatrix<type,n,m>::operator+(const BaseMatrix<type, n, m>& other) const {
    BaseMatrix<type,n,m> result(*this);
    for(uint32_t i = 0; i < n; i++) {
        result[i] += other.mat[i];
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m> BaseMatrix<type,n,m>::operator-(const BaseMatrix<type, n, m>& other) const {
    BaseMatrix<type,n,m> result(*this);
    for(uint32_t i = 0; i < n; i++) {
        result[i] -= other.mat[i];
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
BaseVector<type, n> BaseMatrix<type,n,m>::operator*(const BaseVector<type, m>& other) const {
    BaseVector<type,n> result;
    for(uint32_t i = 0; i < n; i++) {
        result[i] = dot(mat[i], other);
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m> BaseMatrix<type,n,m>::operator/(const type& c) const {
    BaseMatrix<type,n,m> result(*this);
    for(uint32_t i = 0; i < n; i++) {
        result[i] /= c;
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m>& BaseMatrix<type,n,m>::operator+=(const BaseMatrix<type, n, m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] += other.mat[i];
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m>& BaseMatrix<type,n,m>::operator-=(const BaseMatrix<type, n, m>& other) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] -= other.mat[i];
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m>& BaseMatrix<type,n,m>::operator+=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] += c;
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m>& BaseMatrix<type,n,m>::operator-=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] -= c;
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m>& BaseMatrix<type,n,m>::operator*=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] *= c;
    }
    return *this;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n, m>& BaseMatrix<type,n,m>::operator/=(const type& c) {
    for(uint32_t i = 0; i < n; i++) {
        mat[i] /= c;
    }
    return *this;
}

template<typename T, uint32_t N, uint32_t M> BaseMatrix<T,N,M> operator+(const T& c, const BaseMatrix<T,N,M>& other) {
    BaseMatrix<T,N,M> result;
    for(uint32_t i = 0; i < N; i++) {
        result[i] = c + other[i];
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> BaseMatrix<T,N,M> operator-(const T& c, const BaseMatrix<T,N,M>& other) {
    BaseMatrix<T,N,M> result;
    for(uint32_t i = 0; i < N; i++) {
        result[i] = c - other[i];
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> BaseMatrix<T,N,M> operator*(const T& c, const BaseMatrix<T,N,M>& other) {
    BaseMatrix<T,N,M> result;
    for(uint32_t i = 0; i < N; i++) {
        result[i] = c * other[i];
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> BaseMatrix<T,N,M> operator+(const BaseMatrix<T,N,M>& other, const T& c) {
    BaseMatrix<T,N,M> result(other);
    for(uint32_t i = 0; i < N; i++) {
        result[i] += c;
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> BaseMatrix<T,N,M> operator-(const BaseMatrix<T,N,M>& other, const T& c) {
    BaseMatrix<T,N,M> result(other);
    for(uint32_t i = 0; i < N; i++) {
        result[i] -= c;
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> BaseMatrix<T,N,M> operator*(const BaseMatrix<T,N,M>& other, const T& c) {
    BaseMatrix<T,N,M> result(other);
    for(uint32_t i = 0; i < N; i++) {
        result[i] *= c;
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M, uint32_t K> BaseMatrix<T,N,K> operator*(const BaseMatrix<T,N,M>& left, const BaseMatrix<T,M,K>& right) {
    BaseMatrix<T,N,K> result;
    for(uint32_t i = 0; i < N; i++) {
        for(uint32_t j = 0; j < M; j++) {
            for(uint32_t k = 0; k < K; k++) {
                result[i][k] += left[i][j] * right[j][k];
            }
        }
    }
    return result;
}

template<typename T, uint32_t N, uint32_t M> std::ostream& operator<<(std::ostream& out, const BaseMatrix<T,N,M>& other) {
    for(uint32_t i = 0; i < N; i++) {
        std::cout << other.mat[i] << '\n';
    }
    return out;
}

template<typename type, uint32_t n, uint32_t m>
BaseMatrix<type, n-1, m-1> BaseMatrix<type,n,m>::extract(uint32_t i, uint32_t j) const {
    BaseMatrix<type, n-1, m-1> result;
    for(uint32_t it = 0, ie = 0; it < n; it++){
        if(it != i){
            for(uint32_t jt = 0, je = 0; jt < m; jt++){
                if(jt != j){
                    result[ie][je] = mat[it][jt];
                    je++;
                }
            }
            ie++;
        }
    }
    return result;
}

template<typename T, uint32_t M, uint32_t N> BaseMatrix<T,N,M> transpose(const BaseMatrix<T,N,M>& other) {
    BaseMatrix<T,M,N> result;
    for(uint32_t i = 0; i < N; i++) {
        for(uint32_t j = 0; j < M; j++) {
            result[j][i] = other[i][j];
        }
    }
    return result;
}

template<typename type, uint32_t n, uint32_t m> class Matrix;

template<typename type>
class Matrix<type, 2, 2> : public BaseMatrix<type, 2, 2>
{
private:
    static constexpr uint32_t n = 2;

public:
    Matrix(const Vector<type, n>& v0, const Vector<type, n>& v1) {
        this->mat[0] = v0;
        this->mat[1] = v1;
    }
    Matrix(
        const type& m00, const type& m01,
        const type& m10, const type& m11
    ) {
        this->mat[0] = Vector<type,n>{m00, m01};
        this->mat[1] = Vector<type,n>{m10, m11};
    }
    Matrix(const BaseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
    }
    Matrix<type,n,n>& operator=(const BaseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
        return *this;
    }
};

template<typename type>
class Matrix<type, 3, 3> : public BaseMatrix<type, 3, 3>
{
private:
    static constexpr uint32_t n = 3;

public:
    Matrix() : BaseMatrix<type,n,n>() {}
    Matrix(const Vector<type, n>& v0, const Vector<type, n>& v1, const Vector<type, n>& v2) {
        this->mat[0] = v0;
        this->mat[1] = v1;
        this->mat[2] = v2;
    }
    Matrix(const type& m) {
        this->mat[0] = Vector<type,n>{m, 0.0f, 0.0f};
        this->mat[1] = Vector<type,n>{0.0f, m, 0.0f};
        this->mat[2] = Vector<type,n>{0.0f, 0.0f, m};
    }
    Matrix(
        const type& m00, const type& m01, const type& m02,
        const type& m10, const type& m11, const type& m12,
        const type& m20, const type& m21, const type& m22
    ) {
        this->mat[0] = Vector<type,n>{m00, m01, m02};
        this->mat[1] = Vector<type,n>{m10, m11, m12};
        this->mat[2] = Vector<type,n>{m20, m21, m22};
    }
    Matrix(const BaseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
    }
    Matrix<type,n,n>& operator=(const BaseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
        return *this;
    }
};

template<typename type>
class Matrix<type, 4, 4> : public BaseMatrix<type, 4, 4>
{
private:
    static constexpr uint32_t n = 4;

public:
    Matrix() : BaseMatrix<type,n,n>() {}
    Matrix(std::ifstream& file) {
        for(uint32_t i = 0; i < n; i++){
            for(uint32_t j = 0; j < n; j++){
                if(file.peek()!=EOF){
                    file >> this->mat[i][j];
                } else {
                    this->mat[i][j] = type(0);
                }
            }
        }
    }
    Matrix(const type& m) {
        this->mat[0] = Vector<type,n>{m, 0.0f, 0.0f, 0.0f};
        this->mat[1] = Vector<type,n>{0.0f, m, 0.0f, 0.0f};
        this->mat[2] = Vector<type,n>{0.0f, 0.0f, m, 0.0f};
        this->mat[3] = Vector<type,n>{0.0f, 0.0f, 0.0f, m};
    }
    Matrix(const Vector<type, n>& v0, const Vector<type, n>& v1, const Vector<type, n>& v2, const Vector<type, n>& v3) {
        this->mat[0] = v0;
        this->mat[1] = v1;
        this->mat[2] = v2;
        this->mat[3] = v3;
    }
    Matrix(
        const type& m00, const type& m01, const type& m02, const type& m03,
        const type& m10, const type& m11, const type& m12, const type& m13,
        const type& m20, const type& m21, const type& m22, const type& m23,
        const type& m30, const type& m31, const type& m32, const type& m33
    ) {
        this->mat[0] = Vector<type,n>{m00, m01, m02, m03};
        this->mat[1] = Vector<type,n>{m10, m11, m12, m13};
        this->mat[2] = Vector<type,n>{m20, m21, m22, m23};
        this->mat[3] = Vector<type,n>{m30, m31, m32, m33};
    }
    Matrix(const BaseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
    }
    Matrix<type,n,n>& operator=(const BaseMatrix<type,n,n>& other) {
        for(uint32_t i = 0; i < n; i++){
            this->mat[i] = other[i];
        }
        return *this;
    }
};

template<typename type>
type det(const Matrix<type,2,2>& m) {
    return  m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

template<typename type, uint32_t n>
type det(const Matrix<type,n,n>& m) {
    type result = type(0);
    for(uint32_t i = 0; i < n; i++){
        result += (i % 2 ? -1.0f : 1.0f) * m[0][i] * det(Matrix<type,n-1,n-1>(m.extract(0,i)));
    }
    return  result;
}

template<typename type, uint32_t n>
Matrix<type,n,n> inverse(const Matrix<type,n,n>& m) {
    Matrix<type,n,n> result;
    for(uint32_t i = 0; i < n; i++){
        for(uint32_t j = 0; j < n; j++){
            result[i][j] = ((i + j) % 2 ? -1.0f : 1.0f) * det(Matrix<type,n-1,n-1>(m.extract(i,j)));
        }
    }
    return transpose(result) / det(m);
}

template<typename type>
Matrix<type,4,4> translate(Vector<type,3> tr) {
    Matrix<float,4,4> m{1.0f};
    m[0][3] += tr[0];
    m[1][3] += tr[1];
    m[2][3] += tr[2];
    return m;
}

template<typename type>
Matrix<type,4,4> scale(Vector<type,3> sc) {
    Matrix<float,4,4> m{1.0f};
    m[0][0] *= sc[0];
    m[1][1] *= sc[1];
    m[2][2] *= sc[2];
    return m;
}

template<typename type>
Matrix<type,4,4> perspective(const type& fovy, const type& aspect, const type& n = std::numeric_limits<type>::min(), const type& f = std::numeric_limits<type>::max()) {
    const type a = type(1) / std::tan(fovy / type(2));

    Matrix<type,4,4> m(0.0f);
    m[0][0] = a / aspect;
    m[1][1] = - a;
    m[2][2] = f == std::numeric_limits<type>::max() ? - type(1) : (f + n) / (n - f);
    m[2][3] = type(2) * n * (f == std::numeric_limits<type>::max() ? - type(1) : f / (n - f));
    m[3][2] = - type(1);

    return m;
}

template<typename type>
Matrix<type,4,4> orthographic(const type left, const type right, const type bottom, const type top, const type n, const type f) {
    Matrix<type,4,4> m(0.0f);
    m[0][0] = type(2) / (right - left);
    m[1][1] = - type(2) / (top - bottom);
    m[2][2] = - type(2) / (f - n);
    m[3][3] = type(1);

    m[0][3] = - (right + left) / (right - left);
    m[1][3] = - (top + bottom) / (top - bottom);
    m[2][3] = - (f + n) / (f - n);

    return m;
}

template<typename type>
Matrix<type,4,4> orthographic(const type& width, const type& height, const type& n, const type& f) {
    return orthographic(- width / type(2), width / type(2), - height / type(2), height / type(2), n, f);
}

template<typename type>
type radians(const type& angle) {
    return type(M_PI) * angle / type(180);
}

extern template class Matrix<float, 2, 2>;
extern template class Matrix<double, 2, 2>;

extern template class Matrix<float, 3, 3>;
extern template class Matrix<double, 3, 3>;

extern template class Matrix<float, 4, 4>;
extern template class Matrix<double, 4, 4>;

}
#endif // MATRIX_H
