#ifndef QUATERNION_H
#define QUATERNION_H

#include "matrix.h"

namespace moon::math {

template<typename type>
class Quaternion
{
private:
    type s;
    type x;
    type y;
    type z;

public:
    Quaternion();
    Quaternion(const Quaternion<type>& other);
    Quaternion(const type& s,const type& x,const type& y,const type& z);
    Quaternion(const type& s,const Vector<type, 3>& v);
    Quaternion<type>& operator=(const Quaternion<type>& other);
    ~Quaternion() = default;

    type                re() const;
    Vector<type, 3>     im() const;

    bool                operator==(const Quaternion<type>& other) const;
    bool                operator!=(const Quaternion<type>& other) const;
    Quaternion<type>    operator+ (const Quaternion<type>& other) const;
    Quaternion<type>    operator- (const Quaternion<type>& other) const;
    Quaternion<type>    operator* (const Quaternion<type>& other) const;
    Quaternion<type>&   operator+=(const Quaternion<type>& other);
    Quaternion<type>&   operator-=(const Quaternion<type>& other);
    Quaternion<type>&   operator*=(const Quaternion<type>& other);

    Quaternion<type>&   normalize();
    Quaternion<type>&   conjugate();
    Quaternion<type>&   invert();

    template<typename T> friend Quaternion<T>   normalize(const Quaternion<T>& quat);
    template<typename T> friend Quaternion<T>   conjugate(const Quaternion<T>& quat);
    template<typename T> friend Quaternion<T>   invert(const Quaternion<T>& quat);

    template<typename T> friend Quaternion<T> operator* (const T& c, const Quaternion<T>& quat);
    template<typename T> friend std::ostream& operator<< (std::ostream & out, const Quaternion<T>& quat);

    template<typename T> friend Quaternion<T> convert(const Matrix<T,3,3>& O3);
    template<typename T> friend Matrix<T,3,3> convert(const Quaternion<T>& quat);
    template<typename T> friend Matrix<T,4,4> convert4x4(const Quaternion<T>& quat);

    template<typename T> friend Quaternion<T> convert(const T& yaw, const T& pitch, const T& roll);
    template<typename T> friend Quaternion<T> convert(const T& angle, const Vector<T,3>& axis);

    template<typename T> friend Vector<T,3> convertToEulerAngles(const Quaternion<T>& quat);
    template<typename T> friend Quaternion<T> convertToAnglesAndAxis(const Quaternion<T>& quat);

    template<typename T> friend Quaternion<T> slerp(const Quaternion<T>& quat1, const Quaternion<T>& quat2, const T& t);
};


template<typename type>
Quaternion<type>::Quaternion():
    s(static_cast<type>(0)),
    x(static_cast<type>(0)),
    y(static_cast<type>(0)),
    z(static_cast<type>(0))
{}

template<typename type>
Quaternion<type>::Quaternion(const Quaternion<type>& other):
    s(other.s),
    x(other.x),
    y(other.y),
    z(other.z)
{}

template<typename type>
Quaternion<type>::Quaternion(const type& s,const type& x,const type& y,const type& z):
    s(s),
    x(x),
    y(y),
    z(z)
{}

template<typename type>
Quaternion<type>::Quaternion(const type& s, const Vector<type, 3>& v):
    s(s),
    x(static_cast<type>(v[0])),
    y(static_cast<type>(v[1])),
    z(static_cast<type>(v[2]))
{}

template<typename type>
Quaternion<type>& Quaternion<type>::operator=(const Quaternion<type>& other)
{
    s = other.s;
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
}

template<typename type>
type                Quaternion<type>::re()const
{
    return s;
}

template<typename type>
Vector<type, 3>    Quaternion<type>::im()const
{
    return Vector<type, 3>(x,y,z);
}

template<typename type>
bool                Quaternion<type>::operator==(const Quaternion<type>& other)const
{
    return x==other.x&&y==other.y&&z==other.z&&s==other.s;
}

template<typename type>
bool                Quaternion<type>::operator!=(const Quaternion<type>& other)const
{
    return !(x==other.x&&y==other.y&&z==other.z&&s==other.s);
}

template<typename type>
Quaternion<type>    Quaternion<type>::operator+(const Quaternion<type>& other)const
{
    return Quaternion<type>(s+other.s,x+other.x,y+other.y,z+other.z);
}

template<typename type>
Quaternion<type>    Quaternion<type>::operator-(const Quaternion<type>& other)const
{
    return Quaternion<type>(s-other.s,x-other.x,y-other.y,z-other.z);
}

template<typename type>
Quaternion<type>    Quaternion<type>::operator*(const Quaternion<type>& other)const
{
    return Quaternion<type>(
        s*other.s - (x*other.x + y*other.y + z*other.z),
        s*other.x + other.s*x + (y*other.z-z*other.y),
        s*other.y + other.s*y + (z*other.x-x*other.z),
        s*other.z + other.s*z + (x*other.y-y*other.x)
    );
}

template<typename type>
Quaternion<type>&   Quaternion<type>::operator+=(const Quaternion<type>& other)
{
    s += other.s;
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

template<typename type>
Quaternion<type>&   Quaternion<type>::operator-=(const Quaternion<type>& other)
{
    s -= other.s;
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

template<typename type>
Quaternion<type>&   Quaternion<type>::operator*=(const Quaternion<type>& other)
{
    Quaternion<type> copy(*this);
    *this = copy*other;

    return *this;
}

template<typename T>
std::ostream& operator<< (std::ostream & out, const Quaternion<T>& quat)
{
    out<<quat.s<<'\t'<<quat.x<<'\t'<<quat.y<<'\t'<<quat.z;
    return out;
}

template<typename T>
Quaternion<T> operator* (const T& c,const Quaternion<T>& quat)
{
    return Quaternion<T>(c*quat.s,c*quat.x,c*quat.y,c*quat.z);
}

template<typename type>
Quaternion<type>&   Quaternion<type>::normalize()
{
    type norma = s*s+x*x+y*y+z*z;
    norma = std::sqrt(norma);
    s /= norma;
    x /= norma;
    y /= norma;
    z /= norma;
    return *this;
}

template<typename type>
Quaternion<type>&   Quaternion<type>::conjugate()
{
    x = -x;
    y = -y;
    z = -z;
    return *this;
}

template<typename type>
Quaternion<type>&   Quaternion<type>::invert()
{
    Quaternion<type> quat(*this);
    Quaternion<type> ivNorma = quat*this->conjugate();
    ivNorma.s = std::sqrt(ivNorma.s);
    *this = ivNorma*(*this);
    return *this;
}


template<typename T>
Quaternion<T>   normalize(const Quaternion<T>& quat)
{
    T norma = quat.s*quat.s+quat.x*quat.x+quat.y*quat.y+quat.z*quat.z;
    norma = std::sqrt(norma);
    return Quaternion<T>(quat.s/norma,quat.x/norma,quat.y/norma,quat.z/norma);
}

template<typename T>
Quaternion<T>   conjugate(const Quaternion<T>& quat)
{
    return Quaternion<T>(quat.s,-quat.x,-quat.y,-quat.z);
}

template<typename T>
Quaternion<T>   invert(const Quaternion<T>& quat)
{
    Quaternion<T> ivNorma = quat*conjugate(quat);
    ivNorma.s = std::sqrt(ivNorma.s);
    return ivNorma*conjugate(quat);
}

template<typename T>
Quaternion<T> convert(const Matrix<T,3,3>& O3)
{
    Quaternion<T> quat;

    quat.s = std::sqrt(1.0f+O3[0][0]+O3[1][1]+O3[2][2])/2.0f;

    quat.z = (O3[1][0]-O3[0][1])/(T(4)*quat.s);
    quat.y = (O3[0][2]-O3[2][0])/(T(4)*quat.s);
    quat.x = (O3[2][1]-O3[1][2])/(T(4)*quat.s);

    return quat;
}

template<typename T>
Matrix<T,3,3> convert(const Quaternion<T>& quat)
{
    Matrix<T,3,3> R;

    R[0][0] = T(1) - T(2)*(quat.y*quat.y + quat.z*quat.z);      R[0][1] = T(2)*(quat.x*quat.y - quat.z*quat.s);         R[0][2] = T(2)*(quat.x*quat.z + quat.y*quat.s);
    R[1][0] = T(2)*(quat.x*quat.y + quat.z*quat.s);             R[1][1] = T(1) - T(2)*(quat.x*quat.x + quat.z*quat.z);  R[1][2] = T(2)*(quat.y*quat.z - quat.x*quat.s);
    R[2][0] = T(2)*(quat.x*quat.z - quat.y*quat.s);             R[2][1] = T(2)*(quat.y*quat.z + quat.x*quat.s);         R[2][2] = T(1) - T(2)*(quat.x*quat.x + quat.y*quat.y);

    return R;
}

template<typename T>
Matrix<T,4,4> convert4x4(const Quaternion<T>& quat)
{
    Matrix<T,4,4> R{0.0f};

    R[0][0] = T(1) - T(2)*(quat.y*quat.y + quat.z*quat.z);      R[0][1] = T(2)*(quat.x*quat.y - quat.z*quat.s);         R[0][2] = T(2)*(quat.x*quat.z + quat.y*quat.s);
    R[1][0] = T(2)*(quat.x*quat.y + quat.z*quat.s);             R[1][1] = T(1) - T(2)*(quat.x*quat.x + quat.z*quat.z);  R[1][2] = T(2)*(quat.y*quat.z - quat.x*quat.s);
    R[2][0] = T(2)*(quat.x*quat.z - quat.y*quat.s);             R[2][1] = T(2)*(quat.y*quat.z + quat.x*quat.s);         R[2][2] = T(1) - T(2)*(quat.x*quat.x + quat.y*quat.y);
    R[3][3] = T(1);

    return R;
}

template<typename T>
Quaternion<T> convert(const T& yaw, const T& pitch, const T& roll)
{
    T cosy = std::cos(yaw*T(0.5));
    T siny = std::sin(yaw*T(0.5));
    T cosp = std::cos(pitch*T(0.5));
    T sinp = std::sin(pitch*T(0.5));
    T cosr = std::cos(roll*T(0.5));
    T sinr = std::sin(roll*T(0.5));

    T s = cosy*cosp*cosr + siny*sinp*sinr;
    T x = sinr*cosp*cosy - cosr*sinp*siny;
    T y = cosr*sinp*cosy + sinr*cosp*siny;
    T z = cosr*cosp*siny - sinr*sinp*cosy;

    return Quaternion<T>(s,x,y,z);
}

template<typename T>
Quaternion<T> convert(const T& angle, const Vector<T,3>& axis)
{
    return Quaternion<T>(std::cos(angle*T(0.5)),std::sin(angle*T(0.5))*Vector<T,3>(axis[0],axis[1],axis[2]));
}

template<typename T>
Vector<T,3> convertToEulerAngles(const Quaternion<T>& quat)
{
    return  Vector<T,3>(std::atan((quat.s*quat.x+quat.y*quat.z)*T(2)/(T(1)-(quat.x*quat.x+quat.y*quat.y)*T(2))),
                            std::asin((quat.s*quat.y-quat.x*quat.z)*T(2)),
                            std::atan((quat.s*quat.z+quat.y*quat.x)*T(2)/(T(1)-(quat.z*quat.z+quat.y*quat.y)*T(2))));
}

template<typename T>
Quaternion<T> convertToAnglesAndAxis(const Quaternion<T>& quat)
{
    return Quaternion<T>(   std::acos(quat.s)*T(2),
                            Vector<T,3>(quat.x,quat.y,quat.z)/std::sqrt(T(1)-quat.s*quat.s));
}

template<typename T>
Quaternion<T> slerp(const Quaternion<T>& x, const Quaternion<T>& y, const T& t) {
    auto mix = [&t](const T& a, const T& b) {
        return a + t * (b - a);
    };

    const T cosTheta = x.s * y.s + x.x * y.x + x.y * y.y + x.z * y.z;
    Quaternion<T> z = cosTheta < 0.0f ? (-1.0f) * y : y;

    if (cosTheta > static_cast<T>(1) - std::numeric_limits<T>::epsilon()) {
        return normalize(Quaternion<T>(mix(x.s, z.s), mix(x.x, z.x), mix(x.y, z.y), mix(x.z, z.z)));
    }

    const T angle = std::acos(cosTheta);
    return normalize(std::sin((static_cast<T>(1) - t) * angle) / std::sin(angle) * x + std::sin(t * angle) / std::sin(angle) * z);
}

template<typename type>
Matrix<type,4,4> rotate(Quaternion<type> qu){
    return convert4x4(qu);
}

extern template class Quaternion<float>;
extern template class Quaternion<double>;

}
#endif // QUATERNION_H
