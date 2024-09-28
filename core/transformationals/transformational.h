#ifndef TRANSFORMATIONAL_H
#define TRANSFORMATIONAL_H

#include "matrix.h"
#include "quaternion.h"

namespace moon::transformational {

class Transformational
{
public:
    virtual ~Transformational(){};

    virtual Transformational& setGlobalTransform(const moon::math::Matrix<float,4,4>& transform) = 0;
    virtual Transformational& translate(const moon::math::Vector<float,3> & translate) = 0;
    virtual Transformational& rotate(const moon::math::Quaternion<float>& rot) = 0;
    virtual Transformational& rotate(const float & ang, const moon::math::Vector<float,3>& ax) = 0;
    virtual Transformational& scale(const moon::math::Vector<float,3>& scale) = 0;
    virtual Transformational& update() = 0;
};

}

#define DEFAULT_TRANSFORMATIONAL()                                              \
protected:                                                                      \
    moon::math::Quaternion<float>     m_translation{ 0.0f,0.0f,0.0f,0.0f };     \
    moon::math::Quaternion<float>     m_rotation{ 1.0f,0.0f,0.0f,0.0f };        \
    moon::math::Vector<float, 3>      m_scaling{ 1.0f };                        \
    moon::math::Matrix<float, 4, 4>   m_globalTransformation{ 1.0f };

#define DEFAULT_TRANSFORMATIONAL_GETTERS()                      \
public:                                                         \
    moon::math::Quaternion<float>   &translation();             \
    moon::math::Quaternion<float>   &rotation();                \
    moon::math::Vector<float, 3>    &scaling();                 \
    moon::math::Matrix<float, 4, 4> &globalTransformation();

#define DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Name)                                           \
    moon::math::Quaternion<float>   &Name::translation() {return m_translation;}                    \
    moon::math::Quaternion<float>   &Name::rotation() {return m_rotation;}                          \
    moon::math::Vector<float, 3>    &Name::scaling() {return m_scaling;}                            \
    moon::math::Matrix<float, 4, 4> &Name::globalTransformation() {return m_globalTransformation;}

#define DEFAULT_TRANSFORMATIONAL_OVERRIDE(Name)                                                     \
public:                                                                                             \
    Name& setGlobalTransform(const moon::math::Matrix<float, 4, 4>& transform) override;            \
    Name& translate(const moon::math::Vector<float, 3>& translate) override;                        \
    Name& rotate(const float& ang, const moon::math::Vector<float, 3>& ax) override;                \
    Name& scale(const moon::math::Vector<float, 3>& scale) override;                                \
    Name& rotate(const moon::math::Quaternion<float>& rot) override;                                \
    Name& update() override;

#define DEFAULT_TRANSFORMATIONAL_DEFINITION(Name)                                                   \
    Name& Name::setGlobalTransform(const moon::math::Matrix<float, 4, 4>& transform) {              \
        m_globalTransformation = transform;                                                         \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::translate(const moon::math::Vector<float, 3>& translate) {                          \
        m_translation += moon::math::Quaternion<float>(0.0f, translate);                            \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::rotate(const float& ang, const moon::math::Vector<float, 3>& ax) {                  \
        m_rotation = convert(ang, moon::math::Vector<float, 3>(normalize(ax))) * m_rotation;        \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::scale(const moon::math::Vector<float, 3>& scale) {                                  \
        m_scaling = scale;                                                                          \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::rotate(const moon::math::Quaternion<float>& rot) {                                  \
        m_rotation = rot * m_rotation;                                                              \
        return update();                                                                            \
    }

#define DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Name)   \
    Name& rotateX(const float& ang);                    \
    Name& rotateY(const float& ang);

#define DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Name)                                                \
    Name& Name::rotateX(const float& ang) {                                                         \
        const moon::math::Vector<float, 3> ax(1.0f, 0.0f, 0.0f);                                    \
        m_rotation = m_rotation * convert(ang, moon::math::Vector<float, 3>(normalize(ax)));        \
        return update();                                                                            \
    }                                                                                               \
                                                                                                    \
    Name& Name::rotateY(const float& ang) {                                                         \
        const moon::math::Vector<float, 3> ax(0.0f, 0.0f, 1.0f);                                    \
        m_rotation = convert(ang, moon::math::Vector<float, 3>(normalize(ax))) * m_rotation;        \
        return update();                                                                            \
    }

#endif // TRANSFORMATIONAL_H
