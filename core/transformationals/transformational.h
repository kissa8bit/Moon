#ifndef MOON_TRANSFORMATIONALS_TRANSFORMATIONAL_H
#define MOON_TRANSFORMATIONALS_TRANSFORMATIONAL_H

#include <math/linearAlgebra.h>

namespace moon::transformational {

class Transformational
{
public:
    virtual ~Transformational(){};

    virtual Transformational& setGlobalTransform(const math::mat4& transform) = 0;
    virtual Transformational& translate(const math::vec3& translate) = 0;
    virtual Transformational& rotate(const math::quat& rot) = 0;
    virtual Transformational& rotate(const float & ang, const math::vec3& ax) = 0;
    virtual Transformational& scale(const math::vec3& scale) = 0;
    virtual Transformational& update() = 0;
};

} // moon::transformational

#define DEFAULT_TRANSFORMATIONAL()                                              \
protected:                                                                      \
    moon::math::quat m_translation{ 0.0f,0.0f,0.0f,0.0f };                      \
    moon::math::quat m_rotation{ 1.0f,0.0f,0.0f,0.0f };                         \
    moon::math::vec3 m_scaling{ 1.0f };                                         \
    moon::math::mat4 m_globalTransformation = moon::math::mat4::identity();

#define DEFAULT_TRANSFORMATIONAL_GETTERS()                      \
public:                                                         \
    moon::math::quat& translation();                            \
    moon::math::quat& rotation();                               \
    moon::math::vec3& scaling();                                \
    moon::math::mat4& globalTransformation();

#define DEFAULT_TRANSFORMATIONAL_GETTERS_DEFINITION(Name)                                           \
    moon::math::quat& Name::translation() {return m_translation;}                                   \
    moon::math::quat& Name::rotation() {return m_rotation;}                                         \
    moon::math::vec3& Name::scaling() {return m_scaling;}                                           \
    moon::math::mat4& Name::globalTransformation() {return m_globalTransformation;}

#define DEFAULT_TRANSFORMATIONAL_OVERRIDE(Name)                                                     \
public:                                                                                             \
    Name& setGlobalTransform(const moon::math::mat4& transform) override;                           \
    Name& translate(const moon::math::vec3& translate) override;                                    \
    Name& rotate(const float& ang, const moon::math::vec3& ax) override;                            \
    Name& scale(const moon::math::vec3& scale) override;                                            \
    Name& rotate(const moon::math::quat& rot) override;                                             \
    Name& update() override;

#define DEFAULT_TRANSFORMATIONAL_DEFINITION(Name)                                                   \
    Name& Name::setGlobalTransform(const moon::math::mat4& transform) {                             \
        m_globalTransformation = transform;                                                         \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::translate(const moon::math::vec3& translate) {                                      \
        m_translation += moon::math::quat(0.0f, translate);                                         \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::rotate(const float& ang, const moon::math::vec3& ax) {                              \
        m_rotation = convert(ang, moon::math::vec3(normalized(ax))) * m_rotation;                   \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::scale(const moon::math::vec3& scale) {                                              \
        m_scaling = scale;                                                                          \
        return update();                                                                            \
    }                                                                                               \
    Name& Name::rotate(const moon::math::quat& rot) {                                               \
        m_rotation = rot * m_rotation;                                                              \
        return update();                                                                            \
    }

#define DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DECL(Name)   \
    Name& rotateX(const float& ang);                    \
    Name& rotateY(const float& ang);

#define DEFAULT_TRANSFORMATIONAL_ROTATE_XY_DEF(Name)                                                \
    Name& Name::rotateX(const float& ang) {                                                         \
        const moon::math::vec3 ax(1.0f, 0.0f, 0.0f);                                                \
        m_rotation = m_rotation * convert(ang, moon::math::vec3(normalized(ax)));                   \
        return update();                                                                            \
    }                                                                                               \
                                                                                                    \
    Name& Name::rotateY(const float& ang) {                                                         \
        const moon::math::vec3 ax(0.0f, 0.0f, 1.0f);                                                \
        m_rotation = convert(ang, moon::math::vec3(normalized(ax))) * m_rotation;                   \
        return update();                                                                            \
    }

#endif // MOON_TRANSFORMATIONALS_TRANSFORMATIONAL_H
