#include "../../../workflows/shaders/__methods__/lightDrop.glsl"
#include "../../../workflows/shaders/__methods__/pbr.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"
#include "../../../workflows/shaders/__methods__/defines.glsl"

#ifndef POINT_LIGHTING_BASE
#define POINT_LIGHTING_BASE

layout(set = 2, binding = 0) uniform LightBufferObject {
    vec4 position;
    vec4 color;
    vec4 prop;  // x=radius, y=powerFactor, z=dropFactor, w=unused
} light;

vec4 calcPointLight(
    const in vec4 position,
    const in vec4 normal,
    const in vec4 color,
    const in vec4 eyePosition)
{
    if(checkZeroNormal(normal.xyz)) {
        return vec4(0.0f);
    }

    const float dist = length(position.xyz - light.position.xyz);
    const float t = clamp(dist / max(light.prop.x, 0.001), 0.0, 1.0);
    const float window = pow(1.0 - t * t, 2.0);
    if(window <= 0.0) {
        return vec4(0.0f);
    }

    const vec3 lightPosition = light.position.xyz;
    const vec4 pbrColor = pbr(position, normal, color, eyePosition, light.color, lightPosition);

    const float lightDropFactor = light.prop.z;
    const float lightDrop = lightDrop(max(lightDropFactor, 0.01) * max(dist, 0.01));

    const float lightPowerFactor = light.prop.y;
    const float lightPower = lightPowerFactor * window / lightDrop;

    return vec4(lightPower * pbrColor.xyz, pbrColor.a);
}

#endif
