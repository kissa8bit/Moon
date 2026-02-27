#ifndef LIGHT_DROP_GLSL
#define LIGHT_DROP_GLSL

#include "defines.glsl"

float lightDrop(const float distance) {
    return pow(distance, 2);
}

// innerFraction : [0, 1] — fraction of the cone radius with full brightness (0 = no inner zone, 1 = full cone)
// exponent      : falloff steepness in the transition zone (4 = original behavior)
// type          : SPOT_LIGHTING_TYPE_CIRCLE — angular falloff (circular iso-contours)
//                 SPOT_LIGHTING_TYPE_SQUARE — UV Chebyshev falloff (square iso-contours)
float lightDistribusion(const in vec3 position, const in vec3 lightPosition, const in mat4 lightProjMatrix, const in mat4 lightViewMatrix, const in vec3 lightDirection, float innerFraction, float exponent, uint type) {
    float t;
    if (type == SPOT_LIGHTING_TYPE_CIRCLE) {
        float outerAngle = atan(-1.0 / lightProjMatrix[1][1]);
        float innerAngle = innerFraction * outerAngle;
        float theta = acos(dot(normalize(position - lightPosition), lightDirection));
        t = clamp((theta - innerAngle) / max(outerAngle - innerAngle, 1e-5), 0.0, 1.0);
    } else {
        vec4 projPos = lightProjMatrix * lightViewMatrix * vec4(position, 1.0);
        vec2 uv = projPos.xy / projPos.w;
        float dist = max(abs(uv.x), abs(uv.y));
        t = clamp((dist - innerFraction) / max(1.0 - innerFraction, 1e-5), 0.0, 1.0);
    }
    return pow(cos(pi * 0.5 * t), exponent);
}

#endif
