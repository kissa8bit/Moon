#ifndef LIGHT_DROP_GLSL
#define LIGHT_DROP_GLSL

#include "defines.glsl"

float lightDrop(const float distance) {
    return pow(distance, 2);
}

// Both types project to NDC space so aspect ratio is handled correctly.
// innerFraction : [0, 1] — inner radius/edge with full brightness (0 = no inner zone, 1 = full cone)
// exponent      : falloff steepness in the transition zone (4 = original behavior)
// type          : SPOT_LIGHTING_TYPE_CIRCLE — length(uv), elliptical iso-contours matching the frustum
//                 SPOT_LIGHTING_TYPE_SQUARE — max(|u|,|v|), rectangular iso-contours
float lightDistribusion(const in vec3 position, const in mat4 lightProjMatrix, const in mat4 lightViewMatrix, float innerFraction, float exponent, uint type) {
    vec4 projPos = lightProjMatrix * lightViewMatrix * vec4(position, 1.0);
    vec2 uv = projPos.xy / projPos.w;
    float dist = (type == SPOT_LIGHTING_TYPE_CIRCLE) ? length(uv) : max(abs(uv.x), abs(uv.y));
    float t = clamp((dist - innerFraction) / max(1.0 - innerFraction, 1e-5), 0.0, 1.0);
    return pow(cos(pi * 0.5 * t), max(exponent, 0.01));
}

#endif
