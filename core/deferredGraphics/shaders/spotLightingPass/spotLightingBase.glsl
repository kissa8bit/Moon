#include "../../../workflows/shaders/__methods__/lightDrop.glsl"
#include "../../../workflows/shaders/__methods__/pbr.glsl"
#include "../../../workflows/shaders/__methods__/shadow.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"
#include "../../../workflows/shaders/__methods__/defines.glsl"

#ifndef SPOT_LIGHTING_BASE
#define SPOT_LIGHTING_BASE

layout(set = 2, binding = 0) uniform LightBufferObject {
    mat4 proj;
    mat4 view;
    vec4 color;
    vec4 prop;
} light;

vec4 calcLight(
    const in vec4 position,
    const in vec4 normal,
    const in vec4 color,
    const in vec4 eyePosition,
    sampler2D shadowMap,
    sampler2D lightTexture,
    uint type) 
{
    if(checkZeroNormal(normal.xyz)) {
        return vec4(0.0f);
    }

    const vec4 localPosition = light.view * vec4(position.xyz, 1.0);
    if(outsideSpot(light.proj, type, localPosition)){
        return vec4(0.0f);
    }   

    const vec4 projPosition = light.proj * localPosition;
    const vec3 ndcCoords = vec3(projPosition.xy / projPosition.w * 0.5 + 0.5, projPosition.z / projPosition.w);
    const float shadowFactor = calcShadowFactor(light.proj, shadowMap, ndcCoords);

    const vec4 lightTextureColor = texture(lightTexture, ndcCoords.xy);
    const vec4 lightColor = vec4(max(light.color, lightTextureColor).xyz, 1.0);
    
    vec3 lightPosition = viewPosition(light.view);
    const vec4 pbrColor = pbr(position, normal, color, eyePosition, lightColor, lightPosition);

    const float distribusion = lightDistribusion(position.xyz, lightPosition, light.proj, light.view, getDirection(light.view), light.prop.x, light.prop.y, type);

    const float lightDropFactor = light.prop.w;
    const float lightDrop = lightDropFactor * lightDrop(max(length(lightPosition - position.xyz), 0.01));

    const float lightPowerFactor = light.prop.z;
    const float lightPower = shadowFactor * lightPowerFactor * distribusion / (lightDrop > 0.0 ? lightDrop : 1.0);

    return vec4(lightPower * pbrColor.xyz, pbrColor.a);
}

#endif