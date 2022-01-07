#ifndef PBR
#define PBR

#include "colorFunctions.glsl"

float geometricOcclusion(float NdotL, float NdotV, float k) {
    float attenuationL = NdotL / (k + (1.0f - k) * (NdotL));
    float attenuationV = NdotV / (k + (1.0f - k) * (NdotV));
    return attenuationL * attenuationV;
}
float microfacetDistribution(float NdotH, float alphaRoughness) {
    float roughnessSq = alphaRoughness * alphaRoughness;
    float f = NdotH * NdotH * (roughnessSq - 1.0f) + 1.0f;
    return roughnessSq / (pi * f * f);
}

vec3 diffuse(const in vec4 BaseColor, const in float metallic, const in vec3 f0) {
    vec3 diffuseColor = BaseColor.rgb * (vec3(1.0f) - f0);
    diffuseColor *= 1.0f - metallic;

    return diffuseColor / pi;
}

vec3 specularReflection(vec3 specular0, vec3 specular90, float NdotH) {
    return specular0 + (specular90 - specular0) * pow(1.0f - NdotH, 5);
}

vec4 pbr(
    vec4 position,
    vec4 normal,
    vec4 baseColorTexture,
    vec4 eyePosition,
    vec4 lightColor,
    vec3 lightPosition
) {
    vec3 Direction      = normalize(eyePosition.xyz - position.xyz);
    vec3 LightDirection = normalize(lightPosition - position.xyz);
    vec3 Normal         = normal.xyz;
    vec3 H              = normalize(Direction + LightDirection);
    vec4 BaseColor      = baseColorTexture;

    float perceptualRoughness = decodeParameter(0x000000ff, 0, position.a) / 255.0f;
    float metallic = decodeParameter(0x0000ff00, 8, position.a) / 255.0f;

    float alphaRoughness = perceptualRoughness * perceptualRoughness;
    vec3 f0 = vec3(0.04f);
    vec3 specularColor = mix(f0, BaseColor.rgb, metallic);

	float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
	float reflectance90 = clamp(reflectance * 25.0f, 0.0, 1.0);
	vec3 specularEnvironmentR0 = specularColor.rgb;
	vec3 specularEnvironmentR90 = vec3(1.0f, 1.0f, 1.0f) * reflectance90;

    vec3 F = specularReflection(specularEnvironmentR0, specularEnvironmentR90, clamp(dot(H, Direction), 0.0f, 1.0f));
    float G = geometricOcclusion(clamp(dot(Normal, LightDirection), 0.001f, 1.0f), clamp(abs(dot(Normal, Direction)), 0.001f, 1.0f), alphaRoughness);
    float D = microfacetDistribution(clamp(dot(Normal, H), 0.0f, 1.0f), alphaRoughness);

    vec3 diffuseContrib = (1.0f - F) * diffuse(BaseColor, metallic, f0);
    vec3 specContrib = F * G * D / (4.0f * clamp(dot(Normal, LightDirection), 0.001f, 1.0f) * clamp(abs(dot(Normal, Direction)), 0.001f, 1.0f));

    vec4 outColor = vec4(clamp(dot(Normal, LightDirection), 0.001f, 1.0f) * lightColor.xyz * (diffuseContrib + specContrib), BaseColor.a);

    return outColor;
}

#endif
