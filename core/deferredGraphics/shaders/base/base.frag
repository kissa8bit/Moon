#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/colorFunctions.glsl"
#include "../../../workflows/shaders/__methods__/math.glsl"
#include "material.glsl"

layout(constant_id = 0) const int layerIndex = 0;

layout(set = 0, binding = 1) uniform samplerCube samplerCubeMap;
layout(set = 0, binding = 2) uniform sampler2D prevLayerDepthMap;
layout(set = 0, binding = 3) uniform sampler2D prevLayerColorMap;

layout (push_constant) uniform PC{
    Material material;
} pc;

layout(set = 3, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 3, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 3, binding = 2) uniform sampler2D normalTexture;
layout(set = 3, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 3, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0)	in vec4 position;
layout(location = 1)	in vec3 normal;
layout(location = 2)	in vec2 UV0;
layout(location = 3)	in vec2 UV1;
layout(location = 4)	in vec3 tangent;
layout(location = 5)	in vec3 bitangent;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
} local;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outBaseColor;
layout(location = 3) out vec4 outEmissiveColor;

vec3 getNormal() {
    vec3 tangentNormal = normalize(texture(normalTexture, pc.material.normalTextureSet == 0 ? UV0 : UV1).xyz * 2.0 - 1.0);
    mat3 TBN = mat3(tangent, bitangent, normal);
    return normalize(TBN * tangentNormal);
}

float convertMetallic(vec3 diffuse, vec3 specular, float maxSpecular) {
    float perceivedDiffuse = sqrt(0.299 * diffuse.r * diffuse.r + 0.587 * diffuse.g * diffuse.g + 0.114 * diffuse.b * diffuse.b);
    float perceivedSpecular = sqrt(0.299 * specular.r * specular.r + 0.587 * specular.g * specular.g + 0.114 * specular.b * specular.b);
    if(perceivedSpecular < MIN_ROUGHNESS) {
        return 0.0;
    }
    float a = MIN_ROUGHNESS;
    float b = perceivedDiffuse * (1.0 - maxSpecular) / (1.0 - MIN_ROUGHNESS) + perceivedSpecular - 2.0 * MIN_ROUGHNESS;
    float c = MIN_ROUGHNESS - perceivedSpecular;
    float D = max(b * b - 4.0 * a * c, 0.0);
    return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

void metallicRoughnessWorkflow(inout float perceptualRoughness, inout float metallic, inout vec4 baseColor) {
    if (pc.material.physicalDescriptorTextureSet > -1) {
        vec4 mrSample = texture(metallicRoughnessTexture, UV0); // r - (optional) occlusion map, g - roughness, b - metallic
        perceptualRoughness = clamp(mrSample.g * pc.material.roughnessFactor, 0.0, 1.0);
        metallic = clamp(mrSample.b * pc.material.metallicFactor, 0.0, 1.0);
    } else {
        perceptualRoughness = clamp(pc.material.roughnessFactor, MIN_ROUGHNESS, 1.0);
        metallic = clamp(pc.material.metallicFactor, 0.0, 1.0);
    }

    baseColor = pc.material.baseColorFactor * vec4(pc.material.baseColorTextureSet > -1 ? baseColor.xyz : vec3(1.0f), baseColor.a);
}

void specularGlosinessWorkflow(inout float perceptualRoughness, inout float metallic, inout vec4 baseColor) {
    vec4 diffuse = baseColor;
    vec4 sgSample = texture(metallicRoughnessTexture, UV0);
    vec3 specular = sgSample.rgb;
    float maxSpecular = max(max(specular.r, specular.g), specular.b);

    perceptualRoughness = (pc.material.physicalDescriptorTextureSet > -1) ? (1.0 - sgSample.a) : 0.0;
    metallic = convertMetallic(diffuse.rgb, specular, maxSpecular);

    const float epsilon = 1e-6;

    vec3 baseColorDiffusePart = diffuse.rgb * ((1.0 - maxSpecular) / (1 - MIN_ROUGHNESS) / max(1 - metallic, epsilon)) * pc.material.diffuseFactor.rgb;
    vec3 baseColorSpecularPart = specular - (vec3(MIN_ROUGHNESS) * (1 - metallic) * (1 / max(metallic, epsilon))) * pc.material.specularFactor.rgb;
    baseColor = vec4(mix(baseColorDiffusePart, baseColorSpecularPart, metallic * metallic), diffuse.a);
}

void main()
{
    vec4 baseColor = vec4(local.colorFactor.xyz, 1.0f) * texture(baseColorTexture, UV0) + local.constColor;
    vec4 bloomColor = vec4(local.bloomFactor.xyz, 1.0f) * texture(emissiveTexture, UV0) + local.bloomColor;

    const float depth = gl_FragCoord.z;
    const vec2 uv = gl_FragCoord.xy / textureSize(prevLayerDepthMap, 0);

    const float prevDepth = layerIndex > 0 ? texture(prevLayerDepthMap, uv).x : 0.0f;
    const vec4 prevColor = layerIndex > 0 ? texture(prevLayerColorMap, uv) : vec4(0.0f);

    if(layerIndex > 0 && (prevColor.a >= 0.999f || depth <= prevDepth + 1e-6)){
        discard;
    }
     
    if(pc.material.alphaMask == 0.0f){
        baseColor.a = 1.0f;
    }
    
    if(pc.material.alphaMask == 1.0f){
        if( baseColor.a < pc.material.alphaMaskCutoff){
            discard;
        }
        baseColor.a = 1.0f;
    }

//    vec3 I = normalize(position.xyz - eyePosition.xyz);
//    vec3 R = reflect(I, outNormal.xyz);
//    vec4 reflection = texture(samplerCubeMap, R);
//    outBaseColor = vec4(max(outBaseColor.r,reflection.r),max(outBaseColor.g,reflection.g),max(outBaseColor.b,reflection.b), outBaseColor.a);

    float perceptualRoughness;
    float metallic;
    switch(uint(pc.material.workflow)) {
        case WORKFLOW_METALLIC_ROUGHNESS: {
            metallicRoughnessWorkflow(perceptualRoughness, metallic, baseColor);
            break;
        }
        case WORKFLOW_SPECULAR_GLOSINESS: {
            specularGlosinessWorkflow(perceptualRoughness, metallic, baseColor);
            break;
        }
    }
    
    float ao = pc.material.occlusionTextureSet > -1 ? texture(occlusionTexture, UV0).r : 1.0f;

    uint perceptualRoughness_u8 = uint(255.0f * perceptualRoughness);
    uint metallic_u8 = uint(255.0f * metallic);
    uint ao_u8 = uint(255.0f * ao);
    uint outlining = uint(0);
    float params = uintBitsToFloat((perceptualRoughness_u8 << 0) | (metallic_u8 << 8) | (ao_u8 << 16) | (outlining << 24));
    
    float number = uintBitsToFloat(pc.material.number);

    outBaseColor = baseColor;
    outEmissiveColor = bloomColor;
    outPosition = vec4(position.xyz, params);
    outNormal = vec4(pc.material.normalTextureSet > -1 ? getNormal() : normal, number);
}
