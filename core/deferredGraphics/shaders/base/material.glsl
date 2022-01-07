#ifndef BASE_MATERIAL
#define BASE_MATERIAL

struct Material{
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 diffuseFactor;
    vec4 specularFactor;
    float workflow;
    int baseColorTextureSet;
    int physicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float metallicFactor;
    float roughnessFactor;
    float alphaMask;
    float alphaMaskCutoff;
    int number;
};

#endif