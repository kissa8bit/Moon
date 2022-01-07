#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(constant_id = 0) const int transparentLayersCount = 1;

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
} global;

layout(set = 0, binding = 1) uniform sampler2D Sampler;
layout(set = 0, binding = 2) uniform sampler2D bloomSampler;
layout(set = 0, binding = 3) uniform sampler2D position;
layout(set = 0, binding = 4) uniform sampler2D normal;
layout(set = 0, binding = 5) uniform sampler2D depth;
layout(set = 0, binding = 6) uniform sampler2D layersSampler[transparentLayersCount];
layout(set = 0, binding = 7) uniform sampler2D layersBloomSampler[transparentLayersCount];
layout(set = 0, binding = 8) uniform sampler2D layersPosition[transparentLayersCount];
layout(set = 0, binding = 9) uniform sampler2D layersNormal[transparentLayersCount];
layout(set = 0, binding = 10) uniform sampler2D layersDepth[transparentLayersCount];
layout(set = 0, binding = 11) uniform sampler2D skybox;
layout(set = 0, binding = 12) uniform sampler2D skyboxBloom;
layout(set = 0, binding = 13) uniform sampler2D scattering;
layout(set = 0, binding = 14) uniform sampler2D sslrSampler;

layout(push_constant) uniform PC {
    int enableScatteringRefraction;
    int enableTransparentLayers;
    float blurDepth;
} pc;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;
layout(location = 2) out vec4 outBlur;

mat4 projview = global.proj * global.view;
vec3 eyePosition = viewPosition(global.view);

float h = 0.5f;
float nbegin = 1.33f;
float nend = nbegin + 1.5f;
float reflectionProbability = 0.04f;

vec3 findRefrCoords(const in vec3 startPos, const in vec3 layerPointPosition, const in vec3 layerPointNormal, float n) {
    vec3 beamDirection = normalize(layerPointPosition - startPos);
    float cosAlpha = -dot(layerPointNormal, beamDirection);
    float sinAlpha = sqrt(1.0 - cosAlpha * cosAlpha);

    float deviation = h * sinAlpha * (1.0 - cosAlpha / sqrt(n * n - sinAlpha * sinAlpha));
    vec3 direction = -normalize(layerPointNormal + beamDirection * cosAlpha);
    vec4 position = projview * vec4(layerPointPosition + deviation * direction, 1.0);

    return vec3(position.xy / position.w * 0.5 + 0.5, position.z / position.w);
}

vec3 layerPointPosition(const in int i, const in vec2 coord) {
    return texture(layersPosition[i], coord).xyz;
}

vec3 layerPointNormal(const in int i, const in vec2 coord) {
    return normalize(texture(layersNormal[i], coord).xyz);
}

float layerDepth(const in int i, const in vec2 coord) {
    return texture(layersDepth[i], coord).r;
}

bool insideCond(const in vec2 coords) {
    return (coords.x <= 1.0) && (coords.y <= 1.0) && (coords.x >= 0.0) && (coords.y >= 0.0);
}

bool depthCond(float z, vec2 coords) {
    return z <= texture(depth, coords.xy).r;
}

void findRefr(const int i, const float n, inout vec3 startPos, inout vec3 coords) {
    if(insideCond(coords.xy) && depthCond(layerDepth(i, coords.xy), coords.xy)) {
        vec3 start = startPos;
        startPos = layerPointPosition(i, coords.xy);
        if(layerDepth(i, coords.xy) != 1.0) {
            coords = findRefrCoords(start, layerPointPosition(i, coords.xy), layerPointNormal(i, coords.xy), n);
        }
    }
}

vec4 findColor(const in vec3 coord, sampler2D Sampler, sampler2D skybox, bool enableSkybox) {
    vec4 skyboxColor = enableSkybox && texture(depth, coord.xy).r == 1.0 ? texture(skybox, coord.xy) : vec4(0.0);
    return (insideCond(coord.xy) ? texture(Sampler, coord.xy) + skyboxColor : vec4(0.0));
}

vec4 accumulateColor(vec3 beginCoords, vec3 endCoords, float step, sampler2D Sampler, sampler2D Depth, sampler2D skybox, bool enableSkybox, bool enableScattering) {
    vec4 color = vec4(0.0);
    vec4 scat = vec4(0.0);
    for(float t = 0.0; t < 1.0; t += step) {
        vec3 coords = beginCoords + (endCoords - beginCoords) * t;
        vec4 factor = vec4(4.0 * abs(t - 0.5) - 2.0 / 3.0, 1.0 - abs(2.0 * t - 2.0 / 3.0), 1.0 - abs(2.0 * t - 4.0 / 3.0), 1.0);
        float layDepth = texture(Depth, coords.xy).r;
        if(depthCond(layDepth, coords.xy) && coords.z <= layDepth) {
            factor = (beginCoords != vec3(fragTexCoord, 0.0) ? factor : vec4(1.0));
            color += factor * findColor(coords, Sampler, skybox, enableSkybox);
            if(enableScattering && layDepth > texture(scattering, coords.xy).a) {
                scat += factor * vec4(texture(scattering, coords.xy).xyz, 0.0);
            }
        }
    }
    return color + scat;
}

void transparentLayersCombine() {
    bool transparentFrags = (texture(layersSampler[0], fragTexCoord.xy).a != 0.0);
    bool enableScatteringRefraction = pc.enableScatteringRefraction == 1;

    float step = transparentFrags ? 0.02 : 1.0;
    float incrementStep = 2.0;

    vec3 beginCoords = vec3(fragTexCoord, 0.0), beginStartPos = eyePosition;
    vec3 endCoords = vec3(fragTexCoord, 0.0), endStartPos = eyePosition;
    vec4 layerColor = vec4(0.0), layerBloom = vec4(0.0);

    for(int i = 0; i < (transparentFrags ? transparentLayersCount : 0); i++) {
        float layerStep = step * (pow(incrementStep, i));
        vec4 color = (i == 0 ? reflectionProbability : 1.0) * accumulateColor(beginCoords, endCoords, layerStep, layersSampler[i], layersDepth[i], skybox, false, enableScatteringRefraction);
        layerColor = max(layerColor, 2.0 * layerStep * color);

        vec4 bloom = (i == 0 ? reflectionProbability : 1.0) * accumulateColor(beginCoords, endCoords, layerStep, layersBloomSampler[i], layersDepth[i], skybox, false, false);
        layerBloom = max(layerBloom, 2.0 * layerStep * bloom);

        findRefr(i, nbegin, beginStartPos, beginCoords);
        findRefr(i, nend, endStartPos, endCoords);
    }

    float depth0 = texture(layersDepth[0], fragTexCoord.xy).r;
    bool layerBehindScattering = transparentFrags && depth0 > texture(scattering, fragTexCoord.xy).a;
    vec4 frontScatteringColor = layerBehindScattering || !enableScatteringRefraction ? vec4(texture(scattering, fragTexCoord.xy).xyz, 0.0) : vec4(0.0);

    outColor = accumulateColor(beginCoords, endCoords, step, Sampler, depth, skybox, true, enableScatteringRefraction);
    outColor = max(layerColor, max(step * outColor, frontScatteringColor));
    outColor += max(outColor, texture(sslrSampler,fragTexCoord));

    outBloom = accumulateColor(beginCoords, endCoords, step, bloomSampler, depth, skyboxBloom, true, false);
    outBloom = max(layerBloom, step * outBloom);

    float d = transparentFrags ? depth0 : texture(depth, fragTexCoord.xy).r;
    outBlur = vec4(outColor.xyz, d);
    if(d > pc.blurDepth){
        outColor = vec4(0.0f);
    }
}

void main() {
    if(pc.enableTransparentLayers == 0){
        float d = texture(depth, fragTexCoord.xy).r;

        vec4 scatteringColor = vec4(texture(scattering, fragTexCoord.xy).xyz, 1.0);
        outColor = texture(Sampler, fragTexCoord);
        outBloom = texture(bloomSampler, fragTexCoord);
        outColor += d == 1.0 ? texture(skybox, fragTexCoord.xy) : vec4(0.0);
        outBloom += d == 1.0 ? texture(skyboxBloom, fragTexCoord.xy) : vec4(0.0);
        outColor += scatteringColor;
        outColor += max(outColor, texture(sslrSampler,fragTexCoord));
        outBlur = vec4(outColor.xyz, d);
        if(d > pc.blurDepth){
            outColor = vec4(0.0f);
        }

        return;
    }
    transparentLayersCombine();
}