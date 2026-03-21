#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
    vec2 viewport;
} global;

layout(set = 1, binding = 0) uniform sampler2D normal;
layout(set = 1, binding = 1) uniform sampler2D color;
layout(set = 1, binding = 2) uniform sampler2D depth;

layout(push_constant) uniform PC {
    int   kernelSize;
    float radius;
    float aoFactor;
    float aoPower;
} pc;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out float outColor;

const float BIAS = 0.025;

// Per-pixel pseudo-random value in [0, 1)
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float SSAO() {
    if(pc.aoFactor <= 0.0f){
        return 1.0f;
    }

    vec3 worldNormal = decodeSphericalNormal(texture(normal, fragTexCoord).xy);

    if (checkZeroNormal(worldNormal)) return 1.0;

    vec3 worldPos = reconstructPosition(global.invViewProj, fragTexCoord, texture(depth, fragTexCoord).x);

    // Transform surface to view space
    vec3 viewPos    = vec3(global.view * vec4(worldPos, 1.0));
    vec3 viewNormal = normalize(mat3(global.view) * worldNormal);

    // Per-pixel random rotation around view-space normal
    float angle    = hash(fragTexCoord * vec2(textureSize(normal, 0)) / 4.0) * 2.0 * pi;
    vec3 randomVec = vec3(cos(angle), sin(angle), 0.0);

    // Gram–Schmidt TBN aligned to the view-space normal
    vec3 tangent   = normalize(randomVec - viewNormal * dot(randomVec, viewNormal));
    vec3 bitangent = cross(viewNormal, tangent);
    mat3 TBN       = mat3(tangent, bitangent, viewNormal);

    float occlusion = 0.0;

    for (int i = 0; i < pc.kernelSize; i++) {
        // Spherical Fibonacci hemisphere — uniform, low-discrepancy distribution
        float fi    = float(i);
        float theta = 2.0 * pi * fract(fi * 0.618033988749895);
        float v     = (fi + 0.5) / float(pc.kernelSize);
        vec3 dir    = vec3(sqrt(v) * cos(theta), sqrt(v) * sin(theta), sqrt(1.0 - v));

        // Quadratic falloff: cluster samples near the surface
        float t     = fi / float(pc.kernelSize);
        float scale = mix(0.1, 1.0, t * t);
        dir *= scale;

        // Sample position in view space
        vec3 samplePos = viewPos + TBN * dir * pc.radius;

        // Project to screen UV
        vec4 proj4    = global.proj * vec4(samplePos, 1.0);
        vec2 sampleUV = proj4.xy / proj4.w * 0.5 + 0.5;

        if (!isInside(sampleUV)) continue;

        // Actual geometry depth at the projected UV — reconstruct world pos, then to view space
        vec3 geomWorldPos = reconstructPosition(global.invViewProj, sampleUV, texture(depth, sampleUV).x);
        vec3 geomViewPos = vec3(global.view * vec4(geomWorldPos, 1.0));

        // Geometry closer to camera (less negative Z) than the sample → occlusion
        // Range check prevents distant geometry from contributing
        float rangeCheck = smoothstep(0.0, 1.0, pc.radius / abs(viewPos.z - geomViewPos.z));
        occlusion += (geomViewPos.z >= samplePos.z + BIAS ? 1.0 : 0.0) * rangeCheck;
    }

    float raw = 1.0 - occlusion / float(pc.kernelSize);
    return clamp(pc.aoFactor * pow(raw, pc.aoPower), 0.0, 1.0);
}

void main() {
    const vec4 rgba = texture(color, fragTexCoord).rgba;
    if(dot(rgba, rgba) == 0.0f){
        outColor = 1.0f;
        return;
    }
    outColor = SSAO();
}
