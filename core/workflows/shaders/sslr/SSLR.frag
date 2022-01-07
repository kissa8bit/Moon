#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/pbr.glsl"
#include "../__methods__/colorFunctions.glsl"
#include "../__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
} global;
layout(set = 0, binding = 1) uniform sampler2D position;
layout(set = 0, binding = 2) uniform sampler2D normal;
layout(set = 0, binding = 3) uniform sampler2D Sampler;
layout(set = 0, binding = 4) uniform sampler2D depth;
layout(set = 0, binding = 5) uniform sampler2D layerPosition;
layout(set = 0, binding = 6) uniform sampler2D layerNormal;
layout(set = 0, binding = 7) uniform sampler2D layerSampler;
layout(set = 0, binding = 8) uniform sampler2D layerDepth;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

vec4 p0 = vec4(viewPosition(global.view), 1.0);
mat4 projview = global.proj * global.view;

vec2 findIncrement(const in vec4 position, const in vec4 direction) {
    vec4 start = positionProj(projview, position);
    vec4 end = positionProj(projview, position + direction);
    vec2 planeDir = normalize(end.xy - start.xy);
    return planeDir;
}

bool isTransparent(const in vec2 coords){
    return texture(depth, coords).r > texture(layerDepth, coords).r;
}

vec4 findColor(const in vec2 coords) {
    return isTransparent(coords) ? texture(layerSampler, coords) : texture(Sampler, coords);
}

vec4 findPosition(const in vec2 coords) {
    return isTransparent(coords) ? vec4(texture(layerPosition, coords).xyz, 1.0) : vec4(texture(position, coords).xyz, 1.0);
}

vec4 findNormal(const in vec2 coords) {
    return isTransparent(coords) ? vec4(texture(layerNormal, coords).xyz, 0.0) : vec4(texture(normal, coords).xyz, 0.0);
}

float findMaterial(const in vec2 coords) {
    return (isTransparent(coords) ? texture(layerPosition, coords) : texture(position, coords)).a;
}

vec3 u =   normalize(vec3(global.view[0][0],global.view[1][0],global.view[2][0]));
vec3 v =   normalize(vec3(global.view[0][1],global.view[1][1],global.view[2][1]));
vec3 n = - normalize(vec3(global.view[0][2],global.view[1][2],global.view[2][2]));

float h = - 1.0f / global.proj[1][1];
float w = - global.proj[1][1] / global.proj[0][0] * h;

vec4 getDir(vec2 planeCoords){
    planeCoords = 2.0f * planeCoords - 1.0f;
    return normalize(vec4(n + u * w * planeCoords.x - v * h * planeCoords.y, 0.0f));
}

vec4 calcPos(vec4 p, vec4 p0, vec4 r, vec2 planeCoords) {
    vec4 d_i = getDir(planeCoords);
    float t = linesIntersection(p0.xyz, d_i.xyz, p.xyz, r.xyz);
    return p0 + t * d_i;
}

float findPosDis(vec4 p, vec4 p0, vec4 r, vec2 planeCoords){
    bool transparentFrags = isTransparent(planeCoords);
    vec4 i_pos = calcPos(p, p0, r, planeCoords);
    vec4 r_pos = findPosition(planeCoords);
    return zProj(projview, i_pos - r_pos);
}

vec4 SSLR(const in vec4 p, const in vec4 n, const in vec4 d, const in vec4 r, vec2 planeCoords, vec2 increment, int steps) {
    bool binarySearch = false;
    float d_pos = 0.0f;
    for(int i = 0; i < steps; i++) {
        planeCoords += increment;
        d_pos = findPosDis(p, p0, r, planeCoords);
        binarySearch = d_pos > 0.0f;
        if(binarySearch || !isInside(planeCoords)){
            break;
        }
    }
    if(!binarySearch){
        return vec4(0.0);
    }
    for(int j = 0; j < 8; j++){
        int sign = d_pos > 0.0f ? -1 : 1;
        planeCoords += sign * (increment /= 2);
        d_pos = findPosDis(p, p0, r, planeCoords);
    }
    if(d_pos * d_pos < 0.005f){
        float fresnel = 0.0f + 2.8f * pow(1.0f + dot(d, n), 2);
        return fresnel * findColor(planeCoords);
    }

    return vec4(0.0);
}

void main() {
    outColor = vec4(0.0);

    float material = findMaterial(fragTexCoord);
    float roughness = decodeParameter(0x000000ff, 0, material) / 255.0f;
    float s = (1.0f - roughness) * (1.0f - roughness);

    int steps = 50;
    float incrementFactor = 0.5f / steps;

    vec4 p = findPosition(fragTexCoord);
    vec4 n = findNormal(fragTexCoord);
    vec4 d = normalize(p - p0);
    vec4 r = normalize(reflect(d,n));

    vec2 increment = incrementFactor * findIncrement(p, r);

    outColor += s * SSLR(p, n, d, r, fragTexCoord, increment, steps);
}
