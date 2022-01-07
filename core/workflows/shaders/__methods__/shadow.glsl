#ifndef SHADOW
#define SHADOW

#include "defines.glsl"

float getDepthFromMap(const in mat4 proj, sampler2D Sampler, vec2 uv){
    return proj[3][2] / (texture(Sampler, uv.xy).x * proj[2][3] - proj[2][2]);
}

bool inShadow(sampler2D Sampler, vec2 uv, float pointDepth) {
    return texture(Sampler, uv.xy).x - pointDepth < 0.0f;
}

bool inLight(sampler2D Sampler, vec2 uv, float pointDepth) {
    return texture(Sampler, uv.xy).x - pointDepth >= -0.00001f;
}

vec2 center(const in vec2 left, const in vec2 right){
    return (right + left) / 2.0f;
}

float shadowBlur(sampler2D Sampler, vec4 coordinates, vec2 delta) {
    int pix_N = 10;
    int it_N = 4;

    vec2 left = (coordinates.xy / coordinates.w) * 0.5 + 0.5;
    vec2 right = left + pix_N * delta;

    float pointDepth = coordinates.z / coordinates.w;
    if(inLight(Sampler, left, pointDepth)){
        return 1.0f;
    } else if(!inLight(Sampler, right, pointDepth)){
        return 0.0f;
    }

    vec2 c = vec2(0.0f);
    for(vec2 l = left, r = right; it_N >= 0; it_N--){
        c = center(l, r);
        if(inLight(Sampler, c, pointDepth)){
            r = c;
        } else {
            l = c;
        }
    }

    return length(c - right) / length(left - right);
}

float shadowFactor(const in mat4 proj, sampler2D Sampler, vec4 coordinates) {
    vec2 delta = vec2(1.0f) / textureSize(Sampler, 0);

    float res = 0.0f;
    float phi_0 = 0.0f;
    float phi_n = 2.0f * pi;
    int it_N = 32;
    float valid = 0.0f;
    for(int i = 0; i < it_N; i++){
        float phi = float(i) * (phi_n - phi_0) / it_N;
        vec2 dir = delta * vec2(cos(phi), sin(phi));
        float plus = shadowBlur(Sampler, coordinates, dir);
        float minus = shadowBlur(Sampler, coordinates, -dir);
        float s = max(plus, minus);
        if(s != 0.0f){
            res += s;
            valid += 1.0f;
        }
    }
    return res / valid;
}

#endif