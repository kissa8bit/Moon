#ifndef BLUR
#define BLUR

#include "../__methods__/math.glsl"

vec4 blur(sampler2D tex, sampler2D depth, const in vec2 texCoord, const in vec2 dir, const in float userDepthCutoff) {
    const bool isUserSetDepth = userDepthCutoff > 0.0;

    const float focusDepth = isUserSetDepth ? userDepthCutoff : texture(depth, vec2(0.5, 0.5)).r;
    const float pixelDepth = texture(depth, texCoord).r;
    const float diff = abs(pixelDepth - focusDepth);

    if(isUserSetDepth){
        if (pixelDepth < focusDepth) {
            return texture(tex, texCoord);
        }
    } else{
        const float focusRange = 0.05;
        if (diff < focusRange) {
            return texture(tex, texCoord);
        }
    }
    
    const uint N = min(uint(100.0f * clamp(diff, 0.0f, 1.0f)), 16u);
    const uint n = 2 * N;
    const float sum = power(2, n);

    const vec2 texOffset = 1.0 / textureSize(tex, 0);
    vec3 result = texture(tex, texCoord).rgb * C(n, N) / sum;

    for(int i = 1; i < int(N) + 1; ++i) {
        float x = texOffset.x * i;
        float y = texOffset.y * i;
        float weight = C(n, N - uint(i)) / sum;
        result += texture(tex, texCoord + dir * vec2(x, y)).rgb * weight;
        result += texture(tex, texCoord - dir * vec2(x, y)).rgb * weight;
    }

    return vec4(result, 1.0);
}

#endif