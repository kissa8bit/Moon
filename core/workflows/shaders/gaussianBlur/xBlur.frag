#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/math.glsl"

layout(set = 0, binding = 0) uniform sampler2D blurSampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;

layout(push_constant) uniform PC {
    float depth;
} pc;

vec4 blur(sampler2D blurSampler, vec2 TexCoord) {
    float Nf = (texture(blurSampler, TexCoord).a - pc.depth) / (1.0f - pc.depth);
    Nf = 10.0f * clamp(Nf, 0.0f, 1.0f);
    uint N = uint(Nf);
    uint n = 2 * N;
    float sum = power(2, n);

    vec2 texOffset = 1.0 / textureSize(blurSampler, 0);
    vec3 result = texture(blurSampler, TexCoord).rgb * C(n, N) / sum;

    for(int i = 1; i < N + 1; ++i) {
        float x = texOffset.x * i;
        float weight = C(n, N - i) / sum;
        result += texture(blurSampler, TexCoord + vec2(x, 0.0)).rgb * weight;
        result += texture(blurSampler, TexCoord - vec2(x, 0.0)).rgb * weight;
    }

    return vec4(result, 1.0);
}

void main() {
    outColor = vec4(0.0);
    outBlur = blur(blurSampler, fragTexCoord);
}
