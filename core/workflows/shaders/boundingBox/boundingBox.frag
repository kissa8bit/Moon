#version 450

#include "../__methods__/defines.glsl"

layout(set = 1, binding = 0) uniform sampler2D depthSampler;

layout(location = 0) out vec4 outColor;

void main()
{
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(depthSampler, 0));
    float sceneDepth = texture(depthSampler, uv).r;
    if (gl_FragCoord.z > sceneDepth) discard;
    outColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
}