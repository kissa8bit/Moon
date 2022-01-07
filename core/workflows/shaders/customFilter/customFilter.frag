#version 450

#include "../__methods__/defines.glsl"

layout(set = 0, binding = 0) uniform sampler2D Sampler;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    float dx;
    float dy;
    float blitFactor;
} pc;

void main() {
    outColor = vec4(0.0);

    vec2 textel = 1.0 / textureSize(Sampler, 0);
    float sum = 0.0;
    int steps = 2;

    for(int i = 1; i <= steps; i++) {
        outColor += texture(Sampler, fragTexCoord + i * vec2(-textel.x * pc.dx, -textel.y * pc.dy));
        outColor += texture(Sampler, fragTexCoord + i * vec2(-textel.x * pc.dx, textel.y * pc.dy));
        outColor += texture(Sampler, fragTexCoord + i * vec2(textel.x * pc.dx, -textel.y * pc.dy));
        outColor += texture(Sampler, fragTexCoord + i * vec2(textel.x * pc.dx, textel.y * pc.dy));
        sum += 4.0;
    }

    outColor /= sum;
}
