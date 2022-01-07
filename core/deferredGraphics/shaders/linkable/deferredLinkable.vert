#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

vec2 fragCoord[6] = vec2[](
    vec2(0.0f, 0.0f),
    vec2(1.0f, 0.0f),
    vec2(1.0f, 1.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, 1.0f),
    vec2(0.0f, 0.0f)
);

layout(push_constant) uniform PC {
    vec2 offset;
    vec2 size;
} pc;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    vec2 offset = -1.0 + 2.0f * pc.offset;
    vec2 size = 2.0f * pc.size;
    gl_Position = vec4(offset + size * fragCoord[gl_VertexIndex],0.0, 1.0);
    fragTexCoord = fragCoord[gl_VertexIndex];
}
