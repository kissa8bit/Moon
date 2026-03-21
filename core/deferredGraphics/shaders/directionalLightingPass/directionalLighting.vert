#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
    vec2 viewport;
} global;

layout(location = 0) out vec4 eyePosition;

void main() {
    eyePosition = vec4(viewPosition(global.view), 1.0f);

    // Full-screen triangle. Depth = 1.0 so the hardware depth test (GREATER)
    // passes for all pixels that have scene geometry (depth < 1.0) and
    // discards sky pixels (cleared to depth = 1.0).
    const vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    gl_Position = vec4(positions[gl_VertexIndex], 1.0, 1.0);
}
