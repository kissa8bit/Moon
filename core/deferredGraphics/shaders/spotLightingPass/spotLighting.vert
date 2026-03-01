#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 5) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
} global;

layout(set = 2, binding = 0) uniform LightBufferObject {
    mat4 proj;
    mat4 view;
    vec4 color;
    vec4 prop;
} light;

layout(location = 0) out vec4 eyePosition;

vec3 vertex[5];
PYRAMID_INDICES

void main() {
    eyePosition = vec4(viewPosition(global.view), 1.0f);

    const int index = pyramid_indices[gl_VertexIndex];
    const vec3 pos = getPyramidVertex(light.proj, light.view, index);
    gl_Position = global.proj * global.view * vec4(pos, 1.0);
}
