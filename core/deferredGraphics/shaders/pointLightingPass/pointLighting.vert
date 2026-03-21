#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
    vec2 viewport;
} global;

layout(set = 3, binding = 0) uniform LightBufferObject {
    vec4 position;
    vec4 color;
    vec4 prop;  // x=radius, y=powerFactor, z=dropFactor, w=unused
} light;

layout(location = 0) out vec4 eyePosition;

void main() {
    eyePosition = vec4(viewPosition(global.view), 1.0f);

    const vec3 pos = getSphereVertex(light.position.xyz, light.prop.x, gl_VertexIndex);
    gl_Position = global.proj * global.view * vec4(pos, 1.0);
}
