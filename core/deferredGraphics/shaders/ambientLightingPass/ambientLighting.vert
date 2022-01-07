#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

layout(set = 0, binding = 4) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
} global;

vec2 step[6] = vec2[](
    vec2(0.0f, 0.0f),
    vec2(1.0f, 0.0f),
    vec2(1.0f, 1.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, 1.0f),
    vec2(0.0f, 0.0f)
);

void main()
{
    int xsteps = 1;
    int ysteps = 1;

    float x0 = -1.0f;
    float y0 = -1.0f;
    float dx = 2.0f/float(xsteps);
    float dy = 2.0f/float(ysteps);

    int arrayIndex = gl_VertexIndex % 6;
    int tileNumber = (gl_VertexIndex - arrayIndex)/6;
    int tileX = tileNumber % xsteps;
    int tileY = (tileNumber - tileX) / ysteps;

    float x = x0 + tileX*dx + step[arrayIndex].x * dx;
    float y = y0 + tileY*dy + step[arrayIndex].y * dy;

    gl_Position = vec4(vec2(x,y),0.0, 1.0);
}
