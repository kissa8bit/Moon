#version 450

layout(location = 0) out vec4 outColor;

layout (push_constant) uniform PushConstants{
    vec4 min;
    vec4 max;
    vec4 color;
} pushConstants;

void main()
{
    outColor = pushConstants.color;
}