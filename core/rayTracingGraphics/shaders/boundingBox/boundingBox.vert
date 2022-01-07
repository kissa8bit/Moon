#version 450

layout(set = 0, binding = 0) uniform GlobalUniformBuffer{
    mat4 proj;
    mat4 view;
} global;

layout (push_constant) uniform PushConstants{
    vec4 min;
    vec4 max;
    vec4 color;
} pushConstants;

vec3 min = vec3(pushConstants.min);
vec3 max = vec3(pushConstants.max);

vec3 vertex[24] = vec3[](
    vec3(min.x,min.y,min.z),
    vec3(max.x,min.y,min.z),

    vec3(max.x,min.y,min.z),
    vec3(max.x,max.y,min.z),
    
    vec3(max.x,max.y,min.z),
    vec3(min.x,max.y,min.z),
    
    vec3(min.x,max.y,min.z),
    vec3(min.x,min.y,min.z),
    
    vec3(min.x,min.y,max.z),
    vec3(max.x,min.y,max.z),

    vec3(max.x,min.y,max.z),
    vec3(max.x,max.y,max.z),
    
    vec3(max.x,max.y,max.z),
    vec3(min.x,max.y,max.z),
    
    vec3(min.x,max.y,max.z),
    vec3(min.x,min.y,max.z),

    vec3(min.x,min.y,min.z),
    vec3(min.x,min.y,max.z),
    
    vec3(max.x,min.y,min.z),
    vec3(max.x,min.y,max.z),
    
    vec3(max.x,max.y,min.z),
    vec3(max.x,max.y,max.z),
    
    vec3(min.x,max.y,min.z),
    vec3(min.x,max.y,max.z)
);

void main()
{
    gl_Position = global.proj * global.view * vec4(vertex[gl_VertexIndex],1.0f);
}