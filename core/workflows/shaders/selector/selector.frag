#version 450

#include "../__methods__/defines.glsl"

layout(set = 0, binding = 0) buffer cursorBuffer {
    float x;
    float y;
    uint number;
    float depth;
} cursor;
layout(set = 0, binding = 1) uniform sampler2D normal;
layout(set = 0, binding = 2) uniform sampler2D depth;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main()
{
    vec2 texSize = vec2(1.0f) / textureSize(normal, 0);
    uint number = 0xffffffff;
    
    if(abs(fragTexCoord.x - cursor.x) < texSize.x && abs(fragTexCoord.y - cursor.y) < texSize.y){
        uint number = floatBitsToUint(texture(normal, fragTexCoord.xy).a);
    }

    if(number != 0xffffffff){
        cursor.number = number;
    }

    if(abs(fragTexCoord.x - 0.5f) < texSize.x && abs(fragTexCoord.y - 0.5f) < texSize.y){
        cursor.depth = texture(depth, fragTexCoord.xy).r;
    }
}
