#version 450

#include "../__methods__/defines.glsl"

layout(constant_id = 0) const int transparentLayersCount = 1;

layout(set = 0, binding = 0) buffer cursorBuffer {
    float x;
    float y;
    uint number;
    float depth;
} cursor;
layout(set = 0, binding = 1) uniform sampler2D Position;
layout(set = 0, binding = 2) uniform sampler2D Depth;
layout(set = 0, binding = 3) uniform sampler2D LayersPosition[transparentLayersCount];
layout(set = 0, binding = 4) uniform sampler2D LayersDepth[transparentLayersCount];

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main()
{
    vec2 texSize = vec2(1.0f) / textureSize(Position, 0);
    uint number = 0xffffffff;

    uint primitiveNumber[transparentLayersCount + 1];
    float layerDepth[transparentLayersCount + 1];

    for(int i = 0; i < transparentLayersCount; i++) {
        primitiveNumber[i] = (0xffff0000 & floatBitsToUint(texture(LayersPosition[i], fragTexCoord.xy).a)) >> 16;
        layerDepth[i] = texture(LayersDepth[i], fragTexCoord.xy).r;
    }
    primitiveNumber[transparentLayersCount] = (0xffff0000 & floatBitsToUint(texture(Position, fragTexCoord.xy).a)) >> 16;
    layerDepth[transparentLayersCount] = texture(Depth, fragTexCoord.xy).r;

    float depth = layerDepth[0];
    if(abs(fragTexCoord.x - cursor.x) < texSize.x &&
        abs(fragTexCoord.y - cursor.y) < texSize.y){
        for(int i = 0; i < transparentLayersCount + 1; i++){
            if(depth >= layerDepth[i]){
                depth = layerDepth[i];
                number = primitiveNumber[i];
            }
        }
    }
    if(number != 0xffffffff){
        cursor.number = number;
    }
    if(abs(fragTexCoord.x - 0.5f) < texSize.x && abs(fragTexCoord.y - 0.5f) < texSize.y){
        cursor.depth = layerDepth[transparentLayersCount] < layerDepth[0] ? layerDepth[transparentLayersCount] : layerDepth[0];
    }
}
