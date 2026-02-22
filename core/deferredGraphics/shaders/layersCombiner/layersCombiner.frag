#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(constant_id = 0) const int layersCount = 1;

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
} global;

layout(set = 0, binding = 1) uniform sampler2D colorMap[layersCount];
layout(set = 0, binding = 2) uniform sampler2D bloomMap[layersCount];
layout(set = 0, binding = 3) uniform sampler2D positionMap[layersCount];
layout(set = 0, binding = 4) uniform sampler2D normalMap[layersCount];
layout(set = 0, binding = 5) uniform sampler2D depthMap[layersCount];
layout(set = 0, binding = 6) uniform sampler2D skyboxMap;
layout(set = 0, binding = 7) uniform sampler2D skyboxBloomMap;
layout(set = 0, binding = 8) uniform sampler2D scatteringMap;
layout(set = 0, binding = 9) uniform sampler2D sslrMap;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;
layout(location = 2) out vec4 outBlur;

mat4 projview = global.proj * global.view;
vec3 eyePosition = viewPosition(global.view);

vec4 accum(sampler2D maps[layersCount]){
    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;
    
    for(int i = layersCount - 1; i >= 0; --i)
    {
        vec4 src = texture(maps[i], fragTexCoord);

        accumColor = src.rgb * src.a + accumColor * (1.0 - src.a);
        accumAlpha = src.a + accumAlpha * (1.0 - src.a);
    }
    return vec4(accumColor, accumAlpha);
}

void main() {
    outColor = accum(colorMap);
    outBloom = accum(bloomMap);
    
    if(texture(depthMap[0], fragTexCoord).r == 1.0) {
        outColor += texture(skyboxMap, fragTexCoord);
        outBloom += texture(skyboxBloomMap, fragTexCoord);
    }
    
    outColor += vec4(texture(scatteringMap, fragTexCoord.xy).xyz, 0.0);
    outBlur = outColor;
}