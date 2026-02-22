#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/math.glsl"
#include "blur.glsl"

layout(set = 0, binding = 0) uniform sampler2D color;
layout(set = 0, binding = 1) uniform sampler2D depth;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor0;
layout(location = 1) out vec4 outColor1;

layout(push_constant) uniform PC {
    float depth;
} pc;

void main() {
    outColor0 = vec4(0.0);
    // outColor1 is read by the y-pass (yBlur) as bufferAttachment. Its in-focus pixels
    // must NOT be written as vec4(0.0): the y-pass blur kernel samples neighbors from
    // this buffer, and zeros at in-focus positions get mixed into the blurred result of
    // nearby out-of-focus pixels, producing large tile-sized patches of incorrectly dark
    // color that flicker randomly each frame. blur() avoids this by returning
    // texture(tex, texCoord) for in-focus pixels — always a valid color, never a black hole.
    outColor1 = blur(color, depth, fragTexCoord, vec2(1.0, 0.0), pc.depth);
}
