#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS

bool checkBrightness(vec4 color, float threshold) {
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    return brightness > threshold;
}

bool checkBrightness(vec4 color) {
    return checkBrightness(color, 1.0);
}

float decodeParameter(uint mask, uint shift, float code){
    return float((mask & floatBitsToUint(code)) >> shift);
}

float codeToFloat(vec4 val){
    return uintBitsToFloat(
        uint(255.0f * val.r) << 0  |
        uint(255.0f * val.g) << 8  |
        uint(255.0f * val.b) << 16 |
        uint(255.0f * val.a) << 24 );
}

float codeToFloat(vec3 val, float a){
    return uintBitsToFloat(
        uint(255.0f * val.r) << 0  |
        uint(255.0f * val.g) << 8  |
        uint(255.0f * val.b) << 16 |
        uint(255.0f * a    ) << 24 );
}

vec4 decodeFromFloat(float val){
    return vec4(
        decodeParameter(0x000000ff, 0, val) / 255.0f,
        decodeParameter(0x0000ff00, 8, val) / 255.0f,
        decodeParameter(0x00ff0000, 16, val) / 255.0f,
        decodeParameter(0xff000000, 24, val) / 255.0f
    );
}

#endif