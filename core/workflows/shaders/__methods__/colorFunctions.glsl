#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS

bool checkBrightness(vec4 color) {
    return color.x > 0.95 && color.y > 0.95 && color.z > 0.95;
}

vec4 SRGBtoLINEAR(vec4 srgbIn) {
#ifdef MANUAL_SRGB
    #ifdef SRGB_FAST_APPROXIMATION
    vec3 linOut = pow(srgbIn.xyz, vec3(2.2));
    #else //SRGB_FAST_APPROXIMATION
    vec3 bLess = step(vec3(0.04045), srgbIn.xyz);
    vec3 linOut = mix(srgbIn.xyz / vec3(12.92), pow((srgbIn.xyz + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
    #endif
    return vec4(linOut, srgbIn.w);
#else //MANUAL_SRGB
    return srgbIn;
#endif //MANUAL_SRGB
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