#ifndef COLOR_FUNCTIONS
#define COLOR_FUNCTIONS

bool checkBrightness(vec4 color)
{
    return color.x > 0.95f && color.y > 0.95f && color.z > 0.95f;
}

vec4 SRGBtoLINEAR(vec4 srgbIn)
{
#ifdef MANUAL_SRGB
    #ifdef SRGB_FAST_APPROXIMATION
        vec3 linOut = pow(srgbIn.xyz,vec3(2.2));
    #else //SRGB_FAST_APPROXIMATION
        vec3 bLess = step(vec3(0.04045),srgbIn.xyz);
        vec3 linOut = mix( srgbIn.xyz/vec3(12.92), pow((srgbIn.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
    #endif //SRGB_FAST_APPROXIMATION
    return vec4(linOut,srgbIn.w);
#else //MANUAL_SRGB
    return srgbIn;
#endif //MANUAL_SRGB
}

#endif