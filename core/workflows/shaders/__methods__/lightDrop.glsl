#ifndef LIGHT_DROP_GLSL
#define LIGHT_DROP_GLSL

float lightDrop(const float distance) {
    return pow(distance, 2);
}

float lightDistribusion(const in vec3 position, const in vec3 lightPosition, const in mat4 lightProjMatrix, const in vec3 lightDirection) {
    float fov = 2 * atan(-1.0 / lightProjMatrix[1][1]);
    float theta = acos(dot(normalize(position - lightPosition), lightDirection));
    float arg = 3.141592653589793 * theta / fov;
    return pow(cos(arg), 4);
}

#endif