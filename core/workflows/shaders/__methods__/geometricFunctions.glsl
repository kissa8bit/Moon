#ifndef GEOMETRIC_FUNCTIONS
#define GEOMETRIC_FUNCTIONS

#include "defines.glsl"

bool checkZeroNormal(const in vec3 normal) {
    return normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0;
}

float getAspect(const in mat4 proj){
    return - proj[1][1] / proj[0][0];
}

vec3 getDirection(const in mat4 view){
    return - normalize(vec3(view[0][2], view[1][2], view[2][2]));
}

bool outsideSpot(const in mat4 proj, const in uint type, const in vec4 localPosition) {
    vec4 coordinates = localPosition * vec4(getAspect(proj), 1.0, -1.0, 1.0);

    switch(type) {
        case SPOT_LIGHTING_TYPE_CIRCLE: {
            return sqrt(coordinates.x * coordinates.x + coordinates.y * coordinates.y) >= coordinates.z;
        }
        case SPOT_LIGHTING_TYPE_SQUARE: {
            return abs(coordinates.x) >= abs(coordinates.z) || abs(coordinates.y) >= abs(coordinates.z);
        }
        default: {
            break;
        }
    }
    return true;
}

float zProj(const in mat4 projview, const in vec4 position){
    return projview[0][2]*position.x + projview[1][2]*position.y + projview[2][2]*position.z + projview[3][2]*position.w;
}

float wProj(const in mat4 projview, const in vec4 position){
    return projview[0][3]*position.x + projview[1][3]*position.y + projview[2][3]*position.z + projview[3][3]*position.w;
}

float depthProj(const in mat4 projview, vec4 position){
    return zProj(projview, position) / wProj(projview, position);
}

float linesIntersection(vec3 a, vec3 da, vec3 b, vec3 db){
    float eps = 1e-2;
    vec3 a_b = a - b;

    float detM = db.y*da.x - db.x*da.y;
    if(abs(detM) > eps){
        return -(db.y*a_b.x - db.x*a_b.y) / detM;
    }
    detM = db.z*da.x - db.x*da.z;
    if(abs(detM) > eps){
        return -(db.z*a_b.x - db.x*a_b.z) / detM;
    }
    detM = db.z*da.y - db.y*da.z;
    if(abs(detM) > eps){
        return -(db.z*a_b.y - db.y*a_b.z) / detM;
    }

    return 0.0f;
}

bool isInside(const in vec2 coords) {
    return coords.x > 0.0 && coords.y > 0.0 && coords.x < 1.0 && coords.y < 1.0;
}

vec4 positionProj(const in mat4 projview, const in vec4 position) {
    vec4 positionProj = projview * position;
    return positionProj / positionProj.w;
}

vec3 viewPosition(const in mat4 view) {
    return -transpose(mat3(view)) * vec3(view[3]);
}

#endif