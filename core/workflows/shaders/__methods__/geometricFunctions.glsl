#ifndef GEOMETRIC_FUNCTIONS
#define GEOMETRIC_FUNCTIONS

#include "defines.glsl"

bool checkZeroNormal(const in vec3 normal) {
    return normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0;
}

float getAspect(const in mat4 proj){
    return - proj[1][1] / proj[0][0];
}

float getFov(const in mat4 proj){
    return 2.0 * atan(- 1.0 / proj[1][1]);
}

float getFar(const in mat4 proj){
    return proj[3][2] / (proj[2][2] + 1.0);
}

float getH(const in mat4 proj, float far){
    return - far / proj[1][1];
}

float getW(const in mat4 proj, float far){
    return - far / proj[0][0];
}

vec3 getUDirection(const in mat4 view){
    return normalize(vec3(view[0][0], view[1][0], view[2][0]));
}

vec3 getVDirection(const in mat4 view){
    return normalize(vec3(view[0][1], view[1][1], view[2][1]));
}

vec3 getNDirection(const in mat4 view){
    return - normalize(vec3(view[0][2], view[1][2], view[2][2]));
}

bool outsideSpot(const in mat4 proj, const in uint type, const in vec4 localPosition) {
    // Project x and y using the projection matrix elements, which encode both fov and aspect.
    // At the frustum boundary, |ex| and |ey| equal -localPosition.z (nz).
    const float nz = -localPosition.z;
    const float ex = localPosition.x * proj[0][0];
    const float ey = localPosition.y * proj[1][1];

    switch(type) {
        case SPOT_LIGHTING_TYPE_CIRCLE: {
            return sqrt(ex * ex + ey * ey) >= nz;
        }
        case SPOT_LIGHTING_TYPE_SQUARE: {
            return abs(ex) >= nz || abs(ey) >= nz;
        }
        default: {
            break;
        }
    }
    return true;
}

float zProj(const in mat4 projview, const in vec4 position){
    return projview[0][2] * position.x + projview[1][2] * position.y + projview[2][2] * position.z + projview[3][2] * position.w;
}

float wProj(const in mat4 projview, const in vec4 position){
    return projview[0][3] * position.x + projview[1][3] * position.y + projview[2][3] * position.z + projview[3][3] * position.w;
}

float depthProj(const in mat4 projview, vec4 position){
    return zProj(projview, position) / wProj(projview, position);
}

float linesIntersection(vec3 a, vec3 da, vec3 b, vec3 db){
    const float eps = 1e-5;
    const vec3 a_b = a - b;

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
    const vec4 positionProj = projview * position;
    return positionProj / positionProj.w;
}

vec3 normalizedCoords(const in mat4 projMatrix, const in vec4 position){
    const vec4 projection = projMatrix * position;
    return vec3(projection.xy / projection.w * 0.5f + 0.5f, projection.z / projection.w);
}

vec3 viewPosition(const in mat4 view) {
    return -transpose(mat3(view)) * vec3(view[3]);
}

#define PYRAMID_INDICES int pyramid_indices[18] = int[](0,4,1,0,1,2,0,2,3,0,3,4,4,2,1,2,4,3);

vec3 getPyramidVertex(const in mat4 proj, const in mat4 view, int index){
    vec3 u = getUDirection(view);
    vec3 v = getVDirection(view);
    vec3 n = getNDirection(view);
    float far = getFar(proj);
    float h = getH(proj, far);
    float w = getW(proj, far);
    vec3 position = viewPosition(view);
    vec3 center = position + far * n;
    vec3 x = w * u;
    vec3 y = h * v;

    switch(index){
        case 0:
            return position;
        case 1:
            return center + x + y;
        case 2:
            return center + x - y;
        case 3:
            return center - x - y;
        case 4:
            return center - x + y;
    }
    return vec3(0.0);
}

#endif