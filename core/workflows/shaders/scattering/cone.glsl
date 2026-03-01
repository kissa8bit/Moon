#ifndef SCATTERING_CONE
#define SCATTERING_CONE

#include "../__methods__/geometricFunctions.glsl"
#include "../__methods__/defines.glsl"
#include "IntersectionOutput.glsl"

#define maxDist 1e7f

float conicDot(const in vec4 v1, const in vec4 v2){
    return v1.x * v2.x + v1.y * v2.y - v1.z * v2.z;
}

float findOutsideMinInter(const float t1, const float t2){
    return min((t1 > 0.0f ? t1 : maxDist), (t2 > 0.0f ? t2 : maxDist));
}

float findOutsideMaxInter(const float t1, const float t2){
    return max((t1 >0.0f ? t1 : maxDist), (t2 >0.0f ? t2 : maxDist));
}

float findInsideMinInter(const float t1, const float t2){
    return t1*t2 <= 0.0f ? min(t1,t2) : ( t1 < 0.0f ? max(t1,t2) : -maxDist);
}

float findInsideMaxInter(const float t1, const float t2){
    return t1*t2 <= 0.0f ? max(t1,t2) : ( t1 > 0.0f ? min(t1,t2) : maxDist);
}

float findBeginCone(const float t1, const float t2, const bool insideCondition, const bool positionCondition){
    return insideCondition ? ( positionCondition ? findInsideMinInter(t1,t2)
                                                 : max(t1,t2))
                           : findOutsideMinInter(t1,t2);
}

float findEndCone(const float t1, const float t2, const bool insideCondition, const bool positionCondition){
    return insideCondition ? ( positionCondition ? findInsideMaxInter(t1,t2)
                                                 : maxDist )
                           : findOutsideMaxInter(t1,t2);
}

IntersectionOutput findConeIntersection(const in vec4 viewPosition, const in vec4 viewDirection, mat4 lightProjMatrix, const in mat4 lightViewMatrix, uint type){
    IntersectionOutput outputStatus = IntersectionOutputDef();
    
    if(type == SPOT_LIGHTING_TYPE_SQUARE) {
        lightProjMatrix[0][0] *= sqrt(0.5f); 
        lightProjMatrix[1][1] *= sqrt(0.5f);
    }

    const float far = getFar(lightProjMatrix);
    const float height = getH(lightProjMatrix, far) / far;
    const float width = getW(lightProjMatrix, far) / far;

    const vec4 normalization = vec4(1.0f/width, 1.0f/height, 1.0f, 1.0f);

    const vec4 directionInLightCoord = normalization * (lightViewMatrix * viewDirection);
    const vec4 positionInLightCoord = normalization * (lightViewMatrix * viewPosition);
    
    const float dd = conicDot(directionInLightCoord, directionInLightCoord);
    if(dd == 0.0){
        return outputStatus;
    }
    
    const float dp = conicDot(positionInLightCoord, directionInLightCoord);
    const float pp = conicDot(positionInLightCoord, positionInLightCoord);
    const float D = dp * dp - dd * pp;

    outputStatus.inside = pp < 0.0f;
    outputStatus.intersectionCondition = D >= 0;

    if(outputStatus.intersectionCondition){
        const float sqrtD = sqrt(D);
        outputStatus.intersectionPoint1 = (-dp + sqrtD) / dd;
        outputStatus.intersectionPoint2 = (-dp - sqrtD) / dd;
    }

    return outputStatus;
}

bool isPositionBehind(const in vec4 viewPosition, const in vec4 lightPosition, const in vec4 lightDirection){
    return dot(viewPosition.xyz - lightPosition.xyz, lightDirection.xyz) > 0.0f;
}

vec4 findStartPositionCone(const in IntersectionOutput outputStatus, const in vec4 viewPosition, const in vec4 direction, const in vec4 lightPosition, const in vec4 lightDirection){
    const bool positionCondition = 
        outputStatus.inside && 
        isPositionBehind(viewPosition, lightPosition, lightDirection) && 
        (outputStatus.intersectionPoint1 >= 0.0f || outputStatus.intersectionPoint2 >= 0.0f);
    
    const float tBegin = !positionCondition 
        ? findBeginCone(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2, outputStatus.inside, positionCondition)
        : findEndCone(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2, outputStatus.inside, positionCondition);

    return viewPosition + direction * tBegin;
}

float findDirectionSign(const in IntersectionOutput outputStatus, const in vec4 viewPosition, const in vec4 lightPosition, const in vec4 lightDirection){
    const bool positionCondition = 
        outputStatus.inside && 
        isPositionBehind(viewPosition, lightPosition, lightDirection) && 
        (outputStatus.intersectionPoint1 >= 0.0f || outputStatus.intersectionPoint2 >= 0.0f);

    return !positionCondition ? 1.0f : -1.0f;
}

#endif