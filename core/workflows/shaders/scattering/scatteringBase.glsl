#include "../__methods__/lightDrop.glsl"
#include "../__methods__/geometricFunctions.glsl"

#ifndef SCATTERING_BASE
#define SCATTERING_BASE

float maxDist = 1e7f;

struct intersectionOutput{
    bool intersectionCondition;
    bool inside;
    float intersectionPoint1;
    float intersectionPoint2;
};

vec4 findLightDirection(const in mat4 lightViewMatrix){
    return - normalize(vec4(lightViewMatrix[0][2],lightViewMatrix[1][2],lightViewMatrix[2][2],0.0f));
}

bool isPositionBehind(const in vec4 viewPosition, const in vec4 lightPosition, const in vec4 lightDirection){
    return dot(normalize(viewPosition.xyz - lightPosition.xyz),lightDirection.xyz) > 0.0f;
}

vec3 coordinatesInLocalBasis(const in mat4 projViewMatrix, vec4 position){
    vec4 projection = projViewMatrix * position;
    vec3 normProjection = (projection.xyz)/projection.w;
    vec2 coordinatesXY = normProjection.xy * 0.5f + 0.5f;
    return vec3(coordinatesXY,normProjection.z);
}

//==//==//==//==//==//==//==//==//==//==//CONE//==//==//==//==//==//==//==//==//==//==//

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

intersectionOutput findConeIntersection(const in vec4 viewPosition, const in vec4 viewDirection, const in mat4 lightProjMatrix, const in mat4 lightViewMatrix){
    intersectionOutput outputStatus;

    float far = lightProjMatrix[3][2]/(1.0f+lightProjMatrix[2][2]);
    float height = -far/lightProjMatrix[1][1];
    float width = far/lightProjMatrix[0][0];

    float xAxis = width/far;
    float yAxis = height/far;

    vec4 directionInLightCoord = lightViewMatrix * viewDirection;
    vec4 positionInLightCoord = lightViewMatrix * viewPosition;

    directionInLightCoord = vec4(directionInLightCoord.x/xAxis,directionInLightCoord.y/yAxis,directionInLightCoord.z,directionInLightCoord.w);
    positionInLightCoord = vec4(positionInLightCoord.x/xAxis,positionInLightCoord.y/yAxis,positionInLightCoord.z,positionInLightCoord.w);

    float dp = conicDot(positionInLightCoord, directionInLightCoord);
    float dd = conicDot(directionInLightCoord, directionInLightCoord);
    float pp = conicDot(positionInLightCoord, positionInLightCoord);

    float D = dp*dp - dd*pp;

    outputStatus.inside = pp < 0.0f;
    outputStatus.intersectionCondition = D >= 0;
    outputStatus.intersectionPoint1 = outputStatus.intersectionCondition ? (-dp + sqrt(D))/dd : 0.0f;
    outputStatus.intersectionPoint2 = outputStatus.intersectionCondition ? (-dp - sqrt(D))/dd : 0.0f;

    return outputStatus;
}

vec4 findStartPositionCone(intersectionOutput outputStatus, const in vec4 position, const in vec4 direction, const in vec4 lightPosition, const in vec4 lightDirection){
    bool positionCondition = isPositionBehind(position,lightPosition,lightDirection);
    float tBegin = findBeginCone(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.inside, positionCondition);
    float tEnd = findEndCone(outputStatus.intersectionPoint1,outputStatus.intersectionPoint2,outputStatus.inside, positionCondition);

    return (!outputStatus.inside || !positionCondition) ? position + direction * tBegin
                                                        : (outputStatus.intersectionPoint1<0.0f&&outputStatus.intersectionPoint2<0.0f ? position + direction * tBegin
                                                                                                                                      : position + direction * tEnd);
}

float findDirectionCone(intersectionOutput outputStatus, const in vec4 position, const in vec4 lightPosition, const in vec4 lightDirection){
    return !outputStatus.inside || !isPositionBehind(position,lightPosition,lightDirection) ? 1.0f : (outputStatus.intersectionPoint1<0.0f&&outputStatus.intersectionPoint2<0.0f ? 1.0f : -1.0f);
}

vec4 findConePointColor(const in vec3 point, sampler2D shadowMap, sampler2D lightTexture, const in vec4 lightColor, const in mat4 lightProjViewMatrix, const in mat4 lightProjMatrix, const in vec3 lightPosition, const in vec3 lightDirection, const float dropFactor){
    vec4 color = vec4(0.0f);

    vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(point.xyz,1.0f));
    if(coordinates.z<texture(shadowMap, coordinates.xy).x){
        float drop = dropFactor * lightDrop(length(lightPosition - point));
        drop = drop > 0.0f ? drop : 1.0f;
        float distribusion = lightDistribusion(point,lightPosition,lightProjMatrix,lightDirection);

        color = max(texture(lightTexture, coordinates.xy), lightColor)/drop*distribusion;
    }
    return color;
}

vec4 findConePointColor(const in vec3 point, sampler2D lightTexture, const in vec4 lightColor, const in mat4 lightProjViewMatrix, const in mat4 lightProjMatrix, const in vec3 lightPosition, const in vec3 lightDirection, const float dropFactor){
    vec4 color = vec4(0.0f);

    float drop = dropFactor * lightDrop(length(lightPosition - point));
    drop = drop > 0.0f ? drop : 1.0f;
    float distribusion = lightDistribusion(point,lightPosition,lightProjMatrix,lightDirection);

    vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(point.xyz,1.0f));
    color = max(texture(lightTexture, coordinates.xy), lightColor)/drop*distribusion;

    return color;
}

//==//==//==//==//==//==//==//==//==//==//PYRAMID//==//==//==//==//==//==//==//==//==//==//

float findFirstSolution(const in vec3 a, const in vec3 b, const in vec3 c, const in vec3 d){
    vec3 bc = cross(b,c);
    vec3 dcxbc = cross(cross(d,c),bc);
    vec3 acxbc = cross(cross(a,c),bc);

    return dot(dcxbc,acxbc)/dot(acxbc,acxbc);
}

float findTriangleIntersection(const in vec3 p0, const in vec3 d, const in vec3 v0, const in vec3 v1, const in vec3 v2){
    vec3 P0 = p0 - v2;
    vec3 V0 = v0 - v2;
    vec3 V1 = v1 - v2;

    float s0 = findFirstSolution(V0,V1,-d,P0);
    float s1 = findFirstSolution(V1,V0,-d,P0);
    float t  = findFirstSolution(-d,V1,V0,P0);

    return (s1<0.0f||s0+s1>1.0) ? 0.0f : t;
}

intersectionOutput findPyramidIntersection(const in vec4 viewPosition, const in vec4 viewDirection, const in vec4 lightPosition, const in mat4 lightProjMatrix, const in mat4 lightViewMatrix){
    intersectionOutput outputStatus;

    vec4 positionInLightCoord = lightProjMatrix * lightViewMatrix * viewPosition;
    positionInLightCoord /= positionInLightCoord.w;

    vec3 n = - normalize(vec3(lightViewMatrix[0][2],lightViewMatrix[1][2],lightViewMatrix[2][2]));
    vec3 u =   normalize(vec3(lightViewMatrix[0][0],lightViewMatrix[1][0],lightViewMatrix[2][0]));
    vec3 v =   normalize(vec3(lightViewMatrix[0][1],lightViewMatrix[1][1],lightViewMatrix[2][1]));

    float far  = lightProjMatrix[3][2]/(lightProjMatrix[2][2]+1.0f);
    float h = -far/lightProjMatrix[1][1];
    float w = lightProjMatrix[1][1]/lightProjMatrix[0][0]*h;

    vec3 v0 = lightPosition.xyz;
    vec3 v1 = lightPosition.xyz + far*n + w*u + h*v;
    vec3 v2 = lightPosition.xyz + far*n + w*u - h*v;
    vec3 v3 = lightPosition.xyz + far*n - w*u + h*v;
    vec3 v4 = lightPosition.xyz + far*n - w*u - h*v;

    float t[4] = float[4](
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v1,v2),
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v2,v4),
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v4,v3),
        findTriangleIntersection(viewPosition.xyz,viewDirection.xyz,v0,v3,v1)
    );

    outputStatus.inside = (abs(positionInLightCoord.x)<=1.0f)&&(abs(positionInLightCoord.y)<=1.0f)&&(positionInLightCoord.z>=0.0f);
    outputStatus.intersectionCondition = t[0]+t[1]+t[2]+t[3]!=0.0f;

    float t1 = 0.0f;
    float t2 = 0.0f;
    for(int i=0;i<4;i++){
        if(t[i]  !=0.0f) if(t1==0.0f) t1 = t[i];
        if(t[3-i]!=0.0f) if(t2==0.0f) t2 = t[3-i];
    }

    outputStatus.intersectionPoint1 = t1;
    outputStatus.intersectionPoint2 = t2;

    return outputStatus;
}

bool outsidePyramidCondition(vec3 coordinates, float type)
{
    return abs(coordinates.x) < 1.0f && abs(coordinates.y) < 1.0f && abs(coordinates.z) < 1.0f;
}

vec4 findPyramidPointColor(const in vec3 point, sampler2D shadowMap, sampler2D lightTexture, const in vec4 lightColor, const in mat4 lightProjViewMatrix, const in mat4 lightProjMatrix, const in vec3 lightPosition, const in vec3 lightDirection, const float dropFactor){
    vec4 color = vec4(0.0f);

    vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(point.xyz,1.0f));
    if(outsidePyramidCondition(vec3(2.0f*coordinates.x - 1.0f, 2.0f*coordinates.y - 1.0f,coordinates.z), 1.0f) && coordinates.z<texture(shadowMap, coordinates.xy).x){
        float drop = dropFactor * lightDrop(length(lightPosition - point));
        drop = drop > 0.0f ? drop : 1.0f;

        float distribusion = lightDistribusion(point,lightPosition,lightProjMatrix,lightDirection);

        color = max(texture(lightTexture, coordinates.xy), lightColor)/drop/drop;
    }
    return color;
}

vec4 findPyramidPointColor(const in vec3 point, sampler2D lightTexture, const in vec4 lightColor, const in mat4 lightProjViewMatrix, const in mat4 lightProjMatrix, const in vec3 lightPosition, const in vec3 lightDirection, const float dropFactor){
    vec4 color = vec4(0.0f);

    float drop = dropFactor * lightDrop(length(lightPosition - point));
    drop = drop > 0.0f ? drop : 1.0f;

    float distribusion = lightDistribusion(point,lightPosition,lightProjMatrix,lightDirection);

    vec3 coordinates = coordinatesInLocalBasis(lightProjViewMatrix,vec4(point.xyz,1.0f));
    color = max(texture(lightTexture, coordinates.xy), lightColor)/drop/drop;

    return color;
}

//==//==//==//==//==//==//==//==//==//==//GENERAL//==//==//==//==//==//==//==//==//==//==//

vec4 findOpositeDir(vec3 q1, vec3 direction, vec3 lightDirection){
    vec3 n = normalize(cross(direction,q1));
    vec3 tan = normalize(cross(lightDirection,n));
    vec3 bitan = normalize(cross(tan,n));
    return vec4( - normalize(reflect(q1,bitan)), 0.0f);
}

//==//==//==//==//==//==//==//==//==//==//SCATTERING//==//==//==//==//==//==//==//==//==//==//

vec4 LightScattering(
        const int steps,
        const in mat4 lightViewMatrix,
        const in mat4 lightProjMatrix,
        const in mat4 lightProjViewMatrix,
        const in vec4 lightPosition,
        const in vec4 lightColor,
        const in mat4 projView,
        const in vec4 position,
        const in vec4 fragPosition,
        sampler2D lightTexture,
        sampler2D shadowMap,
        const float depthMap,
        const float dropFactor,
        const float type)
{
    vec4 outScatteringColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    float minDepth = 1.0f;

    vec4 direction = normalize(fragPosition - position);
    vec4 lightDirection = findLightDirection(lightViewMatrix);

    intersectionOutput outputStatus = type == 0.0f ? findConeIntersection(position,direction,lightProjMatrix,lightViewMatrix)
                                                   : findPyramidIntersection(position,direction,lightPosition,lightProjMatrix,lightViewMatrix);

    if(outputStatus.intersectionCondition){
        if(type == 1.0f){
            mat4x4 Proj = lightProjMatrix; Proj[0][0] *= sqrt(0.5f); Proj[1][1] *= sqrt(0.5f);
            outputStatus = findConeIntersection(position,direction,Proj,lightViewMatrix);
        }
        vec4 start = findStartPositionCone(outputStatus,position,direction,lightPosition,lightDirection);
        direction *= findDirectionCone(outputStatus,position,lightPosition,lightDirection);

        vec4 directionToLight1 = normalize(lightPosition - start);
        vec4 directionToLight2 = findOpositeDir(directionToLight1.xyz,direction.xyz,lightDirection.xyz);

        for(float distance = 0.0f, step = 1.0f/(steps-1); distance < 1.0f; distance += step){
            vec4 lightDir = - normalize(directionToLight1 + (directionToLight2 - directionToLight1) * distance);
            float t = linesIntersection(start.xyz, direction.xyz, lightPosition.xyz, lightDir.xyz);
            vec4 pointOfScattering = start + direction * t;
            float depthOfScattering = depthProj(projView,pointOfScattering);
            if((depthMap > depthOfScattering) && (t > 0)){
                outScatteringColor = type == 0.0f ? outScatteringColor + step * findConePointColor(pointOfScattering.xyz,shadowMap,lightTexture,lightColor,lightProjViewMatrix,lightProjMatrix,lightPosition.xyz,lightDirection.xyz, dropFactor)
                                                  : max(outScatteringColor, 0.5f * findPyramidPointColor(pointOfScattering.xyz,shadowMap,lightTexture,lightColor,lightProjViewMatrix,lightProjMatrix,lightPosition.xyz,lightDirection.xyz, dropFactor));
                minDepth = min(minDepth, depthOfScattering);
            }
        }
    }
    return vec4(outScatteringColor.xyz, minDepth);
}

#endif
