#ifndef SCATTERING_BASE
#define SCATTERING_BASE

#include "../__methods__/lightDrop.glsl"
#include "../__methods__/geometricFunctions.glsl"
#include "../__methods__/defines.glsl"

#include "IntersectionOutput.glsl"
#include "cone.glsl"

vec4 findOppositeDir(const in vec3 q1, const in vec3 direction, const in vec3 lightDirection){
    const vec3 n = normalize(cross(direction,q1));
    const vec3 tan = normalize(cross(lightDirection,n));
    const vec3 bitan = normalize(cross(tan,n));
    return vec4( - normalize(reflect(q1,bitan)), 0.0f);
}

bool insidePyramidCondition(vec3 coordinates)
{
    return abs(2.0f * coordinates.x - 1.0f) < 1.0f && abs(2.0f * coordinates.y - 1.0f) < 1.0f && coordinates.z >= 0.0f && coordinates.z < 1.0f;
}

vec4 findPointColor(
    const in vec3 point, 
    sampler2D shadowMap, 
    sampler2D lightTexture, 
    const in vec4 lightColor, 
    const in mat4 lightProjViewMatrix, 
    const in mat4 lightProjMatrix, 
    const in mat4 lightViewMatrix, 
    const in vec3 lightPosition,
    const vec4 prop,
    uint type)
{
    const vec3 coordinates = normalizedCoords(lightProjViewMatrix, vec4(point.xyz,1.0f));
    if(coordinates.z >= texture(shadowMap, coordinates.xy).x){
        return vec4(0.0f);
    }
    
    if(type == SPOT_LIGHTING_TYPE_SQUARE && !insidePyramidCondition(coordinates)){
        return vec4(0.0f);
    }
        
    const float innerFraction = prop.x;
    const float exponent      = prop.y;
    const float power         = prop.z;
    const float dropFactor    = prop.w;

    const float drop = lightDrop(max(dropFactor, 0.01) * max(length(lightPosition - point), 0.01));
    const float distribusion = lightDistribusion(point, lightProjMatrix, lightViewMatrix, innerFraction, exponent, type);
    const vec4 color = max(texture(lightTexture, coordinates.xy), lightColor);

    return power / pi * distribusion * color / drop;
}

vec4 LightScattering(
        const int steps,
        const float density,
        const in mat4 lightViewMatrix,
        const in mat4 lightProjMatrix,
        const in vec4 lightColor,
        const in mat4 projView,
        const in vec4 eyePosition,
        const in vec4 fragPosition,
        sampler2D lightTexture,
        sampler2D shadowMap,
        const float depthMap,
        const vec4 prop,
        const uint type)
{
    vec4 outScatteringColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    float minDepth = 1.0f;

    const vec4 eyeDirection = normalize(fragPosition - eyePosition);

    // Find the intersection of the view ray with the light cone
    // If the type of light source is pyramid function finds intersection with describing cone
    IntersectionOutput outputStatus = findConeIntersection(eyePosition, eyeDirection, lightProjMatrix, lightViewMatrix, type);
    
    // If there is no intersection, return the original color with the minimum depth
    if(!outputStatus.intersectionCondition){
        return vec4(outScatteringColor.xyz, minDepth);
    }
    
    const mat4 lightProjViewMatrix = lightProjMatrix * lightViewMatrix;
    const vec4 lightPosition = vec4(viewPosition(lightViewMatrix), 1.0f);
    const vec4 lightDirection = vec4(getNDirection(lightViewMatrix), 0.0f);

    // Find the nearest point which intersects cone along the view direction and find unit direction to this point
    const vec4 start = findStartPositionCone(outputStatus, eyePosition, eyeDirection, lightPosition, lightDirection);
    const vec4 direction = eyeDirection * findDirectionSign(outputStatus, eyePosition, lightPosition, lightDirection);

    // Find the generatrix of a cone and correspondent opposite line on the plane formed by the start point, the view direction and direction to light source
    const vec4 directionToLight1 = normalize(lightPosition - start);
    const vec4 directionToLight2 = findOppositeDir(directionToLight1.xyz, direction.xyz, lightDirection.xyz);

    for(float distance = 0.0f, step = 1.0f / float(steps - 1); distance < 1.0f; distance += step)
    {
        // Interpolate ray to the light source uniformly along the view direction
        const vec3 lightRayDir = - normalize(directionToLight1 + (directionToLight2 - directionToLight1) * distance).xyz;
        // Find point of scattering as intersection of ray from light source and view direction from eye position
        const float t = linesIntersection(eyePosition.xyz, eyeDirection.xyz, lightPosition.xyz, lightRayDir);
        if(t <= 0){
            continue;
        }

        const vec4 pointOfScattering = eyePosition + eyeDirection * t;
        const float depthOfScattering = depthProj(projView, pointOfScattering);
        if(depthMap <= depthOfScattering){
            continue;
        }
        
        // Find color in point of scattering
        // For pyramid light source check if the point is inside the pyramid
        outScatteringColor += step * findPointColor(pointOfScattering.xyz, shadowMap, lightTexture, lightColor, lightProjViewMatrix, lightProjMatrix, lightViewMatrix, lightPosition.xyz, prop, type);
        minDepth = min(minDepth, depthOfScattering);
    }
    
    // Find reducing factor from max fov and current cone section angle
    const float fov = getFov(lightProjMatrix);
    const float localFov = acos(dot(directionToLight1.xyz, directionToLight2.xyz));
    const float fovRatio = localFov / fov;

    return vec4(fovRatio * density * outScatteringColor.xyz, minDepth);
}

#endif
