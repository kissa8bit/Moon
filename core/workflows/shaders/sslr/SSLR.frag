#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/pbr.glsl"
#include "../__methods__/colorFunctions.glsl"
#include "../__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
} global;
layout(set = 0, binding = 1) uniform sampler2D position;
layout(set = 0, binding = 2) uniform sampler2D normal;
layout(set = 0, binding = 3) uniform sampler2D Sampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

vec4 p0       = vec4(viewPosition(global.view), 1.0);
mat4 projview = global.proj * global.view;

vec2 findIncrement(const in vec4 pos, const in vec4 dir) {
    vec4 start = positionProj(projview, pos);
    vec4 end   = positionProj(projview, pos + dir);
    return normalize(end.xy - start.xy);
}

vec3 cam_u = getUDirection(global.view);
vec3 cam_v = getVDirection(global.view);
vec3 cam_n = getNDirection(global.view);

float cam_h = -1.0f / global.proj[1][1];
float cam_w = -global.proj[1][1] / global.proj[0][0] * cam_h;

vec4 getDir(vec2 planeCoords) {
    planeCoords = 2.0f * planeCoords - 1.0f;
    return normalize(vec4(cam_n + cam_u * cam_w * planeCoords.x - cam_v * cam_h * planeCoords.y, 0.0f));
}

vec4 calcPos(vec4 p, vec4 r, vec2 planeCoords) {
    vec4 d_i = getDir(planeCoords);
    float t  = linesIntersection(p0.xyz, d_i.xyz, p.xyz, r.xyz);
    return p0 + t * d_i;
}

float findPosDis(vec4 p, vec4 r, vec2 planeCoords) {
    vec4 i_pos = calcPos(p, r, planeCoords);
    vec4 r_pos = vec4(texture(position, planeCoords).xyz, 1.0);
    return zProj(projview, i_pos - r_pos);
}

vec4 SSLR(const in vec4 p, const in vec4 surf_n, const in vec4 d, const in vec4 r, vec2 planeCoords, vec2 increment, int steps, float f0) {
    bool binarySearch = false;
    float d_pos = 0.0f;
    for (int i = 0; i < steps; i++) {
        planeCoords += increment;
        d_pos = findPosDis(p, r, planeCoords);
        binarySearch = d_pos > 0.0f;
        if (binarySearch || !isInside(planeCoords)) {
            break;
        }
    }
    if (!binarySearch) {
        return vec4(0.0);
    }
    for (int j = 0; j < 8; j++) {
        int sign = d_pos > 0.0f ? -1 : 1;
        planeCoords += sign * (increment /= 2);
        d_pos = findPosDis(p, r, planeCoords);
    }

    // Плавный вес сходимости — заменяет жёсткий порог, убирает зернистость на границах
    float convergence = 1.0f - smoothstep(0.0f, 0.005f, d_pos * d_pos);
    if (convergence <= 0.0f) return vec4(0.0);

    // Отклонить обратные грани: нормаль в точке попадания должна смотреть против луча
    vec3 hitNormal = texture(normal, planeCoords).xyz;
    if (dot(r.xyz, hitNormal) >= 0.0f) return vec4(0.0);

    // Затухание у краёв экрана — прячет резкий обрыв отражений
    vec2 edgeFade = smoothstep(vec2(0.0f), vec2(0.1f), planeCoords) *
                    smoothstep(vec2(1.0f), vec2(0.9f), planeCoords);
    float fade = edgeFade.x * edgeFade.y;

    float cosTheta = 1.0f + dot(d, surf_n);
    float fresnel  = clamp(f0 + (1.0f - f0) * pow(cosTheta, 5), 0.0f, 1.0f);
    return fade * convergence * fresnel * texture(Sampler, planeCoords);
}

void main() {
    outColor = vec4(0.0);

    ivec2 texCoord = ivec2(fragTexCoord * vec2(textureSize(position, 0)));
    float material  = texelFetch(position, texCoord, 0).a;
    float roughness = decodeParameter(0x000000ff, 0, material) / 255.0f;
    float metallic  = decodeParameter(0x0000ff00, 8, material) / 255.0f;
    float ao        = decodeParameter(0x00ff0000, 16, material) / 255.0f;

    float f0         = mix(0.04f, 1.0f, metallic);
    float smoothness = 1.0f - roughness;
    float s          = f0 * smoothness * smoothness;
    if (s < 0.01f) return;

    const int steps = 50;
    const float incrementFactor = 0.5f / steps;

    vec4 p      = vec4(texture(position, fragTexCoord).xyz, 1.0);
    vec4 surf_n = vec4(texture(normal, fragTexCoord).xyz, 0.0);
    vec4 d      = normalize(p - p0);
    vec4 r      = normalize(reflect(d, surf_n));

    vec2 increment = incrementFactor * findIncrement(p, r);

    outColor += ao * s * SSLR(p, surf_n, d, r, fragTexCoord, increment, steps, f0);
}
