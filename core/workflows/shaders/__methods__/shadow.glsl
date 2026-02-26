#ifndef SHADOW
#define SHADOW

#include "defines.glsl"

// Alternative to a hardcoded array — Fibonacci lattice, scales to any N:
// vec2 fibonacciSample(int i, int N) {
//     const float goldenAngle = 2.3998;  // 2π / φ², φ = golden ratio
//     float r     = sqrt((float(i) + 0.5) / float(N));  // uniform area distribution
//     float theta = float(i) * goldenAngle;
//     return vec2(r * cos(theta), r * sin(theta));
// }
// Usage: replace poissonDisk[i] with fibonacciSample(i, N) in loops,
// remove the array below, change 16 to the desired N.

// Poisson disk — uniformly distributed points in a unit circle
const vec2 poissonDisk[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

// Convert NDC depth to linear (view-space) distance.
// abs() is required: perspective matrix returns negative z (right-handed coordinate system)
float linearDepth(const in mat4 proj, float depth) {
    return abs(proj[3][2] / (depth * proj[2][3] - proj[2][2]));
}

// Step 1: average linear depth of blockers within the search region.
// Returns -1.0 if no blockers are found.
float findBlockerDepth(
    sampler2D shadowMap, const in mat4 proj,
    vec2 uv, float currentDepth, float searchWidth, mat2 rot)
{
    vec2 texelSize = vec2(1.0) / textureSize(shadowMap, 0);
    float blockerSum = 0.0;
    int numBlockers = 0;
    for (int i = 0; i < 16; i++) {
        vec2 sampleUV = uv + rot * poissonDisk[i] * searchWidth * texelSize;
        float depth = texture(shadowMap, sampleUV).r;
        if (depth < currentDepth) {
            // accumulate linear depth for correct averaging
            blockerSum += linearDepth(proj, depth);
            numBlockers++;
        }
    }
    return numBlockers > 0 ? blockerSum / float(numBlockers) : -1.0;
}

// Step 2: PCF with a fixed filter radius
float pcf(sampler2D shadowMap, vec2 uv, float currentDepth, float filterRadius, mat2 rot) {
    vec2 texelSize = vec2(1.0) / textureSize(shadowMap, 0);
    float shadow = 0.0;
    for (int i = 0; i < 16; i++) {
        vec2 sampleUV = uv + rot * poissonDisk[i] * filterRadius * texelSize;
        float depth = texture(shadowMap, sampleUV).r;
        shadow += currentDepth > depth ? 1.0 : 0.0;
    }
    return 1.0 - shadow / 16.0;
}

float shadowFactor(const in mat4 proj, sampler2D shadowMap, vec4 coordinates) {
    vec2 uv = coordinates.xy / coordinates.w * 0.5 + 0.5;
    float currentDepth = coordinates.z / coordinates.w;

    // Small bias: prevents self-shadowing without breaking contact and distant shadows.
    // 0.002 was too large — far from the light NDC is compressed,
    // the depth difference between blocker and receiver is < 0.0001, and a large bias hid blockers.
    const float bias = 0.0001;
    float biasedDepth = currentDepth - bias;

    // Per-pixel random rotation of the Poisson disk using screen-space coordinates.
    // NOTE: gl_FragCoord.xy is used instead of light-space uv, because uv is
    // reconstructed through the camera matrix and has float imprecision on movement.
    // A tiny change in uv → sin(...)*43758 → completely different hash → different pattern → flickering.
    // gl_FragCoord.xy are integer pixel coordinates, stable within a frame.
    // IGN (Interleaved Gradient Noise) — uniform distribution without moire at low sample counts
    float angle = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) * 2.0 * pi;
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

    const float lightSize = 10.0; // virtual light source size (5–25)

    // Step 1: average linear blocker depth
    float currentLinearDepth = linearDepth(proj, currentDepth);
    float avgBlockerLinearDepth = findBlockerDepth(shadowMap, proj, uv, biasedDepth, lightSize, rot);

    if (avgBlockerLinearDepth < 0.0) {
        // PCSS found no blockers: either fragment is in light or precision issue at distance.
        // Fallback: soft PCF with minimum radius instead of binary 0/1 —
        // binary caused hard shadow jumps on camera movement.
        return pcf(shadowMap, uv, biasedDepth, 1.0, rot);
    }

    // Step 2: penumbra width in linear space.
    // Linear depth gives correct results at any distance from the light source.
    float penumbraWidth = clamp(
        (currentLinearDepth - avgBlockerLinearDepth) / avgBlockerLinearDepth * lightSize,
        1.0,            // minimum: always at least slightly soft
        lightSize * 3.0 // maximum: avoid sampling far outside the shadow map
    );

    // Step 3: PCF with dynamic kernel
    return pcf(shadowMap, uv, biasedDepth, penumbraWidth, rot);
}

#endif
