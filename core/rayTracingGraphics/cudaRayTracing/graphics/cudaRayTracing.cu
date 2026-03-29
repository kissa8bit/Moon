#include "cudaRayTracing.h"

#include <cudaRayTracing/utils/timer.h>
#include <cudaRayTracing/utils/operations.h>
#include <cudaRayTracing/math/ray.h>
#include <cudaRayTracing/materials/material.h>

namespace cuda::rayTracing {

void RayTracing::setExtent(uint32_t w, uint32_t h){
    width = w;
    height = h;
}

void RayTracing::bind(Object* obj) {
    // Load model in local space (identity transform keeps vertices in object space).
    // The world transform is applied at ray-traversal time via ObjectInstance::hit().
    obj->model->load(mat4f::identity());

    BLASEntry entry;
    entry.toWorld = obj->transform;
    entry.toLocal = inverse(obj->transform);

    for (const auto& primitive : obj->model->primitives) {
        entry.hostTree.storage.push_back(&primitive);
    }

    // Compute world-space bbox by transforming all 8 corners of the local bbox.
    box localBbox;
    for (const auto& primitive : obj->model->primitives) {
        const box b = primitive.bbox;
        localBbox.min = min(localBbox.min, b.min);
        localBbox.max = max(localBbox.max, b.max);
    }
    {
        const vec4f& lo = localBbox.min;
        const vec4f& hi = localBbox.max;
        box wb;
        for (int mask = 0; mask < 8; mask++) {
            const vec4f corner(
                (mask & 1) ? hi[0] : lo[0],
                (mask & 2) ? hi[1] : lo[1],
                (mask & 4) ? hi[2] : lo[2],
                1.0f
            );
            const vec4f wc = entry.toWorld * corner;
            wb.min = min(wb.min, wc);
            wb.max = max(wb.max, wc);
        }
        entry.worldBbox = wb;
    }

    // Fake primitive for the TLAS host tree (only bbox matters for SAH/visualisation).
    entry.tlasPrimitive = Primitive{ Triangle{}, entry.worldBbox };

    blasEntries.push_back(std::move(entry));
}

void RayTracing::setCamera(Devicep<Camera>* cam){ this->cam = cam; }
void RayTracing::clearFrame(){ clear = true; }
RayTracing::Container_host& RayTracing::getTree(){ return hostContainer; }

std::vector<RayTracing::BLASInfo> RayTracing::getBLASInfos() const {
    std::vector<BLASInfo> result;
    result.reserve(blasEntries.size());
    for (const auto& entry : blasEntries)
        result.push_back({ &entry.hostTree, entry.toWorld });
    return result;
}

// ─── FrameBuffer / FrameRecord ───────────────────────────────────────────────

struct FrameBuffer {
    vec4f base{0.0f};
    vec4f bloom{0.0f};

    __device__ FrameBuffer& operator+=(const FrameBuffer& other){
        base  += other.base;
        bloom += other.bloom;
        return *this;
    }
};

struct FrameRecord{
    HitRecord   hit;
    FrameBuffer frame;
};

// ─── Kernels ─────────────────────────────────────────────────────────────────

__global__ void initCurandState(size_t width, size_t height, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (int pixel = j * width + i; (i < width) && (j < height))
        curand_init(clock64(), pixel, 0, &randState[pixel]);
}

void RayTracing::create()
{
    const size_t newSize = static_cast<size_t>(width) * height;
    if (record.getSize() == newSize) return;  // size unchanged: preserve accumulated frames

    record     = Buffer<FrameRecord>(newSize);
    baseColor  = Buffer<uint32_t>(newSize);
    bloomColor = Buffer<uint32_t>(newSize);
    randState  = Buffer<curandState>(newSize);

    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);
    initCurandState<<<blocks, threads>>>(width, height, randState.get());

    clear = true;  // force-clear on next render: freshly allocated record is uninitialised
}

void RayTracing::buildTree()
{
    Timer timer("RayTracing::buildTree");

    // Populate TLAS host tree with one fake primitive per object.
    // Do this here (not in bind) so that blasEntries is stable and
    // tlasPrimitive addresses don't move during push_back.
    hostContainer.storage.clear();
    hostContainer.storage.reserve(blasEntries.size());
    for (auto& entry : blasEntries)
        hostContainer.storage.push_back(&entry.tlasPrimitive);

    hostContainer.makeTree();
    timer.elapsedTime("RayTracing::buildTree : make TLAS host tree");

    // Build per-object BLAS on device.
    for (auto& entry : blasEntries) {
        entry.hostTree.makeTree();

        entry.blas = make_devicep<HitableKDTree>(HitableKDTree());
        const auto shapes = extractTriangles(entry.hostTree.storage);
        HitableContainer::add(entry.blas.get(), shapes);

        const std::vector<NodeDescriptor> nodeDescs = entry.hostTree.buildNodeDescriptors();
        Buffer<NodeDescriptor> devNodeDescs(nodeDescs.size(), nodeDescs.data());
        makeTree(entry.blas.get(), devNodeDescs.get(), nodeDescs.size());
    }
    timer.elapsedTime("RayTracing::buildTree : build per-object BLASes");

    // Build flat device TLAS (array of ObjectInstances).
    std::vector<ObjectInstance> hostInstances;
    hostInstances.reserve(blasEntries.size());
    for (const auto& entry : blasEntries) {
        ObjectInstance inst;
        inst.toWorld   = entry.toWorld;
        inst.toLocal   = entry.toLocal;
        inst.worldBbox = entry.worldBbox;
        inst.blas      = entry.blas.get();
        hostInstances.push_back(inst);
    }
    devInstances = Buffer<ObjectInstance>(hostInstances.size(), hostInstances.data());
    timer.elapsedTime("RayTracing::buildTree : upload TLAS instances");
}

// ─── Rendering ───────────────────────────────────────────────────────────────

__device__ bool isEmit(const HitRecord& rec){
    return (rec.rayDepth == 1 && rec.vertex.props.emissionFactor >= 0.98f) ||
           (rec.scattering.getDirection().length2() > 0.0f && rec.lightIntensity >= 0.95f);
}

__device__ FrameBuffer getFrame(
    uint32_t minRayIterations, uint32_t maxRayIterations,
    Camera* cam, float u, float v,
    HitRecord& rec,
    const ObjectInstance* instances, uint32_t numInstances,
    curandState* randState)
{
    FrameBuffer result;
    do {
        ray r = rec.rayDepth++ ? rec.scattering : cam->getPixelRay(u, v, randState);

        HitCoords coords;
        for (uint32_t i = 0; i < numInstances; i++) {
            if (instances[i].worldBbox.intersect(r))
                instances[i].hit(r, coords);
        }

        if (coords.check()) {
            const vec4f color = rec.vertex.color;
            coords.obj->calcHitRecord(r, coords, rec);
            rec.lightIntensity *= rec.vertex.props.absorptionFactor;
            rec.vertex.color = min(
                vec4f(rec.lightIntensity * rec.vertex.color[0],
                      rec.lightIntensity * rec.vertex.color[1],
                      rec.lightIntensity * rec.vertex.color[2],
                      rec.vertex.color[3]),
                color);
        }

        vec4f scattering = scatter(r, rec.vertex.normal, rec.vertex.props, randState);
        if (scattering.length2() == 0.0f || rec.rayDepth >= maxRayIterations) {
            result.base  = rec.vertex.props.emissionFactor >= 0.98f
                               ? rec.vertex.props.emissionFactor * rec.vertex.color
                               : vec4f(0.0f, 0.0f, 0.0f, 1.0f);
            result.bloom = isEmit(rec) ? rec.vertex.color : vec4f(0.0f, 0.0f, 0.0f, 0.0f);
            rec = HitRecord{};
            break;
        }

        rec.scattering = ray(rec.vertex.point, scattering);
    } while (rec.rayDepth < minRayIterations);
    return result;
}

__device__ uint32_t convertVec4ToUint(const vec4f& v){
    vec4f normVec = v / std::max(1.0f, v.a());
    return uint32_t(255.0f * normVec[2]) << 0  |
           uint32_t(255.0f * normVec[1]) << 8  |
           uint32_t(255.0f * normVec[0]) << 16 |
           uint32_t(255) << 24;
}

__global__ void render(
    bool clear,
    size_t width, size_t height,
    size_t minRayIterations, size_t maxRayIterations,
    uint32_t* baseColor, uint32_t* bloomColor,
    FrameRecord* record,
    Camera* cam,
    const ObjectInstance* instances, uint32_t numInstances,
    curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        float u = 1.0f - 2.0f * float(i) / float(width);
        float v = 2.0f * float(j) / float(height) - 1.0f;

        if (clear) record[pixel] = FrameRecord{};

        record[pixel].frame += getFrame(
            minRayIterations, maxRayIterations,
            cam, u, v,
            record[pixel].hit,
            instances, numInstances,
            &randState[pixel]);

        baseColor[pixel]  = convertVec4ToUint(record[pixel].frame.base);
        bloomColor[pixel] = convertVec4ToUint(record[pixel].frame.bloom);
    }
}

__global__ void updateKernel(Camera* cam){ cam->update(); }

void RayTracing::update(){
    updateKernel<<<1, 1>>>(cam->get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

bool RayTracing::calculateImage(uint32_t* hostBaseColor, uint32_t* hostBloomColor)
{
    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);
    render<<<blocks, threads>>>(
        clear,
        width, height,
        minRayIterations, maxRayIterations,
        baseColor.get(), bloomColor.get(),
        record.get(),
        cam->get(),
        devInstances.get(), static_cast<uint32_t>(blasEntries.size()),
        randState.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clear = false;

    checkCudaErrors(cudaMemcpy(hostBaseColor,  baseColor.get(),  sizeof(uint32_t) * baseColor.getSize(),  cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hostBloomColor, bloomColor.get(), sizeof(uint32_t) * bloomColor.getSize(), cudaMemcpyDeviceToHost));

    return true;
}

}
