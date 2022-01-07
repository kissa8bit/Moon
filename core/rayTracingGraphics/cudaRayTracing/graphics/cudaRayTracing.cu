#include "cudaRayTracing.h"
#include "operations.h"
#include "ray.h"
#include "material.h"
#include "utils/timer.h"

namespace cuda::rayTracing {

void RayTracing::setExtent(uint32_t width, uint32_t height){
    this->width = width;
    this->height = height;
}

void RayTracing::bind(Object* obj) {
    obj->model->load(obj->transform);
    for(const auto& primitive : obj->model->primitives){
        hostContainer.storage.push_back(&primitive);
    }
}

void RayTracing::setCamera(Devicep<Camera>* cam){ this->cam = cam;}
void RayTracing::clearFrame(){clear = true;}
KDTree<std::vector<const Primitive*>>& RayTracing::getTree(){return hostContainer;}

struct FrameBuffer {
    vec4f base{0.0f};
    vec4f bloom{0.0f};

    __device__ FrameBuffer& operator+=(const FrameBuffer& other){
        base += other.base;
        bloom += other.bloom;
        return *this;
    }
};

struct FrameRecord{
    HitRecord hit;
    FrameBuffer frame;
};

__global__ void initCurandState(size_t width, size_t height, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        curand_init(clock64(), pixel, 0, &randState[pixel]);
    }
}

void RayTracing::create()
{
    record = Buffer<FrameRecord>(width * height);
    baseColor = Buffer<uint32_t>(width * height);
    bloomColor = Buffer<uint32_t>(width * height);
    randState = Buffer<curandState>(width * height);

    dim3 blocks(width / xThreads + 1, height / yThreads + 1, 1);
    dim3 threads(xThreads, yThreads, 1);
    initCurandState<<<blocks, threads>>>(width, height, randState.get());
}

void RayTracing::buildTree(){
    Timer timer("RayTracing::buildTree");

    hostContainer.makeTree();

    timer.elapsedTime("RayTracing::buildTree : make host tree");

    devContainer = make_devicep<Container_dev>(Container_dev());
    const auto hitables = extractHitables(hostContainer.storage);
    HitableContainer::add(devContainer, hitables);

    timer.elapsedTime("RayTracing::buildTree : copy hitables to device");

    if(std::is_same<Container_dev, HitableKDTree>::value){
        const std::vector<NodeDescriptor> nodeDescriptors = hostContainer.buildNodeDescriptors();
        Buffer<NodeDescriptor> devNodeDescriptors(nodeDescriptors.size(), (NodeDescriptor*) nodeDescriptors.data());
        makeTree((HitableKDTree*)devContainer.get(), devNodeDescriptors.get(), nodeDescriptors.size());

        timer.elapsedTime("RayTracing::buildTree : make device tree");
    }
}

__device__ bool isEmit(const HitRecord& rec){
    return (rec.rayDepth == 1 && rec.vertex.props.emissionFactor >= 0.98f) || (rec.scattering.getDirection().length2() > 0.0f && rec.lightIntensity >= 0.95f);
}

template<typename ContainerType>
__device__ FrameBuffer getFrame(uint32_t minRayIterations, uint32_t maxRayIterations, Camera* cam, float u, float v, HitRecord& rec, ContainerType* container, curandState* randState) {
    FrameBuffer result;
    do {
        ray r = rec.rayDepth++ ? rec.scattering : cam->getPixelRay(u, v, randState);
        if (HitCoords coords; container->hit(r, coords)) {
            if(vec4 color = rec.vertex.color; coords.check()){
                coords.obj->calcHitRecord(r, coords, rec);
                rec.lightIntensity *= rec.vertex.props.absorptionFactor;
                rec.vertex.color = min(
                    vec4f(rec.lightIntensity * rec.vertex.color[0],
                          rec.lightIntensity * rec.vertex.color[1],
                          rec.lightIntensity * rec.vertex.color[2],
                          rec.vertex.color[3]),
                    color);
            }
        }

        vec4f scattering = scatter(r, rec.vertex.normal, rec.vertex.props, randState);
        if(scattering.length2() == 0.0f || rec.rayDepth >= maxRayIterations){
            result.base = rec.vertex.props.emissionFactor >= 0.98f ? rec.vertex.props.emissionFactor * rec.vertex.color : vec4f(0.0f, 0.0f, 0.0f, 1.0f);
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

template <typename ContainerType>
__global__ void render(bool clear, size_t width, size_t height, size_t minRayIterations, size_t maxRayIterations, uint32_t* baseColor, uint32_t* bloomColor, FrameRecord* record, Camera* cam, ContainerType* container, curandState* randState)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (int pixel = j * width + i; (i < width) && (j < height)) {
        float u = 1.0f - 2.0f * float(i) / float(width);
        float v = 2.0f * float(j) / float(height) - 1.0f;

        if(clear) record[pixel] = FrameRecord{};

        record[pixel].frame += getFrame(minRayIterations, maxRayIterations, cam, u, v, record[pixel].hit, container, &randState[pixel]);

        baseColor[pixel] = convertVec4ToUint(record[pixel].frame.base);
        bloomColor[pixel] = convertVec4ToUint(record[pixel].frame.bloom);
    }
}

__global__ void updateKernel(Camera* cam){
    cam->update();
}

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
        width,
        height,
        minRayIterations,
        maxRayIterations,
        baseColor.get(),
        bloomColor.get(),
        record.get(),
        cam->get(),
        devContainer.get(),
        randState.get());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    clear = false;

    checkCudaErrors(cudaMemcpy(hostBaseColor, baseColor.get(), sizeof(uint32_t) * baseColor.getSize(), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hostBloomColor, bloomColor.get(), sizeof(uint32_t) * bloomColor.getSize(), cudaMemcpyDeviceToHost));

    return true;
}

}
