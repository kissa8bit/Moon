#ifndef CUDARAYTRACING
#define CUDARAYTRACING

#include <cudaRayTracing/transformational/camera.h>
#include <cudaRayTracing/transformational/object.h>
#include <cudaRayTracing/utils/buffer.h>
#include <cudaRayTracing/utils/kdTree.h>
#include <cudaRayTracing/hitable/objectInstance.h>

#include <stdint.h>
#include <vector>

namespace cuda::rayTracing {

struct FrameRecord;

class RayTracing {
public:
    // Host-side KD tree used for TLAS construction and bounding-box visualisation.
    // After buildTree() its storage holds one fake Primitive per bound object
    // (bbox == world-space bbox of that object).
    using Container_host = KDTree<std::vector<const Primitive*>>;

private:
    uint32_t width{0};
    uint32_t height{0};
    Buffer<FrameRecord> record;
    Buffer<uint32_t>    baseColor;
    Buffer<uint32_t>    bloomColor;
    Buffer<curandState> randState;

    uint32_t xThreads{16};
    uint32_t yThreads{16};
    uint32_t minRayIterations{2};
    uint32_t maxRayIterations{12};

    bool clear{false};

    Devicep<Camera>* cam{nullptr};

    // Per-object data collected during bind() calls.
    struct BLASEntry {
        Container_host      hostTree;       // local-space primitives for this object
        Devicep<HitableKDTree> blas;        // per-object device BLAS
        mat4f               toWorld;
        mat4f               toLocal;
        box                 worldBbox;
        Primitive           tlasPrimitive;  // fake Primitive with worldBbox, used by hostContainer

        BLASEntry()                    = default;
        BLASEntry(BLASEntry&&)         = default;
        BLASEntry& operator=(BLASEntry&&) = default;
    };
    std::vector<BLASEntry> blasEntries;

    // TLAS: one fake Primitive per object on the host (for visualisation),
    // flat ObjectInstance array on the device (for rendering).
    Container_host          hostContainer;
    Buffer<ObjectInstance>  devInstances;

public:
    void setExtent(uint32_t width, uint32_t height);
    void bind(Object* obj);
    void setCamera(Devicep<Camera>* cam);

    void create();
    void update();

    bool calculateImage(uint32_t* baseColor, uint32_t* bloomColor);
    void clearFrame();
    void buildTree();

    // Returns the TLAS host tree (one node per bound object).
    // Used by RayTracingGraphics::buildBoundingBoxes().
    Container_host& getTree();

    // Per-object BLAS info for bounding-box visualisation.
    // Uses the underlying KDTree type to avoid requiring Container_host qualification
    // at call sites outside this class.
    using HostTree = KDTree<std::vector<const Primitive*>>;
    struct BLASInfo {
        const HostTree* tree;
        mat4f toWorld;
    };
    std::vector<BLASInfo> getBLASInfos() const;
};

}

#endif // CUDARAYTRACING
