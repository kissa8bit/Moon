#ifndef OBJMODEL_H
#define OBJMODEL_H
#include "model.h"

#include <filesystem>
#include <vector>

#include "utils/texture.h"

namespace cuda::rayTracing {

struct ObjModelInfo{
    Properties props{};
    vec4f color{0.0f};
    bool mergeNormals{false};

    ObjModelInfo(
        Properties props = {},
        vec4f color = {0.0f},
        bool mergeNormals = false) :
        props(props), color(color), mergeNormals(mergeNormals)
    {}
};

class ObjModel : public Model {
private:
    std::filesystem::path path;
    ObjModelInfo info;

    Texture texture;

public:
    ObjModel(const std::filesystem::path& path, const ObjModelInfo& info = {});
    ObjModel(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer, Texture& texture);

    void load(const mat4f& transform) override ;
};

}
#endif // OBJMODEL_H
