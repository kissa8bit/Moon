#include "objmodel.h"

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#endif

#include "utils/texture.h"

#include <unordered_map>

namespace cuda::rayTracing {

ObjModel::ObjModel(const std::filesystem::path& path, const ObjModelInfo& info) :
    path(path),
    info(info)
{}

ObjModel::ObjModel(const std::vector<Vertex>& vertexBuffer, const std::vector<uint32_t>& indexBuffer, Texture& texture)
    : Model(vertexBuffer, indexBuffer, {texture.object}), texture(std::move(texture))
{}

void ObjModel::load(const mat4f& transform)
{
    const auto dir = path.parent_path();

    tinyobj::ObjReader objReader;
    objReader.ParseFromFile(path.string());

    std::vector<std::string> diffuse_texnames;
    for(const auto& material : objReader.GetMaterials()){
        diffuse_texnames.push_back(material.diffuse_texname);
    }

    if(!diffuse_texnames.empty()){
        texture = Texture(dir / diffuse_texnames.front());
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    const auto& attrib = objReader.GetAttrib();
    for (const auto& shape : objReader.GetShapes()) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            if(attrib.vertices.size()){
                vertex.point = transform * vec4f{
                                   attrib.vertices[3 * index.vertex_index + 0],
                                   attrib.vertices[3 * index.vertex_index + 1],
                                   attrib.vertices[3 * index.vertex_index + 2],
                                   1.0f
                               };
            }

            if(attrib.texcoords.size()){
                vertex.u = attrib.texcoords[2 * index.texcoord_index + 0];
                vertex.v = attrib.texcoords[2 * index.texcoord_index + 1];
            }

            if(attrib.normals.size()){
                vertex.normal = transform * vec4f{
                                    attrib.normals[3 * index.normal_index + 0],
                                    attrib.normals[3 * index.normal_index + 1],
                                    attrib.normals[3 * index.normal_index + 2],
                                    0.0f
                                };
            }

            vertex.color = info.color;
            vertex.props = info.props;

            indices.push_back(indices.size());
            vertices.push_back(vertex);
        }
    }

    if(attrib.normals.empty()){
        for(uint32_t i = 0; i < indices.size(); i += 3){
            const vec4f n = normal(cross(
                vertices[indices[i + 1]].point - vertices[indices[i + 0]].point,
                vertices[indices[i + 2]].point - vertices[indices[i + 1]].point
                ));

            vertices[indices[i + 0]].normal += n;
            vertices[indices[i + 1]].normal += n;
            vertices[indices[i + 2]].normal += n;
        }
    }

    std::unordered_map<std::string, vec4f> normalMap;
    if(info.mergeNormals){
        for(auto& vertex : vertices){
            std::string id = std::to_string(vertex.point.x()) + "_" + std::to_string(vertex.point.y()) + "_" + std::to_string(vertex.point.z());
            if(normalMap.count(id)){
                normalMap[id] += vertex.normal;
            } else {
                normalMap[id] = vertex.normal;
            }
        }
    }

    for(auto& vertex : vertices){
        if(info.mergeNormals){
            std::string id = std::to_string(vertex.point.x()) + "_" + std::to_string(vertex.point.y()) + "_" + std::to_string(vertex.point.z());
            vertex.normal = normal(normalMap[id]);
        } else {
            vertex.normal = normal(vertex.normal);
        }
    }

    *this = ObjModel(vertices, indices, texture);
}

}
