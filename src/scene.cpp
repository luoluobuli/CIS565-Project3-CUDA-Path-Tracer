#include "scene.h"

#include "utilities.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include <fstream>
#include <iostream>

#include <unordered_map>
#include <glm/gtc/type_ptr.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_IMPLEMENTION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <stb_image.h>
#include "tiny_gltf.h"

using namespace std;
using json = nlohmann::json;

Scene::Scene(std::vector<string> filenames)
{
    for (string filename : filenames) {
        cout << "Reading scene from " << filename << " ..." << endl;
        cout << " " << endl;
        auto ext = filename.substr(filename.find_last_of('.'));
        if (ext == ".json")
        {
            loadFromJSON(filename);
            continue;
        }
        else if (ext == ".gltf") {
            loadFromGLTF(filename);
            //initializeScene();
            continue;
        }
        else
        {
            cout << "Couldn't read from " << filename << endl;
            exit(-1);
        }
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
            float roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.f - roughness;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

static void processNode(std::vector<Geom>& geoms, int nodeIndex, const tinygltf::Model& model, const glm::mat4& parentTransform)
{
    const tinygltf::Node& node = model.nodes[nodeIndex];
    
    // Get transformation matrix
    glm::mat4 local(1.f);
    if (node.matrix.size() == 16) {
        local = glm::make_mat4(node.matrix.data());
    }
    else {
        glm::vec3 translation(0.0f);
        if (node.translation.size() == 3)
            translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);

        glm::vec3 scale(1.0f);
        if (node.scale.size() == 3)
            scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);

        glm::quat rotation(1, 0, 0, 0);
        if (node.rotation.size() == 4)
            rotation = glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);

        local = glm::translate(glm::mat4(1.0f), translation)
            * glm::mat4_cast(rotation)
            * glm::scale(glm::mat4(1.0f), scale);
    }

    // Hardcoded waterbottle transformation!!
    local = glm::translate(local, glm::vec3(0.f, 3.f, 0.f)); 
    local = glm::scale(local, glm::vec3(10.f));
    local = glm::rotate(local, glm::radians(80.f), glm::vec3(0, 1, 0));
    local = glm::rotate(local, glm::radians(45.f), glm::vec3(0, 0, 1));

    // Get the world transformation
    glm::mat4 world = parentTransform * local; 
    
    if (node.mesh >= 0 && node.mesh < model.meshes.size()) {

        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (const auto& prim : mesh.primitives) {
            Geom tri;
            tri.type = TRIANGLE;
            tri.transform = world;
            tri.inverseTransform = glm::inverse(world);
            tri.invTranspose = glm::inverseTranspose(world);

            // Material Index
            int materialId = prim.material;
            tri.materialid = materialId >= 0 ? materialId : -1;

            // Get indices accessor
            const tinygltf::Accessor& idxAcc = model.accessors[prim.indices];
            const tinygltf::BufferView& idxView = model.bufferViews[idxAcc.bufferView];
            const tinygltf::Buffer& idxBuf = model.buffers[idxView.buffer];
            unsigned short* indices = (unsigned short*)&idxBuf.data[idxView.byteOffset + idxAcc.byteOffset];

            // Get positions accessor
            const tinygltf::Accessor& posAcc = model.accessors[prim.attributes.find("POSITION")->second];
            const tinygltf::BufferView& posView = model.bufferViews[posAcc.bufferView];
            const tinygltf::Buffer& posBuf = model.buffers[posView.buffer];
            float* positions = (float*)&posBuf.data[posView.byteOffset + posAcc.byteOffset];

            // Get normals accessor
            const tinygltf::Accessor& normAcc = model.accessors[prim.attributes.find("NORMAL")->second];
            const tinygltf::BufferView& normView = model.bufferViews[normAcc.bufferView];
            const tinygltf::Buffer& normBuf = model.buffers[normView.buffer];
            float* normals = (float*)&normBuf.data[normView.byteOffset + normAcc.byteOffset];

            // Get texcoord accessor
            auto texIt = prim.attributes.find("TEXCOORD_0");
            bool hasTex = texIt != prim.attributes.end();
            const auto& texAcc = model.accessors[texIt->second];
            const auto& texView = model.bufferViews[texAcc.bufferView];
            const auto& texBuf = model.buffers[texView.buffer];
            float* texcoords = (float*)&texBuf.data[texView.byteOffset + texAcc.byteOffset];

            // Store triangles
            for (size_t i = 0; i < idxAcc.count; i += 3) {
                unsigned short i0 = indices[i + 0];
                unsigned short i1 = indices[i + 1];
                unsigned short i2 = indices[i + 2];

                // Positions
                glm::vec3 p0(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
                glm::vec3 p1(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);
                glm::vec3 p2(positions[3 * i2 + 0], positions[3 * i2 + 1], positions[3 * i2 + 2]);

                tri.v0.position = p0;
                tri.v1.position = p1;
                tri.v2.position = p2;

                // Normals
                glm::vec3 n0(normals[3 * i0 + 0], normals[3 * i0 + 1], normals[3 * i0 + 2]);
                glm::vec3 n1(normals[3 * i1 + 0], normals[3 * i1 + 1], normals[3 * i1 + 2]);
                glm::vec3 n2(normals[3 * i2 + 0], normals[3 * i2 + 1], normals[3 * i2 + 2]);

                tri.v0.normal = n0;
                tri.v1.normal = n1;
                tri.v2.normal = n2;

                // Texcoord
                if (hasTex) {
                    tri.v0.uv = glm::vec2(texcoords[2 * i0 + 0], texcoords[2 * i0 + 1]);
                    tri.v1.uv = glm::vec2(texcoords[2 * i1 + 0], texcoords[2 * i1 + 1]);
                    tri.v2.uv = glm::vec2(texcoords[2 * i2 + 0], texcoords[2 * i2 + 1]);
                }

                geoms.push_back(tri);
            }
        }
    }
    
    for (int child : node.children) {
        processNode(geoms, child, model, world);
    }
}

void Scene::loadFromGLTF(const std::string& filename) {
    // Load gltf file
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    if (!loader.LoadASCIIFromFile(&model, &err, &warn, filename)) {
        std::cout << "Failed to load glTF: " << filename << std::endl;
        return;
    }
    else {
        std::cout << "Loaded glTF: " << filename << std::endl;
    }

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "ERR: " << err << std::endl;
    }

    // Store geometries from nodes
    for (int rootNode : model.scenes[model.defaultScene].nodes) {
        processNode(geoms, rootNode, model, glm::mat4(1.0f));
    }

    // Store materials
    // All material list: model.materials
    // Each material: model.materials.pbrMetallicRoughness
    // Base color: model.materials.pbrMetallicRoughness.baseColorFactor
    std::cout << std::endl << "============ Materials ==============" << std::endl;
    for (const auto& material : model.materials) {
        Material m;
        const auto& pbr = material.pbrMetallicRoughness;
        m.baseColor = glm::vec3((float)pbr.baseColorFactor[0], (float)pbr.baseColorFactor[1], (float)pbr.baseColorFactor[2]);
        //m.roughness = (float)pbr.roughnessFactor;
        m.roughness = 0.5f;
        m.metallic = (float)pbr.metallicFactor;
        m.emittance = (float)material.emissiveFactor[0];
        
        if (pbr.baseColorTexture.index >= 0) {
            const auto& baseTex = model.textures[pbr.baseColorTexture.index];
            const auto& ormTex = model.textures[pbr.metallicRoughnessTexture.index];
            m.diffuseId = baseTex.source;
            m.roughMetalId = ormTex.source;
        }

        if (material.normalTexture.index >= 0) {
            const auto& normalTex = model.textures[material.normalTexture.index];
            m.normalId = normalTex.source;
        }

        std::cout << "Material: " << material.name << std::endl;
        printf("Base color: (%f, %f, %f)\n", m.baseColor.r, m.baseColor.g, m.baseColor.b);
        printf("Roughness: %f\n", m.roughness);
        printf("Metallic: %f\n", m.metallic);
        printf("Emittance: %f\n", m.emittance);
        printf("Reflection: %f\n", m.hasReflective);
        printf("Refraction: %f\n", m.hasRefractive);
        printf("Base color texture id: %i\n", m.diffuseId);
        printf("Roughness/Metallic texture id: %i\n", m.roughMetalId);
        printf("Normal texture id: %i\n", m.normalId);

        materials.push_back(std::move(m));
    }

    // Store textures
    std::cout << std::endl << "============ Textures ==============" << std::endl;
    for (int i = 0; i < model.images.size(); ++i) {
    //for (int i = 0; i < 2; ++i) {
        const auto& image = model.images[i];
        std::cout << "Texture " << i << ": " << image.uri << std::endl;

        int size = image.width * image.height;

        Texture tex;
        tex.pixels.resize(size);
        for (int j = 0; j < size; ++j) {
            float r = image.image[j * image.component + 0] / 255.f; // image.component: the channels of the iamge (3 for RGB or 4 for RGBA)
            float g = image.image[j * image.component + 1] / 255.f;
            float b = image.image[j * image.component + 2] / 255.f;
            float a = (image.component == 4) ? (image.image[j * image.component + 3] / 255.f) : 1.f;

            tex.pixels[j] = make_float4(r, g, b, a); // use float4 instead of glm::vec4 to match cuda settings when passing data
        }

        tex.width = image.width;
        tex.height = image.height;
        textures.push_back(std::move(tex));
    }
    std::cout << std::endl;
}

void Scene::initializeScene() {
    // Manually add a light
    Geom tri1, tri2;
    Material m;

    glm::vec3 p0(-1.f, 5.f, -1.f);
    glm::vec3 p1(-1.f, 5.f, 1.f);
    glm::vec3 p2(1.f, 5.f, 1.f);
    glm::vec3 p3(1.f, 5.f, -1.f);
    glm::vec3 n(0.f, -1.f, 0.f);

    m.baseColor = glm::vec3(1.f);
    m.emittance = 5.f;
    int matId = materials.size();

    tri1.type = TRIANGLE;
    tri1.v0 = { p0, n };
    tri1.v1 = { p1, n };
    tri1.v2 = { p2, n };
    tri1.materialid = matId;

    tri2.type = TRIANGLE;
    tri2.v0 = { p2, n };
    tri2.v1 = { p3, n };
    tri2.v2 = { p0, n };
    tri2.materialid = matId;

    geoms.push_back(tri1);
    geoms.push_back(tri2);
    materials.push_back(m);

    // Add camera
    Camera& camera = state.camera;
    RenderState& state = this->state;

    camera.resolution = glm::ivec2(800, 800);
    camera.position = glm::vec3(0.0, 0.0, 3.0);
    camera.lookAt = glm::vec3(0.0, 0.0, 0.0);
    camera.up = glm::vec3(0.0, 1.0, 0.0);
    
    float fovy = 45.f;
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // Set up rendering state
    state.iterations = 5000;
    state.traceDepth = 2;
    state.imageName = "test";

    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}