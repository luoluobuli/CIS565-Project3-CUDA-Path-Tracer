#pragma once

#include "sceneStructs.h"
#include <vector>
#include <string>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& filename);
    void initializeScene();

public:
    Scene(std::vector<std::string> filenames);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    RenderState state;
};
