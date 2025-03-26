// main.cpp
#include "wass/src/wass_stereo/PovMesh.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <algorithm>

// Load .xyz file into PovMesh
void load_xyz_into_povmesh(PovMesh& mesh, const std::string& filename) {
    std::ifstream file(filename);
    double x, y, z;
    for (int v = 0; v < mesh.height(); ++v) {
        for (int u = 0; u < mesh.width(); ++u) {
            if (!(file >> x >> y >> z)) return;
            mesh.set_point(u, v, cv::Vec3d(x, y, z), 255, 255, 255); // white color
        }
    }
}
/*no anda
void load_ply_into_povmesh(PovMesh& mesh, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        return;
    }

    std::string line;
    int vertex_count = 0;
    std::string format;
    bool header_ended = false;

    // Read header
    while (std::getline(file, line)) {
        if (line.rfind("format", 0) == 0) {
            format = line;
        } else if (line.rfind("element vertex", 0) == 0) {
            sscanf(line.c_str(), "element vertex %d", &vertex_count);
        } else if (line == "end_header") {
            header_ended = true;
            break;
        }
    }

    if (!header_ended) {
        std::cerr << "Invalid PLY header\n";
        return;
    }

    if (format.find("binary") != std::string::npos) {
        std::cerr << "Error: This loader only supports ASCII PLY, but file is binary.\n";
        return;
    }

    // Load vertices for ASCII PLY
    double x, y, z;
    int width = mesh.width(), height = mesh.height();
    int u = 0, v = 0, i = 0;

    while (i < vertex_count && file >> x >> y >> z) {
        mesh.set_point(u, v, cv::Vec3d(x, y, z), 255, 255, 255);
        ++u;
        if (u >= width) { u = 0; ++v; }
        if (v >= height) break;
        ++i;
    }
}
*/
int main(int argc, char** argv) {
    int width = 2456;//640;
    int height = 2058;//480;
    std::string filename = "wass/mesh_converted.xyz"; // default

    if (argc > 1) {
        filename = argv[1];
    }

    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    PovMesh mesh(width, height);

    if (ext == "xyz") {
        load_xyz_into_povmesh(mesh, filename);
    } else if (ext == "ply") {
        load_ply_into_povmesh(mesh, filename);
    } else {
        std::cerr << "Unsupported file format: " << ext << "\n";
        return 1;
    }

    bool success = mesh.ransac_find_plane(10000, 0.05);

    if (success) {
        std::vector<double> plane = mesh.get_plane_params();
        std::cout << "Plane equation: " 
                  << plane[0] << "x + " 
                  << plane[1] << "y + "
                  << plane[2] << "z + " 
                  << plane[3] << " = 0\n";
    } else {
        std::cout << "RANSAC failed to find a good plane.\n";
    }

    success = mesh.save_as_xyz_compressed("wass_data/output/000000_wd/mesh_cam_mio.xyzC") ;
    if (success){
        std::cout << "Saved xyzC succesfully .\n";
    }
    else{
        std::cout << "Couldnt save xyzC  .\n";
    }
    
    //Calculate Save xyzC along
    return 0;
}
