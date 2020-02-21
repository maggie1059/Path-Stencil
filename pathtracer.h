#ifndef PATHTRACER_H
#define PATHTRACER_H

#include <QImage>

#include "scene/scene.h"
#include <random>
#include <Eigen/Geometry>

#define PI 3.1415926535f


class PathTracer
{
public:
    PathTracer(int width, int height);

    void traceScene(QRgb *imageData, const Scene &scene);

private:
    int m_width, m_height;

    void toneMap(QRgb *imageData, std::vector<Eigen::Vector3f> &intensityValues);

    Eigen::Vector3f tracePixel(int x, int y, const Scene &scene, const Eigen::Matrix4f &invViewMatrix, int n);
    Eigen::Vector3f traceRay(const Ray& r, const Scene &scene, int depth, bool fromBack);

    Eigen::Vector3f sampleNextDir();
    Eigen::Vector3f sampleNextDir2(Eigen::Vector3f normal);
    float continueProb();
    float diffuseBRDF();
    Eigen::Vector3f phongBRDF(Eigen::Vector3f wi, Eigen::Vector3f n, Eigen::Vector3f wo, Eigen::Vector3f s, int exp);
    Eigen::Vector3f specRefractBRDF(Eigen::Vector3f wi, Eigen::Vector3f n, float ior_in, float ior_out);
    float random();
    float clamp(float n, float low, float hi);
    Eigen::Vector3f reflect(Eigen::Vector3f in, Eigen::Vector3f n);
    Eigen::Vector3f directLighting(IntersectionInfo i, Eigen::Vector3f id, const Scene& scene, bool spec, Eigen::Vector3f rayd);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;
};

#endif // PATHTRACER_H
