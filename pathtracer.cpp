#include "pathtracer.h"

#include <iostream>

#include <Eigen/Dense>
#include <algorithm>
#include <util/CS123Common.h>
#include <random>
#include <math.h>

typedef Eigen::Affine3d Transformation;
typedef Eigen::Vector3d Point;
typedef Eigen::Vector3d Vector;
typedef Eigen::Translation<double,3> Translation;

using namespace Eigen;

PathTracer::PathTracer(int width, int height)
    : m_width(width), m_height(height), distribution(0.0,1.0)
{
}

void PathTracer::traceScene(QRgb *imageData, const Scene& scene)
{
    std::vector<Vector3f> intensityValues(m_width * m_height);
    Matrix4f invViewMat = (scene.getCamera().getScaleMatrix() * scene.getCamera().getViewMatrix()).inverse();
    for(int y = 0; y < m_height; ++y) {
        //#pragma omp parallel for
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            intensityValues[offset] = tracePixel(x, y, scene, invViewMat, 100);
        }
    }

    toneMap(imageData, intensityValues);
}

Vector3f PathTracer::tracePixel(int x, int y, const Scene& scene, const Matrix4f &invViewMatrix, int n)
{
    Vector3f p(0, 0, 0);
    Vector3f d((2.f * x / m_width) - 1, 1 - (2.f * y / m_height), -1);
    d.normalize();

    Ray r(p, d);
    r = r.transform(invViewMatrix);
    Vector3f out(0,0,0);
    for (int i = 0; i < n; i++){
        out += traceRay(r, scene, 0);
        // reset direction and redefine ray
        d = sampleNextDir();
//        d.normalize();
        // create new ray and use change of basis
        r.o = p;
        r.d = d;
//        r = r.transform(invViewMatrix);
    }
    out = out/n;
    return out;
}

Vector3f PathTracer::sampleNextDir(){
//    std::default_random_engine generator;
//    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double num1 = distribution(generator);
    double num2 = distribution(generator);
    float phi = 2.0f * PI * num1;
    float theta = acos(1.0f-num2);
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    Vector3f final(x,y,z);
    final.normalize();
    return final;
}

Vector3f PathTracer::traceRay(const Ray& r, const Scene& scene, int depth)
{
    IntersectionInfo i;
    Ray ray(r);
    Vector3f L(0, 0, 0);
    if(scene.getIntersection(ray, &i)) {
          //** Example code for accessing materials provided by a .mtl file **
        const Triangle *t = static_cast<const Triangle *>(i.data); //Get the triangle in the mesh that was intersected
        const tinyobj::material_t& mat = t->getMaterial(); //Get the material of the triangle from the mesh
        const tinyobj::real_t *d = mat.diffuse; //Diffuse color as array of floats
        const tinyobj::real_t *e = mat.emission; //Diffuse color as array of floats
//        const std::string diffuseTex = mat.diffuse_texname;//Diffuse texture name
        Vector3f d_vec(d[0], d[1], d[2]);
        const Vector3f normal = t->getNormal(i.hit);

//        L = directLighting(i.hit, ray.inv_d);
        L = Vector3f(e[0], e[1], e[2]);//p.emitted(-w); //p is intersection, -w is ray.inv_d
        float pdf_rr = 0.8; //continueProb();
        if (random() < pdf_rr){
            Vector3f wi = sampleNextDir();
            float pdf = 1.0/(2.0*PI);
            ray.o = i.hit;
            ray.d = wi;
            Vector3f val = traceRay(ray, scene, depth+1) * diffuseBRDF(d_vec) * clamp(wi.dot(normal), 0.0f, 1.0f)/ (pdf * pdf_rr);
            val.array() *= d_vec.array();
            L += val;
        }
//        if (depth == 0){
//            L += p.emitted(-w);
//        }
        return L;
    } else {
        return L;
    }
}

float PathTracer::clamp(float n, float low, float hi){
    if (n > hi){
        n = hi;
    }
    else if (n < low){
        n = low;
    }
    return n;
}

float PathTracer::diffuseBRDF(Vector3f d){
    float num = 1.0/PI;
    return num;
}

float PathTracer::random(){
    float num = distribution(generator);
    return num;
}

Transformation findTransformBetween2CS(Point fr0,Point fr1,Point fr2,Point to0,Point to1,Point to2) {

  Transformation T, T2, T3 = Transformation::Identity();
  Vector3d x1,y1,z1, x2,y2,z2;

  // Axes of the coordinate system "fr"
  x1 = (fr1 - fr0).normalized(); // the versor (unitary vector) of the (fr1-fr0) axis vector
  y1 = (fr2 - fr0).normalized();

  // Axes of the coordinate system "to"
  x2 = (to1 - to0).normalized();
  y2 = (to2 - to0).normalized();

  // transform from CS1 to CS2
  // Note: if fr0==(0,0,0) --> CS1==CS2 --> T2=Identity
  T2.linear() << x1, y1, x1.cross(y1);

  // transform from CS1 to CS3
  T3.linear() << x2, y2, x2.cross(y2);

  // T = transform to CS2 to CS3
  // Note: if CS1==CS2 --> T = T3
  T.linear() = T3.linear() * T2.linear().inverse();


  T.translation() = t0;

  return T;
}

void PathTracer::toneMap(QRgb *imageData, std::vector<Vector3f> &intensityValues) {
    for(int y = 0; y < m_height; ++y) {
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            int r = 255 * (intensityValues[offset][0] / (1.f + intensityValues[offset][0]));
            int g = 255 * (intensityValues[offset][1] / (1.f + intensityValues[offset][1]));
            int b = 255 * (intensityValues[offset][2] / (1.f + intensityValues[offset][2]));
            imageData[offset] = qRgb(r,g,b);
        }
    }

}
