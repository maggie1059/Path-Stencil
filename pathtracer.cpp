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
    for (int i = 0; i < 1; i++){
        out += traceRay(r, scene, 0);
        // reset direction and redefine ray
//        double num1 = distribution(generator);
//        double num2 = distribution(generator);
//        Vector3f d(((2.f+num1) * x / m_width) - 1, 1 - ((2.f+num2) * y / m_height), -1);
//        d = sampleNextDir();
//        d.normalize();
        // create new ray and use change of basis
//        r.o = p;
//        r.d = d;
//        r = Ray(r.o, d);
//        r = r.transform(invViewMatrix);
    }
    out = out/n;
    return out;
}

Vector3f PathTracer::sampleNextDir(){
    double num1 = distribution(generator);
    double num2 = distribution(generator);
    float phi = 2.0f * PI * num1;
    float theta = acos(1.0f-num2);
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    Vector3f final(x,y,z);
    final.normalize();
//    Vector3f d(((2.f+num1) * x / m_width) - 1, 1 - ((2.f+num2) * y / m_height), -1);
    return final;
}

Vector3f PathTracer::sampleNextDir2(Vector3f normal){
    double num1 = distribution(generator);
    double num2 = distribution(generator);
    float phi = 2.0f * PI * num1;
    float theta = acos(1.0f-num2);

    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);
    Vector3f final(x,y,z);
//    std::cout << normal[0] << " "<< normal[1] << " "<< normal[2] << std::endl;
    final.normalize();
    Matrix3f a;
    Vector3f zaxis(0,0,1);
//    std::cout << zaxis.dot(final) << " "<< std::endl;
    Quaternionf q = Quaternionf::FromTwoVectors(zaxis, normal);

    final = q*final;
    final.normalize();
//    std::cout << normal.dot(final) << " "<< std::endl;
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
        L = directLighting(i, normal, scene);
        L.array() *= d_vec.array();
//        L = Vector3f(e[0], e[1], e[2]);//p.emitted(-w); //p is intersection, -w is ray.inv_d
        float pdf_rr = 0.8; //continueProb();
        if (random() < pdf_rr){

            Vector3f wi = sampleNextDir2(normal);
//            std::cout << "normal: " << normal[0] << " " << normal[1] << " "<< normal[2] << std::endl;
//            std::cout << "dir: " << wi[0] << " " << wi[1] << " "<< wi[2] << std::endl;
            float pdf = 1.0/(2.0*PI);
            Ray ray(i.hit, wi);
            Vector3f val;
//            switch(mat->illum){

//            }

            val = traceRay(ray, scene, depth+1) * diffuseBRDF() * clamp(wi.dot(normal), 0.0f, 1.0f)/ (pdf * pdf_rr);
            //for mirror:
            //Ray ray(i.hit, reflect(ray.inv_d, normal);
            //Vector3f val = traceRay(ray, scene, depth+1) / pdf_rr;
            val.array() *= d_vec.array();

            L += val;
        }
        if (depth == 0){
            L += Vector3f(e[0], e[1], e[2]); //p.emitted(-w);
        }
        return L;
    } else {
        return L;
    }
}

Vector3f PathTracer::directLighting(IntersectionInfo i, Vector3f normal, const Scene& scene){
    Vector3f p = i.hit;
    Vector3f light(0,0,0);
    std::vector<Triangle *> lights = scene._light_triangles;
    float total_area = 0;
    float num = distribution(generator);
    num = floor(num*lights.size());
    static_cast<int>(num);
    for (int i = 0; i < lights.size(); i++){
        Triangle *triangle = lights[i];
        total_area += triangle->getArea();
    }
    Triangle *sample = lights[num];
    //randomly sample point from sample
    float r1 = random();
    float r2 = random();
    Vector3f A = sample->_v1;
    Vector3f B = sample->_v2;
    Vector3f C = sample->_v3;
    Vector3f em = (1.f-sqrt(r1))*A + (sqrt(r1)*(1.f-r2)*B) + (r2*sqrt(r1)*C);
    Vector3f newdir = em-p;
    newdir.normalize();
    IntersectionInfo new_i;
    Ray ray(p, newdir);

    Vector3f t_norm = sample->getNormal(i);

    float c_theta = newdir.dot(normal);
    float c_p_theta = (-newdir).dot(t_norm);

    if (c_theta < 0){
        return Vector3f(0,0,0);
    }
    else if (c_p_theta < 0){
        return Vector3f(0,0,0);
    }
    else if(scene.getIntersection(ray, &new_i)) {
        const Triangle *hit = static_cast<const Triangle *>(new_i.data);
        if (hit->getIndex() == sample->getIndex()){
            const tinyobj::material_t& mat = sample->getMaterial();
            const tinyobj::real_t *e = mat.emission;
            light += Vector3f(e[0], e[1], e[2])*(c_theta*c_p_theta)/(pow((p-new_i.hit).norm(), 2));
        }
    }
    float pdf = 1.f/total_area;
    return light/pdf;
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

Vector3f PathTracer::reflect(Vector3f in, Vector3f n){
    Vector3f out = in - 2*in.dot(n)*n;
    return out;
}

float PathTracer::diffuseBRDF(){
    float num = 1.0/PI;
    return num;
}

float PathTracer::phongBRDF(Vector3f wi, Vector3f n, Vector3f wo, int s){
    Vector3f reflect = wi - 2*wi.dot(n)*n;
    float dotted = reflect.dot(wo);
    float num = (s+2)/(2*PI) * dotted * dotted;
    return num;
}

float PathTracer::specRefractBRDF(Vector3f wi, Vector3f n){
//    float ni = 1.0; //air
//    float nt = 1.55; //glass
//    wi.normalize();
//    float theta_i = acos(wi.dot(n));
//    float c_theta_i = wi.dot(n);
//    //if ni < nt
//    float ni_nt = ni/nt;
//    float cos_theta_t = sqrt(1 - (ni_nt*ni_nt)*(1-(pow(c_theta_i, 2))));
//    Vector3f wt = (ni_nt)*wi + (ni_nt*c_theta_i - cos_theta_t*n);
//    Ray ray(i.hit, wt);
//    Vector3f val = traceRay(ray, scene, depth+1) / pdf_rr;
//    float refract = (1 - wt.dot(wo))/cos(theta); //only if in direction wt
//    float fres = (ni-nt)/(ni+nt);
//    float R0 = pow(fres, 2);
//    float Rti = R0 + ((1-R0)*pow(1-c_theta_i, 5.0));
//    //reflect*Rti + refract(1-Rti)
//    return Rti;
    return 0;
}

float PathTracer::random(){
    float num = distribution(generator);
    return num;
}

void PathTracer::toneMap(QRgb *imageData, std::vector<Vector3f> &intensityValues) {
    for(int y = 0; y < m_height; ++y) {
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            float r = 255.f * ((float)intensityValues[offset][0]*100 / (1.f + (float)intensityValues[offset][0]*100));
            float g = 255.f * ((float)intensityValues[offset][1]*100 / (1.f + (float)intensityValues[offset][1]*100));
            float b = 255.f * ((float)intensityValues[offset][2]*100 / (1.f + (float)intensityValues[offset][2]*100));
            imageData[offset] = qRgb(r,g,b);
        }
    }
}
