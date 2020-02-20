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
            intensityValues[offset] = tracePixel(x, y, scene, invViewMat, 10);
        }
    }

    toneMap(imageData, intensityValues);
}

Vector3f PathTracer::tracePixel(int x, int y, const Scene& scene, const Matrix4f &invViewMatrix, int n)
{
    Vector3f p(0, 0, 0);
    Vector3f d((2.f * x / m_width)  - 1, 1 - (2.f * y / m_height), -1);

    d.normalize();

    Ray r(p, d);
    r = r.transform(invViewMatrix);
    Vector3f out(0,0,0);
    for (int i = 0; i < n; i++){
        out += traceRay(r, scene, 0);
        // reset direction and redefine ray
        double num1 = distribution(generator);
        double num2 = distribution(generator);
        double offsetx = num1 * 2.f / m_width;
        double offsety = num2 * (-2.f) / m_height;
        Vector3f d((2.f * x / m_width)  - 1 + offsetx, 1 - (2.f * y / m_height) + offsety, -1);
        d.normalize();
        // create new ray
        r = Ray(p, d);
        r = r.transform(invViewMatrix);
    }
    out = out/n;
    return out;
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
    final.normalize();
    Vector3f zaxis(0,0,1);
    Quaternionf q = Quaternionf::FromTwoVectors(zaxis, normal);

    final = q*final;
    final.normalize();
    return final;
}

Vector3f PathTracer::traceRay(const Ray& r, const Scene& scene, int depth)
{
    IntersectionInfo i;
    Ray ray(r);
    Vector3f L(0, 0, 0);
    if(scene.getIntersection(ray, &i)) {
        const Triangle *t = static_cast<const Triangle *>(i.data); //Get the triangle in the mesh that was intersected
        const tinyobj::material_t& mat = t->getMaterial(); //Get the material of the triangle from the mesh
        const tinyobj::real_t *d = mat.diffuse; //Diffuse color as array of floats
        const tinyobj::real_t *e = mat.emission; //Diffuse color as array of floats
        const tinyobj::real_t *s = mat.specular; //Diffuse color as array of floats

//        const std::string diffuseTex = mat.diffuse_texname;//Diffuse texture name
        Vector3f d_vec(d[0], d[1], d[2]);
        Vector3f s_vec(s[0], s[1], s[2]);
        float pdf = 1.0/(2.0*PI);
        const Vector3f normal = t->getNormal(i.hit);

        L = directLighting(i, normal, scene);
        L.array() *= d_vec.array();

        /*L = Vector3f(d[0], d[1], d[2]);*///p.emitted(-w); //p is intersection, -w is ray.inv_d
        float pdf_rr = 0.8; //continueProb();
        if (random() < pdf_rr){

            Vector3f wi = sampleNextDir2(normal);
//            std::cout << "d: " << ray.d[0] << " " << ray.d[1] << " "<< ray.d[2] << std::endl;
//            std::cout << "-d: " << -ray.d[0] << " " << -ray.d[1] << " "<< -ray.d[2] << std::endl;
//            std::cout << "normal: " << normal[0] << " " << normal[1] << " "<< normal[2] << std::endl;

            Vector3f val;
            switch(mat.illum){
                case 2:
                    {
                        if (s_vec != Vector3f(0,0,0)){
                            //phong
                            Ray rayp(i.hit, wi);
                            Vector3f next = traceRay(rayp, scene, depth+1);
                            Vector3f brdf = phongBRDF(ray.d, normal, wi, s_vec[0])*d_vec;
                            val = next.array() * brdf.array() * clamp(wi.dot(normal), 0.0f, 1.0f)/ (pdf * pdf_rr);
                            break;
                        } else {
                            //diffuse
                            Ray rayd(i.hit, wi);
                            Vector3f next = traceRay(rayd, scene, depth+1);
                            Vector3f brdf = diffuseBRDF()*d_vec;
                            val = next.array() * brdf.array() * clamp(wi.dot(normal), 0.0f, 1.0f)/ (pdf * pdf_rr);
                            break;
                        }
                    }
                case 5:
                    {
                        //mirror
                        Vector3f reflected = reflect(ray.d, normal);
                        reflected.normalize();
                        Ray raym(i.hit, reflected);

                        Vector3f next = traceRay(raym, scene, 0);
                        val = next.array()/ pdf_rr;
                        break;
                    }
                case 7:
                    {
                        //fresnel refraction/reflection
//                        std::cout << "reflected: " << reflected[0] << " " << reflected[1] << " "<< reflected[2] << std::endl;
//                        std::cout << "normal: " << normal[0] << " " << normal[1] << " "<< normal[2] << std::endl;
                        break;
                    }
            }
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
    Vector3f out = in - (2*in.dot(n)*n);
    return out;
}

float PathTracer::diffuseBRDF(){
    float num = 1.0/PI;
    return num;
}

float PathTracer::phongBRDF(Vector3f wi, Vector3f n, Vector3f wo, int s){
    Vector3f reflect = wi - 2.f*wi.dot(n)*n;
    float dotted = reflect.dot(wo);
    float num = (s+2.f)/(2.f*PI) * pow(dotted, 2.f);
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

            float r = 255.f * ((float)intensityValues[offset][0] / (1.f + (float)intensityValues[offset][0]));
            float g = 255.f * ((float)intensityValues[offset][1]/ (1.f + (float)intensityValues[offset][1]));
            float b = 255.f * ((float)intensityValues[offset][2] / (1.f + (float)intensityValues[offset][2]));
//            std::cout << "r: " << intensityValues[offset][0] << " " << r << std::endl;
            imageData[offset] = qRgb(r,g,b);
        }
    }
}
