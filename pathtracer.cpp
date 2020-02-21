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
            intensityValues[offset] = tracePixel(x, y, scene, invViewMat, 8);
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
        out += traceRay(r, scene, 0, false);
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

Vector3f PathTracer::traceRay(const Ray& r, const Scene& scene, int depth, bool fromBack)
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
        const tinyobj::real_t ior = mat.ior;
        const tinyobj::real_t shininess = mat.shininess;
        float ior_air = 1.f;

//        const std::string diffuseTex = mat.diffuse_texname;//Diffuse texture name
        Vector3f d_vec(d[0], d[1], d[2]);
        Vector3f s_vec(s[0], s[1], s[2]);
        float pdf = 1.0/(2.0*PI);
        const Vector3f normal = t->getNormal(i.hit);
        bool spec = false;
        if (s_vec != Vector3f(0,0,0)){
            spec = true;
        }

        L = directLighting(i, normal, scene, spec, ray.d);
        if (d_vec != Vector3f(0,0,0)){
            L.array() *= d_vec.array();
        }

        /*L = Vector3f(d[0], d[1], d[2]);*///p.emitted(-w); //p is intersection, -w is ray.inv_d
        float pdf_rr = 0.5; //continueProb();
        if (random() < pdf_rr){

            Vector3f wi = sampleNextDir2(normal);
//            std::cout << "normal: " << normal[0] << " " << normal[1] << " "<< normal[2] << std::endl;

            Vector3f val;
            switch(mat.illum){
                case 2:
                    {
                        if (s_vec != Vector3f(0,0,0)){
                            //phong
                            Vector3f brdf = phongBRDF(ray.d, normal, wi, s_vec, shininess);
//                            std::cout << shininess << std::endl;
//                            std::cout << "brdf: " << brdf[0] << " " << brdf[1] << " "<< brdf[2] << std::endl;
//                            L.array() *= brdf.array();
                            Ray rayp(i.hit, wi);
                            Vector3f next = traceRay(rayp, scene, depth+1, false);
                            val = next.array() * brdf.array() * clamp(wi.dot(normal), 0.0f, 1.0f)/ (pdf * pdf_rr);
                            break;
                        } else {
                            //diffuse
//                            L.array() *= d_vec.array();
                            Ray rayd(i.hit, wi);
                            Vector3f next = traceRay(rayd, scene, depth+1, false);
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

                        Vector3f next = traceRay(raym, scene, 0, false);
                        val = next.array()/ pdf_rr;
                        break;
                    }
                case 7:
                    {
                        Vector3f reflected = reflect(ray.d, normal);
                        reflected.normalize();
                        Ray raym(i.hit, reflected);

                        Vector3f reflected_half = traceRay(raym, scene, 0, false);
                        reflected_half = reflected_half.array();// pdf_rr;

                        //fresnel refraction/reflection
                        //negative nt = 1, ni = ior, flip normal and ni_nt if coming from back of triangle
                        Vector3f wt;
//                        std::cout << ray.d.dot(normal) << std::endl;
                        if (fromBack){
//                            std::cout << "here" << std::endl;
                            wt = specRefractBRDF(ray.d, -normal, ior, ior_air);
                        } else {
//                            std::cout << "nope" << std::endl;
                            wt = specRefractBRDF(ray.d, normal, ior_air, ior);
                        }

//                        Vector3f wt = specRefractBRDF(ray.d, normal, ior_air, ior);
                        wt.normalize();
                        if (wt.dot(normal) < 0.f){
//                            std::cout << "here" << std::endl;
                            fromBack = true;
                        } else {
                            fromBack = false;
                        }
                        Ray rayf(i.hit, wt);
//                        IntersectionInfo i1;
//                        scene.getIntersection(rayf, &i1);
//                        const Triangle *t1 = static_cast<const Triangle *>(i1.data);
//                        const tinyobj::material_t& mat1 = t1->getMaterial();
//                        const tinyobj::real_t ior1 = mat1.ior;
//                        const Vector3f normal1 = t1->getNormal(i1.hit);

//                        Vector3f wt2 = specRefractBRDF(wt, -normal1, ior1, ior_air);
//                        wt2.normalize();
//                        Ray rayf2(i1.hit, wt2);
                        Vector3f refracted_half = traceRay(rayf, scene, 0, fromBack);
//                        val = refracted_half/pdf_rr;
//                        Vector3f refracted_half(0,0,0);
//                        Vector3f reflected_half(0,0,0);
                        refracted_half = refracted_half.array();// pdf_rr;
                        float c_theta_i = ray.d.dot(normal);
                        if (c_theta_i < 0.f){
                            c_theta_i = -c_theta_i;
                        }
                        float fres = (ior-ior_air)/(ior+ior_air);
                        if (fromBack){
                            fres = (ior_air-ior)/(ior_air+ior);
//                            c_theta_i = clamp(ray.d.dot(-normal), 0.f, 1.f);
                        }
                        float R0 = pow(fres, 2);
                        float Rti = R0 + ((1-R0)*pow(1-c_theta_i, 5.0));
//                        std::cout<< Rti << std::endl;
                        val = (reflected_half * Rti) + (refracted_half*(1-Rti));
                        val.array() /= pdf_rr;
//                        std::cout << "val: " << val[0] << " " << val[1] << " "<< val[2] << std::endl;
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

Vector3f PathTracer::directLighting(IntersectionInfo i, Vector3f normal, const Scene& scene, bool spec, Vector3f rd){
    const Triangle *t = static_cast<const Triangle *>(i.data);
    const tinyobj::material_t& mat = t->getMaterial();
    const tinyobj::real_t *s = mat.specular;
    const tinyobj::real_t shininess = mat.shininess;
    Vector3f s_vec(s[0], s[1], s[2]);
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
            if (spec){
                Vector3f brdf = phongBRDF(newdir, normal, rd, s_vec, shininess);
                light.array() *= brdf.array();
            }
        }
//        else if (hit->getMaterial().illum == 7){
//            const tinyobj::material_t& mat1 = hit->getMaterial();
//            const tinyobj::real_t ior1 = mat1.ior;
//            const Vector3f normal1 = hit->getNormal(new_i.hit);
//        }
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

Vector3f PathTracer::phongBRDF(Vector3f wi, Vector3f n, Vector3f wo, Vector3f s, int exp){
    Vector3f reflect = wi - 2.f*wi.dot(n)*n;
    float dotted = reflect.dot(wo);
    float num = (exp+2.f)/(2.f*PI) * pow(dotted, exp);
    return s*num;
}

Vector3f PathTracer::specRefractBRDF(Vector3f wi, Vector3f n, float ior_in, float ior_out){
    wi.normalize();
    float theta_i = acos(wi.dot(n));
    float c_theta_i = wi.dot(n);
    if (c_theta_i < 0.f){
        c_theta_i = -c_theta_i;
    }
//    std::cout << c_theta_i << std::endl;
    //if ni < nt
    float ni_nt = ior_in/ior_out;
    float cos_theta_t = sqrt(1 - (pow(ni_nt, 2)*(1-(pow(c_theta_i, 2)))));
    Vector3f wt = ((ni_nt)*wi) + ((ni_nt*c_theta_i - cos_theta_t)*n);
    return wt;
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
            imageData[offset] = qRgb(r,g,b);
        }
    }
}
