#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float triangleIntersectionTest(
    Geom tri,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec3& tangent,
    glm::vec3& bitangent)
{
    // Transform ray to object space
    glm::vec3 ro = multiplyMV(tri.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(tri.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    // Ray-plane intersection
    glm::vec3 n = glm::normalize(glm::cross(tri.v1.position - tri.v0.position, tri.v2.position - tri.v0.position));
    float t = (glm::dot(tri.v0.position, n) - glm::dot(ro, n)) / dot(rd, n);
    if (t < 0.001f) return -1;
    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    // Test if the intersection is inside the triangle
    glm::vec3 v0v1 = tri.v1.position - tri.v0.position;
    glm::vec3 v0v2 = tri.v2.position - tri.v0.position;
    glm::vec3 v0p = objspaceIntersection - tri.v0.position;

    float d00 = glm::dot(v0v1, v0v1);
    float d01 = glm::dot(v0v1, v0v2);
    float d11 = glm::dot(v0v2, v0v2);
    float d20 = glm::dot(v0p, v0v1);
    float d21 = glm::dot(v0p, v0v2);
    float denom = d00 * d11 - d01 * d01;

    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

 

    if (u >= 0 && v >= 0 && w >= 0) {
        // Position
        intersectionPoint = multiplyMV(tri.transform, glm::vec4(objspaceIntersection, 1.f)); // Transform to world space
        
        // Normal
        glm::vec3 objspaceNormal = glm::normalize(u * tri.v0.normal + v * tri.v1.normal + w * tri.v2.normal);
        normal = glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(objspaceNormal, 0.f))); // Transform to world space

        // UV
        uv = u * tri.v0.uv + v * tri.v1.uv + w * tri.v2.uv;

        // Tangent and bitangent
        glm::vec2 uv01 = tri.v1.uv - tri.v0.uv;
        glm::vec2 uv02 = tri.v2.uv - tri.v0.uv;
        float num = 1.0f / (uv01.x * uv02.y - uv01.y * uv02.x);
        glm::vec3 objspaceTangent = glm::normalize((v0v1 * uv02.y - v0v2 * uv01.y) * num);
        glm::vec3 objspaceBitangent = glm::normalize((v0v2 * uv01.x - v0v1 * uv02.x) * num);
        tangent = glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(objspaceTangent, 0.f)));
        bitangent = glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(objspaceBitangent, 0.f)));

        return t;
    }

    return -1;
}