#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>
#include <algorithm>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 sampleSpecular(
    const glm::vec3& normal,
    const glm::vec3& wo,
    float shininess,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float u1 = u01(rng);
    float u2 = u01(rng);

    float theta = acosf(powf(u1, 1.0f / (shininess + 1.0f)));
    float phi = 2.0f * PI * u2;

    float xs = sinf(theta) * cosf(phi);
    float ys = sinf(theta) * sinf(phi);
    float zs = cosf(theta);

    glm::vec3 r = reflect(-wo, normal);
    glm::vec3 w = normalize(r);
    glm::vec3 u = normalize(cross((fabs(w.x) > 0.1f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), w));
    glm::vec3 v = cross(w, u);

    // Transform to world coord
    return normalize(xs * u + ys * v + zs * w);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    glm::vec3 color,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    glm::vec3 wi;
    glm::vec3 wo = -pathSegment.ray.direction;
    float pdf;
    glm::vec3 bsdf;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float p = u01(rng);

    // Specular
    if (m.hasReflective == 1.f || (m.hasReflective > 0.f && p < m.hasReflective)) { 
        wi = reflect(-wo, normal);
        bsdf = color;
        pdf = dot(normal, wi); // to cancel out the cosTheta term
    }
    // Diffuse
    else { 
        wi = calculateRandomDirectionInHemisphere(normal, rng);
        bsdf = color / PI;
        pdf = dot(normal, wi) / PI;
    }

    pathSegment.color *= bsdf * dot(normal, wi) / pdf;
    pathSegment.ray.direction = normalize(wi);
    pathSegment.ray.origin = intersect + EPSILON * pathSegment.ray.direction;
}
