#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>
#include <optional>
#include <vector>

#include "Executive.hpp"
#include "UsagiCore.hpp"

namespace RT
{

// -----------------------------------------------------------------------------
// Math & Utilities
// -----------------------------------------------------------------------------
constexpr float PI = 3.14159265359f;

struct Vector3f
{
    float x, y, z;

    float & operator[](int i) { return (&x)[i]; }

    const float & operator[](int i) const { return (&x)[i]; }

    Vector3f operator-() const { return { -x, -y, -z }; }

    Vector3f operator+(const Vector3f & o) const
    {
        return { x + o.x, y + o.y, z + o.z };
    }

    Vector3f & operator+=(const Vector3f & o)
    {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    Vector3f operator-(const Vector3f & o) const
    {
        return { x - o.x, y - o.y, z - o.z };
    }

    Vector3f operator*(const Vector3f & o) const
    {
        return { x * o.x, y * o.y, z * o.z };
    }

    Vector3f operator*(float s) const { return { x * s, y * s, z * s }; }

    float dot(const Vector3f & o) const { return x * o.x + y * o.y + z * o.z; }

    Vector3f cross(const Vector3f & o) const
    {
        return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x };
    }

    float length_squared() const { return x * x + y * y + z * z; }

    float length() const { return std::sqrt(length_squared()); }

    Vector3f normalize() const
    {
        float len = length();
        if(len > 0.000001f) return { x / len, y / len, z / len };
        return { 0.0f, 0.0f, 0.0f };
    }
};

typedef Vector3f Normal3f;
typedef Vector3f Color3f;

/*
 * Shio: Xorshift32 for fast per-pixel random number generation.
 */
struct XorShift32
{
    uint32_t state;

    void seed(uint32_t s) { state = s ? s : 1'337; }

    uint32_t next_u32()
    {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    // Returns float in [0, 1)
    float next_float() { return (next_u32() & 0xFF'FFFF) / 16777216.0f; }
};

/*
 * Shio: Generates a random point inside the unit sphere.
 * Used for diffuse scattering.
 */
inline Vector3f random_in_unit_sphere(XorShift32 & rng)
{
    while(true)
    {
        Vector3f p = { rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f,
            rng.next_float() * 2.0f - 1.0f };
        if(p.length_squared() < 1.0f) return p;
    }
}

// -----------------------------------------------------------------------------
// Scene Definitions
// -----------------------------------------------------------------------------

enum class MaterialType
{
    Lambert,
    Metal,
    Light,
};

struct Material
{
    MaterialType type;
    Color3f      albedo;
    Color3f      emission;
    float        roughness; // For Metal
};

struct HitRecord
{
    float    t;
    Vector3f point;
    Normal3f normal;
    int      material_index;
};

struct Sphere
{
    Vector3f center;
    float    radius;
    int      material_index;

    std::optional<float> intersect(
        const Vector3f & o, const Vector3f & d, float t_min, float t_max) const
    {
        Vector3f oc           = o - center;
        float    a            = d.length_squared();
        float    half_b       = oc.dot(d);
        float    c            = oc.length_squared() - radius * radius;
        float    discriminant = half_b * half_b - a * c;

        if(discriminant < 0) return std::nullopt;
        float sqrtd = std::sqrt(discriminant);

        float root = (-half_b - sqrtd) / a;
        if(root < t_min || root > t_max)
        {
            root = (-half_b + sqrtd) / a;
            if(root < t_min || root > t_max) return std::nullopt;
        }
        return root;
    }
};

/*
 * Shio: Axis-Aligned Box defined by min/max points.
 */
struct Box
{
    Vector3f min;
    Vector3f max;
    int      material_index;

    std::optional<float> intersect(
        const Vector3f & o, const Vector3f & d, float t_min, float t_max) const
    {
        float t0 = t_min, t1 = t_max;

        for(int i = 0; i < 3; ++i)
        {
            float invD  = 1.0f / d[i];
            float tNear = (min[i] - o[i]) * invD;
            float tFar  = (max[i] - o[i]) * invD;
            if(invD < 0.0f) std::swap(tNear, tFar);
            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;
            if(t1 <= t0) return std::nullopt;
        }
        return t0;
    }

    Normal3f get_normal(const Vector3f & p) const
    {
        // Determine which face we hit
        const float epsilon = 0.0001f;
        if(std::abs(p.x - min.x) < epsilon) return { -1, 0, 0 };
        if(std::abs(p.x - max.x) < epsilon) return { 1, 0, 0 };
        if(std::abs(p.y - min.y) < epsilon) return { 0, -1, 0 };
        if(std::abs(p.y - max.y) < epsilon) return { 0, 1, 0 };
        if(std::abs(p.z - min.z) < epsilon) return { 0, 0, -1 };
        if(std::abs(p.z - max.z) < epsilon) return { 0, 0, 1 };
        return { 0, 1, 0 };
    }
};

// -----------------------------------------------------------------------------
// Components
// -----------------------------------------------------------------------------

struct ComponentRay
{
    Vector3f origin;
    Normal3f direction;
    float    t_max;
};

struct ComponentPixel
{
    int x, y;
};

/*
 * Shio: Holds the path tracing state for a single sample.
 * Allows the ray to 'pause' and 'resume' between system updates (bounces).
 */
struct ComponentPathState
{
    Color3f    throughput;
    Color3f    accumulated_radiance;
    XorShift32 rng;
    int        depth;
    bool       active;
    int        sample_count; // For progressive accumulation
};

// -----------------------------------------------------------------------------
// Services
// -----------------------------------------------------------------------------

struct ServiceScheduler
{
    Usagi::TaskGraphExecutionHost * host = nullptr;
};

struct ServiceRayQueue
{
    std::vector<uint32_t> queue;
    std::vector<uint32_t> next_queue;
};

struct ServiceGDICanvasProvider
{
    uint32_t *       pixel_buffer;
    int              width;
    int              height;
    std::atomic<int> frame_count = 0;
};

/*
 * Shio: Stores the scene geometry and materials.
 */
struct ServiceScene
{
    std::vector<Sphere>   spheres;
    std::vector<Box>      boxes;
    std::vector<Material> materials;

    std::optional<HitRecord> intersect(
        const Vector3f & o, const Vector3f & d, float t_max)
    {
        HitRecord rec;
        rec.t             = t_max;
        bool hit_anything = false;

        for(const auto & s : spheres)
        {
            if(auto t = s.intersect(o, d, 0.001f, rec.t))
            {
                rec.t              = *t;
                rec.point          = o + d * rec.t;
                rec.normal         = (rec.point - s.center).normalize();
                rec.material_index = s.material_index;
                hit_anything       = true;
            }
        }

        for(const auto & b : boxes)
        {
            if(auto t = b.intersect(o, d, 0.001f, rec.t))
            {
                rec.t              = *t;
                rec.point          = o + d * rec.t;
                rec.normal         = b.get_normal(rec.point);
                rec.material_index = b.material_index;
                hit_anything       = true;
            }
        }

        if(hit_anything) return rec;
        return std::nullopt;
    }
};

// -----------------------------------------------------------------------------
// Systems
// -----------------------------------------------------------------------------

/*
 * Shio: Initializes rays for the start of a frame (or a new sample).
 */
struct SystemGenerateCameraRays
{
    using WriteComponent = Usagi::ComponentList<ComponentRay, ComponentPathState>;
    using ReadComponent  = Usagi::ComponentList<ComponentPixel>;
    using WriteService   = Usagi::ComponentList<ServiceGDICanvasProvider, ServiceRayQueue>;
    using ReadService    = Usagi::ComponentList<ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & canvas    = services.template get<ServiceGDICanvasProvider>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();

        canvas.frame_count++;

        Vector3f cam_pos      = { 0.0f, 5.0f, -18.0f }; // Moved back to see full box
        float    aspect_ratio = static_cast<float>(canvas.width) / static_cast<float>(canvas.height);

        size_t count = entities.size();
        ray_queue.queue.resize(count);

        auto pixels = entities.template get_array<ComponentPixel>();
        auto rays   = entities.template get_array<ComponentRay>();
        auto states = entities.template get_array<ComponentPathState>();

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                ray_queue.queue[i] = static_cast<uint32_t>(i);

                auto & pixel = pixels[i];
                auto & ray   = rays[i];
                auto & state = states[i];

                // Initialize RNG once per pixel
                if(state.rng.state == 0)
                {
                    state.rng.seed(pixel.y * canvas.width + pixel.x + 1);
                }

                // Jitter for anti-aliasing
                float u = (pixel.x + state.rng.next_float()) / (float)canvas.width;
                float v = (pixel.y + state.rng.next_float()) / (float)canvas.height;

                // NDC to Camera Space
                float px = (2.0f * u - 1.0f) * aspect_ratio;
                float py = 1.0f - 2.0f * v;

                ray.origin    = cam_pos;
                ray.direction = Vector3f { px, py, 1.0f }.normalize();
                ray.t_max     = 1000.0f;

                // Reset path state for new sample
                state.throughput = { 1.0f, 1.0f, 1.0f };

                if(canvas.frame_count == 1)
                    state.accumulated_radiance = { 0, 0, 0 };

                state.depth  = 0;
                state.active = true;
            }
        });
    }
};

/*
 * Shio: The core path tracing logic.
 * Performs one bounce per update call and uses deferred tasks for cyclic
 * execution.
 * Now fully data-parallel across the ray queue.
 */
struct SystemPathBounce
{
    using WriteComponent =
        Usagi::ComponentList<ComponentRay, ComponentPathState>;
    using ReadService = Usagi::ComponentList<ServiceScene>;
    using WriteService =
        Usagi::ComponentList<ServiceRayQueue, ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();

        ray_queue.next_queue.clear();
        size_t count = ray_queue.queue.size();
        if(count == 0) return;

        auto rays   = entities.template get_array<ComponentRay>();
        auto states = entities.template get_array<ComponentPathState>();

        std::mutex merge_mutex;

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            std::vector<uint32_t> local_next;
            local_next.reserve(end - start);

            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.queue[i];
                auto & ray   = rays[id];
                auto & state = states[id];

                if(!state.active) continue;

                auto hit = scene.intersect(ray.origin, ray.direction, ray.t_max);

                if(hit)
                {
                    const auto & mat = scene.materials[hit->material_index];

                    // Emissive contribution
                    state.accumulated_radiance += state.throughput * mat.emission;

                    // Scatter
                    if(mat.type == MaterialType::Light || state.depth >= 5)
                    {
                        state.active = false;
                        continue;
                    }

                    // Lambertian Scatter
                    Vector3f target = hit->point + hit->normal +
                        random_in_unit_sphere(state.rng).normalize();

                    ray.origin    = hit->point;
                    ray.direction = (target - hit->point).normalize();
                    ray.t_max     = 1000.0f;

                    state.throughput = state.throughput * mat.albedo;
                    state.depth++;

                    local_next.push_back(id);
                }
                else
                {
                    // Background (Black for Cornell Box)
                    state.active = false;
                }
            }

            if(!local_next.empty())
            {
                std::lock_guard<std::mutex> lock(merge_mutex);
                ray_queue.next_queue.insert(ray_queue.next_queue.end(), local_next.begin(), local_next.end());
            }
        });

        if(!ray_queue.next_queue.empty())
        {
            std::swap(ray_queue.queue, ray_queue.next_queue);
            scheduler.host->submit_deferred_task(
                [this, &entities, &services]() {
                    this->update(entities, services);
                });
        }
    }
};

struct SystemRenderGDICanvas
{
    using ReadComponent  = Usagi::ComponentList<ComponentPixel, ComponentPathState>;
    using WriteService   = Usagi::ComponentList<ServiceGDICanvasProvider>;
    using ReadService    = Usagi::ComponentList<ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & canvas    = services.template get<ServiceGDICanvasProvider>();
        auto & scheduler = services.template get<ServiceScheduler>();
        float  scale     = 1.0f / (float)std::max(1, canvas.frame_count.load());

        size_t count = entities.size();
        auto pixels  = entities.template get_array<ComponentPixel>();
        auto states  = entities.template get_array<ComponentPathState>();

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                auto & pixel = pixels[i];
                auto & state = states[i];

                // Average samples
                Color3f color = state.accumulated_radiance * scale;

                // Simple Gamma Correction (approximate sqrt)
                float r_f = std::sqrt(color.x);
                float g_f = std::sqrt(color.y);
                float b_f = std::sqrt(color.z);

                uint8_t r = static_cast<uint8_t>(std::clamp(r_f * 255.0f, 0.0f, 255.0f));
                uint8_t g = static_cast<uint8_t>(std::clamp(g_f * 255.0f, 0.0f, 255.0f));
                uint8_t b = static_cast<uint8_t>(std::clamp(b_f * 255.0f, 0.0f, 255.0f));

                canvas.pixel_buffer[pixel.y * canvas.width + pixel.x] = (r << 16) | (g << 8) | b;
            }
        });
    }
};

} // namespace RT
