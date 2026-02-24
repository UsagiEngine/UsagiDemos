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
 * Shio: High-quality PCG32 PRNG for per-pixel sampling.
 */
struct SamplerPCG32
{
    uint64_t state;
    uint64_t inc;

    void seed(uint64_t initstate, uint64_t initseq)
    {
        state = 0U;
        inc = (initseq << 1u) | 1u;
        next_u32();
        state += initstate;
        next_u32();
    }

    uint32_t next_u32()
    {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    // Returns float in [0, 1)
    float next_float() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }
};

struct ONB {
    Vector3f u, v, w;
    void build_from_w(const Normal3f& n) {
        w = n.normalize();
        Vector3f a = (std::abs(w.x) > 0.9f) ? Vector3f{0, 1, 0} : Vector3f{1, 0, 0};
        v = w.cross(a).normalize();
        u = w.cross(v);
    }
    Vector3f local(const Vector3f& a) const {
        return u * a.x + v * a.y + w * a.z;
    }
};

inline Vector3f cosine_sample_hemisphere(float u1, float u2) {
    float r = std::sqrt(u1);
    float theta = 2.0f * PI * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    return {x, y, z};
}

/*
 * Shio: Generates a random point inside the unit sphere.
 * Used for diffuse scattering.
 */
inline Vector3f random_in_unit_sphere(SamplerPCG32 & rng)
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

struct ComponentRayHit
{
    HitRecord hit;
    bool      did_hit;
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
    Color3f      throughput;
    Color3f      radiance;
    SamplerPCG32 rng;
    int          depth;
    bool         active;
    float        sample_x;
    float        sample_y;
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
    std::vector<uint32_t> active_rays;
    
    std::vector<uint32_t> q_lambert;
    std::vector<uint32_t> q_metal;
    std::vector<uint32_t> q_light;
    
    std::vector<uint32_t> next_rays;
};

struct ServiceGDICanvasProvider
{
    uint32_t *       pixel_buffer;
    int              width;
    int              height;
    std::atomic<int> frame_count = 0;
};

/*
 * Shio: Thread-safe framebuffer allowing sub-pixel filtered samples
 * to be splatted concurrently.
 */
struct ServiceFilm
{
    struct Pixel {
        std::atomic<float> r{0}, g{0}, b{0}, weight{0};
    };

    std::vector<Pixel> pixels;
    int width = 0;
    int height = 0;
    float filter_radius_x = 2.0f;
    float filter_radius_y = 2.0f;
    float alpha = 2.0f;
    float expX = 1.0f;
    float expY = 1.0f;

    void init(int w, int h, float rx = 2.0f, float ry = 2.0f)
    {
        width = w;
        height = h;
        pixels = std::vector<Pixel>(w * h);
        filter_radius_x = rx;
        filter_radius_y = ry;
        expX = std::exp(-alpha * rx * rx);
        expY = std::exp(-alpha * ry * ry);
    }

    float gaussian(float d, float expv) const {
        return std::max(0.0f, std::exp(-alpha * d * d) - expv);
    }

    float evaluate_filter(float dx, float dy) const {
        return gaussian(dx, expX) * gaussian(dy, expY);
    }

    void add_sample(float px, float py, const Color3f& L) {
        int x0 = std::max(0, (int)std::ceil(px - 0.5f - filter_radius_x));
        int x1 = std::min(width - 1, (int)std::floor(px - 0.5f + filter_radius_x));
        int y0 = std::max(0, (int)std::ceil(py - 0.5f - filter_radius_y));
        int y1 = std::min(height - 1, (int)std::floor(py - 0.5f + filter_radius_y));

        auto atomic_add = [](std::atomic<float>& target, float val) {
            float old = target.load(std::memory_order_relaxed);
            while(!target.compare_exchange_weak(old, old + val, std::memory_order_relaxed));
        };

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                float dx = px - 0.5f - x;
                float dy = py - 0.5f - y;
                float weight = evaluate_filter(dx, dy);
                if (weight > 0) {
                    int idx = y * width + x;
                    atomic_add(pixels[idx].r, L.x * weight);
                    atomic_add(pixels[idx].g, L.y * weight);
                    atomic_add(pixels[idx].b, L.z * weight);
                    atomic_add(pixels[idx].weight, weight);
                }
            }
        }
    }
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
        ray_queue.active_rays.resize(count);

        auto pixels = entities.template get_array<ComponentPixel>();
        auto rays   = entities.template get_array<ComponentRay>();
        auto states = entities.template get_array<ComponentPathState>();

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                ray_queue.active_rays[i] = static_cast<uint32_t>(i);

                auto & pixel = pixels[i];
                auto & ray   = rays[i];
                auto & state = states[i];

                // Initialize RNG once per pixel
                if(state.rng.inc == 0)
                {
                    state.rng.seed(pixel.y * canvas.width + pixel.x + 1, 1337);
                }

                // Jitter for anti-aliasing
                float j_x = state.rng.next_float();
                float j_y = state.rng.next_float();
                state.sample_x = pixel.x + j_x;
                state.sample_y = pixel.y + j_y;

                float u = state.sample_x / (float)canvas.width;
                float v = state.sample_y / (float)canvas.height;

                // NDC to Camera Space
                float px = (2.0f * u - 1.0f) * aspect_ratio;
                float py = 1.0f - 2.0f * v;

                ray.origin    = cam_pos;
                ray.direction = Vector3f { px, py, 1.0f }.normalize();
                ray.t_max     = 1000.0f;

                // Reset path state for new sample
                state.throughput = { 1.0f, 1.0f, 1.0f };
                state.radiance   = { 0.0f, 0.0f, 0.0f };
                state.depth      = 0;
                state.active     = true;
            }
        });
    }
};

/*
 * Shio: Intersects active rays with the scene and routes them to material queues.
 * Decoupled from material evaluation to demonstrate data-oriented processing.
 */
struct SystemIntersectRays
{
    using ReadComponent  = Usagi::ComponentList<ComponentRay, ComponentPathState>;
    using WriteComponent = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler, ServiceFilm>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & film      = services.template get<ServiceFilm>();

        ray_queue.next_rays.clear();
        ray_queue.q_lambert.clear();
        ray_queue.q_metal.clear();
        ray_queue.q_light.clear();

        size_t count = ray_queue.active_rays.size();
        if(count == 0) return;

        auto rays   = entities.template get_array<ComponentRay>();
        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();

        std::mutex merge_mutex;

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            std::vector<uint32_t> loc_lambert;
            std::vector<uint32_t> loc_metal;
            std::vector<uint32_t> loc_light;

            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.active_rays[i];
                auto & ray   = rays[id];
                auto & hit_c = hits[id];
                auto & state = states[id];

                if(!state.active) continue;

                auto opt_hit = scene.intersect(ray.origin, ray.direction, ray.t_max);

                if(opt_hit)
                {
                    hit_c.did_hit = true;
                    hit_c.hit = *opt_hit;

                    const auto & mat = scene.materials[opt_hit->material_index];
                    
                    if (state.depth >= 5) {
                        state.active = false;
                        loc_light.push_back(id); // Send to light to pick up emission, but it won't bounce
                    } else if (mat.type == MaterialType::Lambert) {
                        loc_lambert.push_back(id);
                    } else if (mat.type == MaterialType::Metal) {
                        loc_metal.push_back(id);
                    } else if (mat.type == MaterialType::Light) {
                        loc_light.push_back(id);
                    }
                }
                else
                {
                    // Background
                    hit_c.did_hit = false;
                    state.active = false;
                    film.add_sample(state.sample_x, state.sample_y, state.radiance);
                }
            }

            std::lock_guard<std::mutex> lock(merge_mutex);
            ray_queue.q_lambert.insert(ray_queue.q_lambert.end(), loc_lambert.begin(), loc_lambert.end());
            ray_queue.q_metal.insert(ray_queue.q_metal.end(), loc_metal.begin(), loc_metal.end());
            ray_queue.q_light.insert(ray_queue.q_light.end(), loc_light.begin(), loc_light.end());
        });
    }
};

/*
 * Shio: Evaluates Lambertian materials.
 */
struct SystemEvaluateLambert
{
    using WriteComponent = Usagi::ComponentList<ComponentRay, ComponentPathState>;
    using ReadComponent  = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();

        size_t count = ray_queue.q_lambert.size();
        if(count == 0) return;

        auto rays   = entities.template get_array<ComponentRay>();
        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();

        std::mutex merge_mutex;

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            std::vector<uint32_t> loc_next;

            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.q_lambert[i];
                auto & ray   = rays[id];
                auto & hit   = hits[id].hit;
                auto & state = states[id];
                const auto & mat = scene.materials[hit.material_index];

                state.radiance += state.throughput * mat.emission;

                ONB onb;
                onb.build_from_w(hit.normal);
                Vector3f scatter_dir = onb.local(cosine_sample_hemisphere(state.rng.next_float(), state.rng.next_float()));

                ray.origin    = hit.point;
                ray.direction = scatter_dir.normalize();
                ray.t_max     = 1000.0f;

                state.throughput = state.throughput * mat.albedo;
                state.depth++;
                loc_next.push_back(id);
            }

            std::lock_guard<std::mutex> lock(merge_mutex);
            ray_queue.next_rays.insert(ray_queue.next_rays.end(), loc_next.begin(), loc_next.end());
        });
    }
};

/*
 * Shio: Evaluates Metallic materials.
 */
struct SystemEvaluateMetal
{
    using WriteComponent = Usagi::ComponentList<ComponentRay, ComponentPathState>;
    using ReadComponent  = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler, ServiceFilm>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & film      = services.template get<ServiceFilm>();

        size_t count = ray_queue.q_metal.size();
        if(count == 0) return;

        auto rays   = entities.template get_array<ComponentRay>();
        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();

        std::mutex merge_mutex;

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            std::vector<uint32_t> loc_next;

            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.q_metal[i];
                auto & ray   = rays[id];
                auto & hit   = hits[id].hit;
                auto & state = states[id];
                const auto & mat = scene.materials[hit.material_index];

                state.radiance += state.throughput * mat.emission;

                Vector3f reflected = ray.direction - hit.normal * 2.0f * ray.direction.dot(hit.normal);
                Vector3f target = reflected;
                
                if(mat.roughness > 0.0f) {
                    target = target + random_in_unit_sphere(state.rng) * mat.roughness;
                }

                if(target.dot(hit.normal) > 0.0f) {
                    ray.origin    = hit.point;
                    ray.direction = target.normalize();
                    ray.t_max     = 1000.0f;

                    state.throughput = state.throughput * mat.albedo;
                    state.depth++;
                    loc_next.push_back(id);
                } else {
                    state.active = false;
                    film.add_sample(state.sample_x, state.sample_y, state.radiance);
                }
            }

            std::lock_guard<std::mutex> lock(merge_mutex);
            ray_queue.next_rays.insert(ray_queue.next_rays.end(), loc_next.begin(), loc_next.end());
        });
    }
};

/*
 * Shio: Evaluates Emissive/Light materials.
 */
struct SystemEvaluateLight
{
    using WriteComponent = Usagi::ComponentList<ComponentPathState>;
    using ReadComponent  = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler, ServiceFilm>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & film      = services.template get<ServiceFilm>();

        size_t count = ray_queue.q_light.size();
        if(count == 0) return;

        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.q_light[i];
                auto & hit   = hits[id].hit;
                auto & state = states[id];
                const auto & mat = scene.materials[hit.material_index];

                state.radiance += state.throughput * mat.emission;
                state.active = false;
                film.add_sample(state.sample_x, state.sample_y, state.radiance);
            }
        });
    }
};

/*
 * Shio: Cyclic Coordinator for Path Tracing Loop
 * Enqueues the next iteration if any rays remain active.
 */
struct SystemPathTracingCoordinator
{
    using ReadService  = Usagi::ComponentList<>;
    using WriteService = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & ray_queue = services.template get<ServiceRayQueue>();
        [[maybe_unused]]
        auto & scheduler = services.template get<ServiceScheduler>();

        // If this is the initial pass (e.g. from GenerateCameraRays), active_rays will be populated.
        // We only process if there is something in active_rays.
        if(!ray_queue.active_rays.empty())
        {
            this->update_bounce_graph(entities, services);
        }
    }

    void update_bounce_graph(auto && entities, auto && services)
    {
        SystemIntersectRays sys_intersect;
        sys_intersect.update(entities, services);

        SystemEvaluateLambert sys_lambert;
        sys_lambert.update(entities, services);

        SystemEvaluateMetal sys_metal;
        sys_metal.update(entities, services);

        SystemEvaluateLight sys_light;
        sys_light.update(entities, services);

        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();

        if(!ray_queue.next_rays.empty())
        {
            std::swap(ray_queue.active_rays, ray_queue.next_rays);
            ray_queue.next_rays.clear();

            scheduler.host->submit_deferred_task([this, &entities, &services]() {
                // Re-evaluate the entire bounce graph with the new active_rays
                this->update_bounce_graph(entities, services);
            });
        }
        else
        {
            ray_queue.active_rays.clear();
        }
    }
};

struct SystemRenderGDICanvas
{
    using ReadComponent  = Usagi::ComponentList<ComponentPixel>;
    using WriteService   = Usagi::ComponentList<ServiceGDICanvasProvider>;
    using ReadService    = Usagi::ComponentList<ServiceScheduler, ServiceFilm>;

    void update(auto && entities, auto && services)
    {
        auto & canvas    = services.template get<ServiceGDICanvasProvider>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & film      = services.template get<ServiceFilm>();

        size_t count = entities.size();
        auto pixels  = entities.template get_array<ComponentPixel>();

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                auto & pixel = pixels[i];

                int idx = pixel.y * canvas.width + pixel.x;
                float weight = film.pixels[idx].weight.load(std::memory_order_relaxed);

                Color3f color = {0, 0, 0};
                if (weight > 0.0f) {
                    float inv_w = 1.0f / weight;
                    color.x = film.pixels[idx].r.load(std::memory_order_relaxed) * inv_w;
                    color.y = film.pixels[idx].g.load(std::memory_order_relaxed) * inv_w;
                    color.z = film.pixels[idx].b.load(std::memory_order_relaxed) * inv_w;
                }

                // Simple Gamma Correction (approximate sqrt)
                float r_f = std::sqrt(color.x);
                float g_f = std::sqrt(color.y);
                float b_f = std::sqrt(color.z);

                uint8_t r = static_cast<uint8_t>(std::clamp(r_f * 255.0f, 0.0f, 255.0f));
                uint8_t g = static_cast<uint8_t>(std::clamp(g_f * 255.0f, 0.0f, 255.0f));
                uint8_t b = static_cast<uint8_t>(std::clamp(b_f * 255.0f, 0.0f, 255.0f));

                canvas.pixel_buffer[idx] = (r << 16) | (g << 8) | b;
            }
        });
    }
};

} // namespace RT
