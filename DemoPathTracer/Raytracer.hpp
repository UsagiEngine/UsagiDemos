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
    Translucent,
};

struct Material
{
    MaterialType type;
    Color3f      albedo;
    Color3f      emission;
    float        roughness; // For Metal / Translucent
    float        ior;       // Index of Refraction for Translucent
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

    void sample_surface(SamplerPCG32& rng, Vector3f& p, Normal3f& n, float& pdf) const
    {
        Vector3f d = max - min;
        float ax = d.y * d.z;
        float ay = d.x * d.z;
        float az = d.x * d.y;
        float total_area = 2.0f * (ax + ay + az);
        pdf = 1.0f / total_area;

        float r = rng.next_float() * (ax + ay + az);
        if (r < ax) {
            p.x = rng.next_float() > 0.5f ? min.x : max.x;
            p.y = min.y + rng.next_float() * d.y;
            p.z = min.z + rng.next_float() * d.z;
            n = {p.x == min.x ? -1.0f : 1.0f, 0.0f, 0.0f};
        } else if (r < ax + ay) {
            p.y = rng.next_float() > 0.5f ? min.y : max.y;
            p.x = min.x + rng.next_float() * d.x;
            p.z = min.z + rng.next_float() * d.z;
            n = {0.0f, p.y == min.y ? -1.0f : 1.0f, 0.0f};
        } else {
            p.z = rng.next_float() > 0.5f ? min.z : max.z;
            p.x = min.x + rng.next_float() * d.x;
            p.y = min.y + rng.next_float() * d.y;
            n = {0.0f, 0.0f, p.z == min.z ? -1.0f : 1.0f};
        }
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

constexpr int MAX_PATH_DEPTH = 15;

struct PathVertex {
    Vector3f p;
    Normal3f n;
    Color3f  beta;
    Color3f  albedo;
    bool     is_delta;
};

struct ComponentCameraPath {
    PathVertex vertices[MAX_PATH_DEPTH];
    int count = 0;
    Color3f direct_emission = {0,0,0};
    int direct_depth = 0;
};

struct ComponentLightPath {
    PathVertex vertices[MAX_PATH_DEPTH];
    int count = 0;
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
    bool         last_bounce_specular;
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

struct ServiceRenderState
{
    int pass = 0; // 0: InitCam, 1: BounceCam, 2: InitLight, 3: BounceLight, 4: Connect
};

struct ServiceRayQueue
{
    std::vector<uint32_t> active_rays;
    
    std::vector<uint32_t> q_lambert;
    std::vector<uint32_t> q_metal;
    std::vector<uint32_t> q_translucent;
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
        std::atomic<double> r{0}, g{0}, b{0}, weight{0};
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

        auto atomic_add = [](std::atomic<double>& target, double val) {
            double old = target.load(std::memory_order_relaxed);
            while(!target.compare_exchange_weak(old, old + val, std::memory_order_relaxed));
        };

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                float dx = px - 0.5f - x;
                float dy = py - 0.5f - y;
                float weight = evaluate_filter(dx, dy);
                if (weight > 0) {
                    int idx = y * width + x;
                    atomic_add(pixels[idx].r, static_cast<double>(L.x * weight));
                    atomic_add(pixels[idx].g, static_cast<double>(L.y * weight));
                    atomic_add(pixels[idx].b, static_cast<double>(L.z * weight));
                    // Note: Splat weights accumulate without dividing by frame count here. 
                    // This allows them to sum up perfectly over millions of rays.
                    atomic_add(pixels[idx].weight, static_cast<double>(weight));
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

    bool sample_light(SamplerPCG32& rng, Vector3f& p, Normal3f& n, Color3f& emission, float& pdf) const {
        for (const auto& b : boxes) {
            if (materials[b.material_index].type == MaterialType::Light) {
                b.sample_surface(rng, p, n, pdf);
                emission = materials[b.material_index].emission;
                return true;
            }
        }
        return false;
    }

    float get_light_area() const {
        float area = 0.0f;
        for (const auto& b : boxes) {
            if (materials[b.material_index].type == MaterialType::Light) {
                Vector3f d = b.max - b.min;
                area += 2.0f * (d.x*d.y + d.x*d.z + d.y*d.z);
            }
        }
        return area;
    }
};

inline float compute_mis_weight(const PathVertex* path, int k, int s, float light_area) {
    float sum_W2 = 0.0f;
    float W2_s = 0.0f;

    for (int i = 0; i < k; ++i) {
        float W_i = 0.0f;
        if (i == 0) {
            W_i = light_area; // PDF of sampling the area light directly
        } else {
            if (path[i-1].is_delta || path[i].is_delta) {
                W_i = 0.0f; // Cannot connect to a delta/specular surface
            } else {
                Vector3f diff = path[i].p - path[i-1].p;
                float dist2 = diff.length_squared();
                if (dist2 > 0.00001f) {
                    float dist = std::sqrt(dist2);
                    Vector3f L = diff * (1.0f / dist);
                    float ndotl_a = std::abs(path[i-1].n.dot(L));
                    float ndotl_b = std::abs(path[i].n.dot(-L));
                    if (ndotl_a > 0.0f && ndotl_b > 0.0f) {
                        float G = (ndotl_a * ndotl_b) / dist2;
                        W_i = PI / G; // 1 / Area PDF
                    }
                }
            }
        }
        float w2 = W_i * W_i;
        sum_W2 += w2;
        if (i == s) W2_s = w2;
    }
    return sum_W2 > 0.0f ? W2_s / sum_W2 : 0.0f;
}

// -----------------------------------------------------------------------------
// Systems
// -----------------------------------------------------------------------------

/*
 * Shio: Initializes rays for Camera (pass=0) or Light (pass=2).
 */
struct SystemGenerateRays
{
    using WriteComponent = Usagi::ComponentList<ComponentRay, ComponentPathState, ComponentCameraPath, ComponentLightPath>;
    using ReadComponent  = Usagi::ComponentList<ComponentPixel>;
    using WriteService   = Usagi::ComponentList<ServiceGDICanvasProvider, ServiceRayQueue>;
    using ReadService    = Usagi::ComponentList<ServiceScheduler, ServiceRenderState, ServiceScene>;

    void update(auto && entities, auto && services)
    {
        auto & state_svc = services.template get<ServiceRenderState>();
        if (state_svc.pass != 0 && state_svc.pass != 2) return;

        auto & canvas    = services.template get<ServiceGDICanvasProvider>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & scene     = services.template get<ServiceScene>();

        if (state_svc.pass == 0) canvas.frame_count++;

        Vector3f cam_pos      = { 0.0f, 5.0f, -18.0f }; // Moved back to see full box
        float    aspect_ratio = static_cast<float>(canvas.width) / static_cast<float>(canvas.height);

        size_t count = entities.size();
        ray_queue.active_rays.resize(count);

        auto pixels = entities.template get_array<ComponentPixel>();
        auto rays   = entities.template get_array<ComponentRay>();
        auto states = entities.template get_array<ComponentPathState>();
        auto cpaths = entities.template get_array<ComponentCameraPath>();
        auto lpaths = entities.template get_array<ComponentLightPath>();

        bool is_cam = (state_svc.pass == 0);

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
                    
                    // Advance RNG by the current frame count so we don't repeat the exact 
                    // same light/camera subpaths every single frame if the PRNG logic was flawed
                    // or reset elsewhere. PCG32 is robust, but advancing ensures a unique temporal stream.
                    for (int f = 0; f < canvas.frame_count.load(); ++f) {
                        state.rng.next_u32();
                    }
                }

                state.throughput = { 1.0f, 1.0f, 1.0f };
                state.radiance   = { 0.0f, 0.0f, 0.0f };
                state.depth      = 0;
                state.active     = true;
                state.last_bounce_specular = true;

                if (is_cam) {
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

                    cpaths[i].count = 0;
                    cpaths[i].direct_emission = {0,0,0};

                    // Note: In BDPT, camera is theoretically vertex 0.
                    // We only record surface hits for now.
                } else {
                    lpaths[i].count = 0;
                    Vector3f p; Normal3f n; Color3f e; float pdf;
                    if (scene.sample_light(state.rng, p, n, e, pdf)) {
                        state.throughput = e * (1.0f / pdf);
                        ONB onb; onb.build_from_w(n);
                        Vector3f dir = onb.local(cosine_sample_hemisphere(state.rng.next_float(), state.rng.next_float()));
                        
                        // First light vertex is the point on the light source.
                        // Setting albedo to PI ensures the 1/PI lambertian factor correctly cancels when connecting directly.
                        lpaths[i].vertices[0] = {p, n, state.throughput, {PI,PI,PI}, false};
                        lpaths[i].count = 1;
                        
                        ray.origin = p + n * 0.001f;
                        ray.direction = dir.normalize();
                        ray.t_max = 1000.0f;
                    } else {
                        state.active = false;
                    }
                }
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
        ray_queue.q_translucent.clear();

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
            std::vector<uint32_t> loc_translucent;

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
                    
                    if (state.depth >= MAX_PATH_DEPTH) {
                        state.active = false;
                        loc_light.push_back(id); // Send to light to pick up emission, but it won't bounce
                    } else if (mat.type == MaterialType::Lambert) {
                        loc_lambert.push_back(id);
                    } else if (mat.type == MaterialType::Metal) {
                        loc_metal.push_back(id);
                    } else if (mat.type == MaterialType::Translucent) {
                        loc_translucent.push_back(id);
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
            ray_queue.q_translucent.insert(ray_queue.q_translucent.end(), loc_translucent.begin(), loc_translucent.end());
        });
    }
};

/*
 * Shio: Evaluates Translucent materials using Schlick's approximation for dielectric reflection/refraction.
 */
struct SystemEvaluateTranslucent
{
    using WriteComponent = Usagi::ComponentList<ComponentRay, ComponentPathState, ComponentCameraPath, ComponentLightPath>;
    using ReadComponent  = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene, ServiceRenderState>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();

        size_t count = ray_queue.q_translucent.size();
        if(count == 0) return;

        bool is_cam = (state_svc.pass == 1);

        auto rays   = entities.template get_array<ComponentRay>();
        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();
        auto cpaths = entities.template get_array<ComponentCameraPath>();
        auto lpaths = entities.template get_array<ComponentLightPath>();

        std::mutex merge_mutex;

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            std::vector<uint32_t> loc_next;

            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.q_translucent[i];
                auto & ray   = rays[id];
                auto & hit   = hits[id].hit;
                auto & state = states[id];
                const auto & mat = scene.materials[hit.material_index];

                if (is_cam && cpaths[id].count < MAX_PATH_DEPTH) {
                    cpaths[id].vertices[cpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo, true};
                } else if (!is_cam && lpaths[id].count < MAX_PATH_DEPTH) {
                    lpaths[id].vertices[lpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo, true};
                }

                // Determine if we are entering or exiting the medium
                float ndotd = ray.direction.dot(hit.normal);
                Normal3f outward_normal = ndotd > 0.0f ? -hit.normal : hit.normal;
                float eta = ndotd > 0.0f ? mat.ior : (1.0f / mat.ior);
                float cos_theta_i = std::min(-ndotd, 1.0f);
                if (ndotd > 0.0f) cos_theta_i = ndotd; // Inside to outside

                // Schlick's approximation for Fresnel reflectance
                float r0 = (1.0f - eta) / (1.0f + eta);
                r0 = r0 * r0;
                float R = r0 + (1.0f - r0) * std::pow(1.0f - cos_theta_i, 5.0f);

                // Total Internal Reflection check
                float sin_theta_t_sq = eta * eta * (1.0f - cos_theta_i * cos_theta_i);
                if (sin_theta_t_sq > 1.0f) {
                    R = 1.0f; // TIR
                }

                Vector3f scatter_dir;
                if (state.rng.next_float() < R) {
                    // Reflect
                    scatter_dir = ray.direction - hit.normal * 2.0f * ndotd;
                } else {
                    // Refract
                    float cos_theta_t = std::sqrt(1.0f - sin_theta_t_sq);
                    scatter_dir = ray.direction * eta + outward_normal * (eta * cos_theta_i - cos_theta_t);
                }
                
                if (mat.roughness > 0.0f) {
                    scatter_dir = scatter_dir + random_in_unit_sphere(state.rng) * mat.roughness;
                }
                
                scatter_dir = scatter_dir.normalize();

                // Move origin along the normal relative to the direction of propagation 
                // to prevent self-intersection. Bias slightly inward if refracting!
                float bias_dir = scatter_dir.dot(hit.normal) > 0.0f ? 1.0f : -1.0f;
                ray.origin    = hit.point + hit.normal * (0.001f * bias_dir);
                ray.direction = scatter_dir;
                ray.t_max     = 1000.0f;

                state.throughput = state.throughput * mat.albedo;
                state.depth++;
                state.last_bounce_specular = true;

                if (state.depth < MAX_PATH_DEPTH) {
                    loc_next.push_back(id);
                } else {
                    state.active = false;
                }
            }

            std::lock_guard<std::mutex> lock(merge_mutex);
            ray_queue.next_rays.insert(ray_queue.next_rays.end(), loc_next.begin(), loc_next.end());
        });
    }
};
struct SystemEvaluateLambert
{
    using WriteComponent = Usagi::ComponentList<ComponentRay, ComponentPathState, ComponentCameraPath, ComponentLightPath>;
    using ReadComponent  = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene, ServiceRenderState>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();

        size_t count = ray_queue.q_lambert.size();
        if(count == 0) return;

        bool is_cam = (state_svc.pass == 1);

        auto rays   = entities.template get_array<ComponentRay>();
        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();
        auto cpaths = entities.template get_array<ComponentCameraPath>();
        auto lpaths = entities.template get_array<ComponentLightPath>();

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

                if (is_cam && cpaths[id].count < MAX_PATH_DEPTH) {
                    cpaths[id].vertices[cpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo, false};
                } else if (!is_cam && lpaths[id].count < MAX_PATH_DEPTH) {
                    lpaths[id].vertices[lpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo, false};
                }

                ONB onb;
                onb.build_from_w(hit.normal);
                Vector3f scatter_dir = onb.local(cosine_sample_hemisphere(state.rng.next_float(), state.rng.next_float()));

                ray.origin    = hit.point + hit.normal * 0.001f;
                ray.direction = scatter_dir.normalize();
                ray.t_max     = 1000.0f;

                state.throughput = state.throughput * mat.albedo;
                state.depth++;
                state.last_bounce_specular = false;
                
                if (state.depth < (is_cam ? MAX_PATH_DEPTH : MAX_PATH_DEPTH)) {
                    loc_next.push_back(id);
                } else {
                    state.active = false;
                }
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
    using WriteComponent = Usagi::ComponentList<ComponentRay, ComponentPathState, ComponentCameraPath, ComponentLightPath>;
    using ReadComponent  = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene, ServiceRenderState>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();

        size_t count = ray_queue.q_metal.size();
        if(count == 0) return;

        bool is_cam = (state_svc.pass == 1);

        auto rays   = entities.template get_array<ComponentRay>();
        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();
        auto cpaths = entities.template get_array<ComponentCameraPath>();
        auto lpaths = entities.template get_array<ComponentLightPath>();

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

                if (is_cam && cpaths[id].count < MAX_PATH_DEPTH) {
                    cpaths[id].vertices[cpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo, true};
                } else if (!is_cam && lpaths[id].count < MAX_PATH_DEPTH) {
                    lpaths[id].vertices[lpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo, true};
                }

                Vector3f reflected = ray.direction - hit.normal * 2.0f * ray.direction.dot(hit.normal);
                Vector3f target = reflected;
                
                if(mat.roughness > 0.0f) {
                    target = target + random_in_unit_sphere(state.rng) * mat.roughness;
                }

                if(target.dot(hit.normal) > 0.0f) {
                    ray.direction = target.normalize();
                    ray.origin    = hit.point + hit.normal * 0.001f;
                    ray.t_max     = 1000.0f;

                    state.throughput = state.throughput * mat.albedo;
                    state.depth++;
                    state.last_bounce_specular = true;
                    
                    if (state.depth < (is_cam ? MAX_PATH_DEPTH : MAX_PATH_DEPTH)) {
                        loc_next.push_back(id);
                    } else {
                        state.active = false;
                    }
                } else {
                    state.active = false;
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
    using WriteComponent = Usagi::ComponentList<ComponentPathState, ComponentCameraPath>;
    using ReadComponent  = Usagi::ComponentList<ComponentRayHit>;
    using ReadService    = Usagi::ComponentList<ServiceScene, ServiceRenderState>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();

        size_t count = ray_queue.q_light.size();
        if(count == 0) return;

        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();
        auto cpaths = entities.template get_array<ComponentCameraPath>();

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.q_light[i];
                auto & hit   = hits[id].hit;
                auto & state = states[id];
                const auto & mat = scene.materials[hit.material_index];

                if (state_svc.pass == 1) { // Removed last_bounce_specular check to cover diffuse direct hits
                    auto & cpath = cpaths[id];
                    
                    PathVertex path[MAX_PATH_DEPTH + 1];
                    path[0] = {hit.point, hit.normal, {1,1,1}, {PI,PI,PI}, false};
                    for (int j = 0; j < cpath.count; ++j) {
                        path[1 + j] = cpath.vertices[cpath.count - 1 - j];
                    }
                    
                    float light_area = scene.get_light_area();
                    float mis_weight = compute_mis_weight(path, cpath.count + 1, 0, light_area);
                    
                    cpath.direct_emission = state.throughput * mat.emission * mis_weight;
                }
                state.active = false;
            }
        });
    }
};

/*
 * Shio: Connects Path Vertices
 */
struct SystemConnectPaths
{
    using ReadComponent  = Usagi::ComponentList<ComponentCameraPath, ComponentLightPath, ComponentPathState>;
    using WriteService   = Usagi::ComponentList<ServiceFilm>;
    using ReadService    = Usagi::ComponentList<ServiceRenderState, ServiceScheduler, ServiceScene>;

    void update(auto && entities, auto && services)
    {
        auto & state_svc = services.template get<ServiceRenderState>();
        if (state_svc.pass != 4) return;

        auto & film      = services.template get<ServiceFilm>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & scene     = services.template get<ServiceScene>();

        size_t count = entities.size();
        auto cpaths = entities.template get_array<ComponentCameraPath>();
        auto lpaths = entities.template get_array<ComponentLightPath>();
        auto states = entities.template get_array<ComponentPathState>();

        scheduler.host->parallel_for(count, 4096, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                auto & cpath = cpaths[i];
                auto & lpath = lpaths[i];
                auto & state = states[i];

                float light_area = scene.get_light_area();
                Color3f total_L = {0,0,0};

                // Add explicit camera direct hit emission
                if (cpath.direct_emission.x > 0.0f || cpath.direct_emission.y > 0.0f || cpath.direct_emission.z > 0.0f) {
                     total_L += cpath.direct_emission;
                }

                for (int t = 1; t <= cpath.count; ++t) {
                    auto & cv = cpath.vertices[t - 1];
                    if (cv.is_delta) continue;

                    for (int s = 1; s <= lpath.count; ++s) {
                        auto & lv = lpath.vertices[s - 1];
                        if (lv.is_delta) continue;

                        Vector3f diff = lv.p - cv.p;
                        float dist2 = diff.length_squared();
                        if (dist2 < 0.00001f) continue;
                        float dist = std::sqrt(dist2);
                        Vector3f L = diff * (1.0f / dist);

                        float ndotl_c = cv.n.dot(L);
                        float ndotl_l = lv.n.dot(-L);

                        if (ndotl_c > 0.0f && ndotl_l > 0.0f) {
                            if (!scene.intersect(cv.p + cv.n * 0.001f, L, dist - 0.002f)) {
                                float G = (ndotl_c * ndotl_l) / dist2;
                                Color3f brdf_c = cv.albedo * (1.0f / PI);
                                Color3f brdf_l = lv.albedo * (1.0f / PI);
                                Color3f contrib = cv.beta * brdf_c * G * brdf_l * lv.beta;
                                
                                PathVertex path[MAX_PATH_DEPTH * 2];
                                for (int j = 0; j < s; ++j) path[j] = lpath.vertices[j];
                                for (int j = 0; j < t; ++j) path[s + j] = cpath.vertices[t - 1 - j];
                                
                                float mis_weight = compute_mis_weight(path, s + t, s, light_area);
                                total_L += contrib * mis_weight;
                            }
                        }
                    }
                }

                film.add_sample(state.sample_x, state.sample_y, total_L);
            }
        });
    }
};

/*
 * Shio: Cyclic Coordinator for BDPT Path Tracing Loop
 */
struct SystemPathTracingCoordinator
{
    using ReadService  = Usagi::ComponentList<>;
    using WriteService = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler, ServiceRenderState>;

    void update(auto && entities, auto && services)
    {
        [[maybe_unused]]
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();

        // Start pass
        if (state_svc.pass == 0) {
            SystemGenerateRays sys_gen;
            sys_gen.update(entities, services);
            state_svc.pass = 1;
            scheduler.host->submit_deferred_task([this, &entities, &services]() {
                this->update_bounce_graph(entities, services);
            });
        } else if (state_svc.pass == 2) {
            SystemGenerateRays sys_gen;
            sys_gen.update(entities, services);
            state_svc.pass = 3;
            scheduler.host->submit_deferred_task([this, &entities, &services]() {
                this->update_bounce_graph(entities, services);
            });
        } else if (state_svc.pass == 4) {
            SystemConnectPaths sys_conn;
            sys_conn.update(entities, services);
            state_svc.pass = 0; // Next frame
            // Only submit back to the DAG executor, not inline recursive
        }
    }

    void update_bounce_graph(auto && entities, auto && services)
    {
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();

        if (ray_queue.active_rays.empty()) {
            if(state_svc.pass == 1) {
                state_svc.pass = 2;
                scheduler.host->submit_deferred_task([this, &entities, &services]() {
                    this->update(entities, services);
                });
            } else if (state_svc.pass == 3) {
                state_svc.pass = 4;
                scheduler.host->submit_deferred_task([this, &entities, &services]() {
                    this->update(entities, services);
                });
            }
            return;
        }

        SystemIntersectRays sys_intersect;
        sys_intersect.update(entities, services);

        SystemEvaluateLambert sys_lambert;
        sys_lambert.update(entities, services);

        SystemEvaluateMetal sys_metal;
        sys_metal.update(entities, services);

        SystemEvaluateTranslucent sys_translucent;
        sys_translucent.update(entities, services);

        SystemEvaluateLight sys_light;
        sys_light.update(entities, services);

        if(!ray_queue.next_rays.empty())
        {
            std::swap(ray_queue.active_rays, ray_queue.next_rays);
            ray_queue.next_rays.clear();

            scheduler.host->submit_deferred_task([this, &entities, &services]() {
                this->update_bounce_graph(entities, services);
            });
        }
        else
        {
            ray_queue.active_rays.clear();
            if(state_svc.pass == 1) {
                state_svc.pass = 2;
                scheduler.host->submit_deferred_task([this, &entities, &services]() {
                    this->update(entities, services);
                });
            } else if (state_svc.pass == 3) {
                state_svc.pass = 4;
                scheduler.host->submit_deferred_task([this, &entities, &services]() {
                    this->update(entities, services);
                });
            }
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
                double weight = film.pixels[idx].weight.load(std::memory_order_relaxed);

                Color3f color = {0, 0, 0};
                if (weight > 0.0) {
                    double inv_w = 1.0 / weight;
                    // Because each path tracer pass naturally deposits 1 unit of weight *per frame* over the filter radius,
                    // dividing the total accumulated color by the *total weight splatted* computes the expectation!
                    // Dividing by frame count AGAIN darkens the image linearly over time towards 0! We ONLY divide by weight.
                    color.x = static_cast<float>(film.pixels[idx].r.load(std::memory_order_relaxed) * inv_w);
                    color.y = static_cast<float>(film.pixels[idx].g.load(std::memory_order_relaxed) * inv_w);
                    color.z = static_cast<float>(film.pixels[idx].b.load(std::memory_order_relaxed) * inv_w);
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
