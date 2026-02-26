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

    Vector3f & operator-=(const Vector3f & o)
    {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
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

/*
 * Shio: Generates well-distributed points using radical inverse (Halton).
 * Replaces pure white-noise clustering with low-discrepancy sequences
 * that visually converge far smoother.
 */
inline float radical_inverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f; // / 0x100000000
}

inline float halton(uint32_t index, uint32_t base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f = f / (float)base;
        r = r + f * (float)(index % base);
        index = index / base;
    }
    return r;
}

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
    // Shio: Efficient uniform mapping from unit cube to sphere interior
    // using purely uncorrelated PCG32 to avoid multi-dimensional structural artifacts.
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float u3 = rng.next_float();
    
    float z = 1.0f - 2.0f * u1;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI * u2;
    Vector3f n = { r * std::cos(phi), r * std::sin(phi), z };
    
    // Scale by cubic root to fill the sphere uniformly (not just shell)
    float r3 = std::cbrt(u3);
    return n * r3;
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
    bool         is_moon = false;
    float        density = 0.0f; // Volumetric density for Translucent SSS
};

struct HitRecord
{
    float    t;
    Vector3f point;
    Normal3f normal;
    int      material_index;
    Vector3f obj_center;
    bool     front_face;
    
    inline void set_face_normal(const Vector3f& r_dir, const Normal3f& outward_normal) {
        front_face = r_dir.dot(outward_normal) < 0.0f;
        normal = front_face ? outward_normal : outward_normal * -1.0f;
    }
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

    void sample_surface(SamplerPCG32& rng, Vector3f& p, Normal3f& n, float& pdf) const
    {
        float z = 1.0f - 2.0f * rng.next_float();
        float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        float phi = 2.0f * PI * rng.next_float();
        n = { r * std::cos(phi), r * std::sin(phi), z };
        p = center + n * radius;
        pdf = 1.0f / (4.0f * PI * radius * radius);
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
        float t0 = -100000.0f;
        float t1 = 100000.0f;

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
        
        // Shio: Now correctly returns the exit hit (t1) if the ray starts inside (t0 < t_min)!
        if (t0 >= t_min && t0 <= t_max) return t0;
        if (t1 >= t_min && t1 <= t_max) return t1;
        
        return std::nullopt;
    }

    Normal3f get_normal(const Vector3f & p) const
    {
        // Robust normal calculation immune to floating point corner epsilons
        Vector3f c = (min + max) * 0.5f;
        Vector3f local_p = p - c;
        Vector3f extents = (max - min) * 0.5f;
        Vector3f n = { local_p.x / extents.x, local_p.y / extents.y, local_p.z / extents.z };
        Vector3f abs_n = { std::abs(n.x), std::abs(n.y), std::abs(n.z) };
        if (abs_n.x >= abs_n.y && abs_n.x >= abs_n.z) return { n.x > 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f };
        if (abs_n.y >= abs_n.x && abs_n.y >= abs_n.z) return { 0.0f, n.y > 0.0f ? 1.0f : -1.0f, 0.0f };
        return { 0.0f, 0.0f, n.z > 0.0f ? 1.0f : -1.0f };
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

/*
 * Shio: AABB structure to accelerate ray intersections using a BVH.
 */
struct AABB {
    Vector3f min = { 1e30f, 1e30f, 1e30f };
    Vector3f max = { -1e30f, -1e30f, -1e30f };

    void expand(const Vector3f & p) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }

    void expand(const AABB & b) {
        expand(b.min);
        expand(b.max);
    }

    Vector3f center() const { return (min + max) * 0.5f; }

    float surface_area() const {
        Vector3f d = max - min;
        if (d.x < 0.0f || d.y < 0.0f || d.z < 0.0f) return 0.0f;
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    bool intersect(const Vector3f & o, const Vector3f & inv_d, float t_min, float t_max) const {
        float t0 = -1e30f;
        float t1 = 1e30f;
        for(int i = 0; i < 3; ++i) {
            float tNear = (min[i] - o[i]) * inv_d[i];
            float tFar  = (max[i] - o[i]) * inv_d[i];
            if(inv_d[i] < 0.0f) std::swap(tNear, tFar);
            t0 = std::max(t0, tNear);
            t1 = std::min(t1, tFar);
            if(t0 > t1) return false;
        }
        return (t0 <= t_max && t1 >= t_min);
    }
};

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
    uint32_t     sample_index; // For low-discrepancy evaluation
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
    
    // Shio: Boot extremely fast with exactly num_cores proportion of rays (e.g. 1/8th of screen on 32-core)
    int ray_budget = (800 * 600) / std::max<int>(1, std::thread::hardware_concurrency() / 4); 
    
    std::vector<uint32_t> active_pixels;

    // Fast inline PRNG for unbiased random pixel selection during sparse rendering
    uint32_t xorshift_state = 1337;
    uint32_t next_u32() {
        uint32_t x = xorshift_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        xorshift_state = x;
        return x;
    }
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

struct ServiceCamera
{
    Vector3f position = { 0.0f, 5.0f, -18.0f };
    float yaw = 0.0f;
    float pitch = 0.0f;
    std::atomic<bool> moved = false;

    Vector3f forward() const {
        return { std::cos(pitch) * std::sin(yaw), -std::sin(pitch), std::cos(pitch) * std::cos(yaw) };
    }
    Vector3f right() const {
        return { std::cos(yaw), 0.0f, -std::sin(yaw) };
    }
    Vector3f up() const {
        return forward().cross(right()).normalize();
    }
};

struct ServiceTime
{
    float current_time = 0.0f;
    bool is_paused = true;
};

/*
 * Shio: Thread-safe framebuffer supporting Temporal Anti-Aliasing (TAA)
 * with Variance/Neighborhood Clamping to eliminate ghosting and noise.
 */
struct ServiceFilm
{
    struct Pixel {
        std::atomic<double> direct_r{0}, direct_g{0}, direct_b{0}, direct_w{0};
        std::atomic<double> indirect_r{0}, indirect_g{0}, indirect_b{0}, indirect_w{0};
    };

    std::vector<Pixel> pixels;
    std::vector<Pixel> expired_pixels; // Double-buffering state to smoothly decay motion-blur without blacking out!

    int width = 0;
    int height = 0;
    float filter_radius = 2.0f; // Mitchell filter radius
    float B = 1.0f / 3.0f;
    float C = 1.0f / 3.0f;
    float sample_weight_multiplier = 1.0f;
    float expired_decay_scalar = 1.25f; // User tunable: 25% faster decay for expired ghost buffer

    void init(int w, int h, float rx = 2.0f, float ry = 2.0f)
    {
        width = w;
        height = h;
        pixels = std::vector<Pixel>(w * h);
        expired_pixels = std::vector<Pixel>(w * h);
        filter_radius = rx;
    }

    void swap_to_expired() {
        for(size_t i = 0; i < pixels.size(); ++i) {
            expired_pixels[i].direct_r.store(pixels[i].direct_r.load());
            expired_pixels[i].direct_g.store(pixels[i].direct_g.load());
            expired_pixels[i].direct_b.store(pixels[i].direct_b.load());
            expired_pixels[i].direct_w.store(pixels[i].direct_w.load());
            expired_pixels[i].indirect_r.store(pixels[i].indirect_r.load());
            expired_pixels[i].indirect_g.store(pixels[i].indirect_g.load());
            expired_pixels[i].indirect_b.store(pixels[i].indirect_b.load());
            expired_pixels[i].indirect_w.store(pixels[i].indirect_w.load());

            pixels[i].direct_r.store(0.0);
            pixels[i].direct_g.store(0.0);
            pixels[i].direct_b.store(0.0);
            pixels[i].direct_w.store(0.0);
            pixels[i].indirect_r.store(0.0);
            pixels[i].indirect_g.store(0.0);
            pixels[i].indirect_b.store(0.0);
            pixels[i].indirect_w.store(0.0);
        }
    }

    void clear_direct() {
        for(auto& p : pixels) {
            p.direct_r.store(0.0, std::memory_order_relaxed);
            p.direct_g.store(0.0, std::memory_order_relaxed);
            p.direct_b.store(0.0, std::memory_order_relaxed);
            p.direct_w.store(0.0, std::memory_order_relaxed);
        }
    }

    // Shio: PBRT-style Mitchell-Netravali filter (frequency domain properties)
    float evaluate_filter(float x, float y) const {
        auto mitchell_1d = [this](float v) {
            v = std::abs(2.0f * v / filter_radius);
            if (v > 2.0f) return 0.0f;
            float v2 = v * v;
            float v3 = v * v * v;
            if (v < 1.0f) {
                return (1.0f / 6.0f) * ((12.0f - 9.0f * B - 6.0f * C) * v3 + (-18.0f + 12.0f * B + 6.0f * C) * v2 + (6.0f - 2.0f * B));
            } else {
                return (1.0f / 6.0f) * ((-B - 6.0f * C) * v3 + (6.0f * B + 30.0f * C) * v2 + (-12.0f * B - 48.0f * C) * v + (8.0f * B + 24.0f * C));
            }
        };
        return mitchell_1d(x) * mitchell_1d(y);
    }

    void add_sample(float px, float py, const Color3f& L_in, bool is_direct) {
        // Shio: Firefly Clamping & NaN prevention (Essential for BDPT to prevent massive variance spikes)
        Color3f L = L_in;
        if (std::isnan(L.x) || std::isnan(L.y) || std::isnan(L.z)) return;

        // ONLY clamp indirect light (bounces). 
        // Direct light handles primary rays (looking directly at the sun: 2000.0f)
        // If we clamp direct light, the sun turns dark grey or black!
        if (!is_direct) {
            float lum = L.x * 0.2126f + L.y * 0.7152f + L.z * 0.0722f;
            float max_lum = 100.0f; 
            if (lum > max_lum) {
                float scale = max_lum / lum;
                L.x *= scale;
                L.y *= scale;
                L.z *= scale;
            }
        }

        // Continuous pixel coordinates
        int x0 = std::max(0, (int)std::ceil(px - 0.5f - filter_radius));
        int x1 = std::min(width - 1, (int)std::floor(px - 0.5f + filter_radius));
        int y0 = std::max(0, (int)std::ceil(py - 0.5f - filter_radius));
        int y1 = std::min(height - 1, (int)std::floor(py - 0.5f + filter_radius));

        auto atomic_add = [](std::atomic<double>& target, double val) {
            double old = target.load(std::memory_order_relaxed);
            while(!target.compare_exchange_weak(old, old + val, std::memory_order_relaxed));
        };

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                float dx = px - 0.5f - x;
                float dy = py - 0.5f - y;
                float weight = evaluate_filter(dx, dy) * sample_weight_multiplier; // Shio: Energy conservation for sparse rendering
                if (weight != 0.0f) {
                    int idx = y * width + x;
                    if (is_direct) {
                        atomic_add(pixels[idx].direct_r, static_cast<double>(L.x * weight));
                        atomic_add(pixels[idx].direct_g, static_cast<double>(L.y * weight));
                        atomic_add(pixels[idx].direct_b, static_cast<double>(L.z * weight));
                        atomic_add(pixels[idx].direct_w, static_cast<double>(weight));
                    } else {
                        atomic_add(pixels[idx].indirect_r, static_cast<double>(L.x * weight));
                        atomic_add(pixels[idx].indirect_g, static_cast<double>(L.y * weight));
                        atomic_add(pixels[idx].indirect_b, static_cast<double>(L.z * weight));
                        atomic_add(pixels[idx].indirect_w, static_cast<double>(weight));
                    }
                }
            }
        }
    }

    void apply_ema_decay(double decay_factor, double expired_decay = 0.5) {
        for(auto& p : pixels) {
            auto atomic_scale = [](std::atomic<double>& target, double factor) {
                double old = target.load(std::memory_order_relaxed);
                while(!target.compare_exchange_weak(old, old * factor, std::memory_order_relaxed));
            };
            
            // Shio: To prevent the "strobe blindness" from sparse dynamic rendering when moving, 
            // we apply the explicit decay_factor to BOTH direct and indirect evenly!
            atomic_scale(p.direct_r, decay_factor);
            atomic_scale(p.direct_g, decay_factor);
            atomic_scale(p.direct_b, decay_factor);
            atomic_scale(p.direct_w, decay_factor);

            atomic_scale(p.indirect_r, decay_factor);
            atomic_scale(p.indirect_g, decay_factor);
            atomic_scale(p.indirect_b, decay_factor);
            atomic_scale(p.indirect_w, decay_factor);
        }

        // Extremely aggressive decay for the motion-blurred ghost buffer.
        // It disappears completely within ~3-5 frames (50% per frame),
        // leaving only the crisp new samples behind without ever dipping to zero-energy black!
        for(auto& p : expired_pixels) {
            auto atomic_scale = [](std::atomic<double>& target, double factor) {
                double old = target.load(std::memory_order_relaxed);
                while(!target.compare_exchange_weak(old, old * factor, std::memory_order_relaxed));
            };
            
            atomic_scale(p.direct_r, expired_decay);
            atomic_scale(p.direct_g, expired_decay);
            atomic_scale(p.direct_b, expired_decay);
            atomic_scale(p.direct_w, expired_decay);

            atomic_scale(p.indirect_r, expired_decay);
            atomic_scale(p.indirect_g, expired_decay);
            atomic_scale(p.indirect_b, expired_decay);
            atomic_scale(p.indirect_w, expired_decay);
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
    Vector3f              sun_dir = {0, 1, 0};
    Vector3f              moon_dir = {0, -1, 0};

    // Shio: BVH data structures
    struct PrimRef {
        int type; // 0: sphere, 1: box
        int index;
        AABB bounds;
    };

    struct BVHNode {
        AABB bounds;
        int left_child = -1;
        int right_child = -1;
        int first_prim = -1;
        int prim_count = 0;
        
        bool is_leaf() const { return prim_count > 0; }
    };

    std::vector<PrimRef> bvh_prims;
    std::vector<BVHNode> bvh_nodes;

        // Shio: Rebalance and optimize the BVH using a Surface Area Heuristic (SAH) build.
        // This optimization method can be called dynamically at runtime if new objects
        // are added or if existing objects have moved significantly to rebalance the tree.
        void optimize_bvh() {
            bvh_prims.clear();
            for (int i = 0; i < spheres.size(); ++i) {
                AABB b;
                b.expand(spheres[i].center - Vector3f{spheres[i].radius, spheres[i].radius, spheres[i].radius});
                b.expand(spheres[i].center + Vector3f{spheres[i].radius, spheres[i].radius, spheres[i].radius});
                bvh_prims.push_back({0, i, b});
            }
            for (int i = 0; i < boxes.size(); ++i) {
                AABB b;
                b.expand(boxes[i].min);
                b.expand(boxes[i].max);
                bvh_prims.push_back({1, i, b});
            }
            
            bvh_nodes.clear();
            if (bvh_prims.empty()) return;
    
            bvh_nodes.emplace_back();
            subdivide_bvh_sah(0, 0, bvh_prims.size());
        }
    
        void subdivide_bvh_sah(int node_idx, int first, int count) {
            auto& node = bvh_nodes[node_idx];
            node.first_prim = first;
            node.prim_count = count;
            
            for (int i = 0; i < count; ++i) {
                node.bounds.expand(bvh_prims[first + i].bounds);
            }
            
            if (count <= 2) return; // Leaf threshold
            
            // Find best split using SAH
            float best_cost = 1e30f;
            int best_axis = -1;
            float best_split_pos = 0.0f;
            const int BINS = 8;
            
            for (int axis = 0; axis < 3; ++axis) {
                float min_centroid = 1e30f;
                float max_centroid = -1e30f;
                for (int i = 0; i < count; ++i) {
                    float c = bvh_prims[first + i].bounds.center()[axis];
                    min_centroid = std::min(min_centroid, c);
                    max_centroid = std::max(max_centroid, c);
                }
                
                if (min_centroid == max_centroid) continue;
                
                struct Bin {
                    AABB bounds;
                    int count = 0;
                } bins[BINS];
                
                float scale = BINS / (max_centroid - min_centroid);
                for (int i = 0; i < count; ++i) {
                    float c = bvh_prims[first + i].bounds.center()[axis];
                    int bin_idx = std::min(BINS - 1, std::max(0, (int)((c - min_centroid) * scale)));
                    if (bin_idx == BINS) bin_idx = BINS - 1; // Safeguard against floating point rounding
                    bins[bin_idx].bounds.expand(bvh_prims[first + i].bounds);
                    bins[bin_idx].count++;
                }
                
                float left_area[BINS - 1];
                int left_count[BINS - 1];
                AABB left_box;
                int sum_count = 0;
                for (int i = 0; i < BINS - 1; ++i) {
                    left_box.expand(bins[i].bounds);
                    sum_count += bins[i].count;
                    left_area[i] = left_box.surface_area();
                    left_count[i] = sum_count;
                }
                
                AABB right_box;
                sum_count = 0;
                for (int i = BINS - 1; i > 0; --i) {
                    right_box.expand(bins[i].bounds);
                    sum_count += bins[i].count;
                    
                    float cost = left_count[i - 1] * left_area[i - 1] + sum_count * right_box.surface_area();
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_axis = axis;
                        best_split_pos = min_centroid + i * (max_centroid - min_centroid) / BINS;
                    }
                }
            }
            
            float node_area = node.bounds.surface_area();
            float leaf_cost = count * node_area;
            
            if (best_cost >= leaf_cost || best_axis == -1) {
                return; // Create leaf
            }
            
            int i = first;
            int j = first + count - 1;
            while (i <= j) {
                if (bvh_prims[i].bounds.center()[best_axis] < best_split_pos) {
                    i++;
                } else {
                    std::swap(bvh_prims[i], bvh_prims[j]);
                    j--;
                }
            }
            
            int left_count = i - first;
            if (left_count == 0 || left_count == count) {
                left_count = count / 2;
            }
            
            int left_child_idx = bvh_nodes.size();
            bvh_nodes.emplace_back();
            int right_child_idx = bvh_nodes.size();
            bvh_nodes.emplace_back();
            
                    bvh_nodes[node_idx].left_child = left_child_idx;
                    bvh_nodes[node_idx].right_child = right_child_idx;
                    bvh_nodes[node_idx].prim_count = 0; // Internal node
                    
                    subdivide_bvh_sah(left_child_idx, first, left_count);
                    subdivide_bvh_sah(right_child_idx, first + left_count, count - left_count);
                }
            
                Color3f evaluate_emission(const Material& mat, const Vector3f& p, const Vector3f& center) const {        if (!mat.is_moon) return mat.emission;
        
        // Procedural craters/surface noise
        Vector3f local = (p - center).normalize();
        float n = std::sin(local.x * 30.0f) * std::cos(local.y * 30.0f) * std::sin(local.z * 30.0f);
        n += 0.5f * std::sin(local.x * 60.0f + local.y * 60.0f);
        float noise = (n + 1.5f) * 0.4f;
        noise = std::clamp(noise, 0.4f, 1.0f); // Prevents the moon from having jet-black voids
        
        return mat.emission * noise;
    }

    Color3f evaluate_sky(const Vector3f& view_dir) const {
        if (view_dir.y <= 0.0f) return {0,0,0};

        Vector3f beta_r = {0.0038f, 0.0135f, 0.0331f}; 
        Vector3f beta_m = {0.0210f, 0.0210f, 0.0210f};
        Vector3f beta_sum = { beta_r.x + beta_m.x, beta_r.y + beta_m.y, beta_r.z + beta_m.z };

        float v_y = std::max(0.001f, view_dir.y);
        float opt_depth_v = 1.0f / v_y;

        auto calc_sky_for_light = [&](const Vector3f& l_dir, const Color3f& intensity) -> Color3f {
            if (l_dir.y <= 0.0f) return {0,0,0};
            float cos_theta = view_dir.dot(l_dir);
            float rayleigh_phase = 0.75f * (1.0f + cos_theta * cos_theta);
            
            float g = 0.98f;
            float mie_phase = 1.5f * ((1.0f - g*g) / (2.0f + g*g)) * (1.0f + cos_theta*cos_theta) / std::pow(1.0f + g*g - 2.0f*g*cos_theta + 0.001f, 1.5f);

            float s_y = std::max(0.001f, l_dir.y);
            float opt_depth_s = 1.0f / s_y;

            Vector3f tau = {
                beta_sum.x * (opt_depth_v + opt_depth_s) * 5.0f,
                beta_sum.y * (opt_depth_v + opt_depth_s) * 5.0f,
                beta_sum.z * (opt_depth_v + opt_depth_s) * 5.0f
            };
            Vector3f scatter_attenuation = { std::exp(-tau.x), std::exp(-tau.y), std::exp(-tau.z) };

            Vector3f scatter = {
                (beta_r.x * rayleigh_phase + beta_m.x * mie_phase),
                (beta_r.y * rayleigh_phase + beta_m.y * mie_phase),
                (beta_r.z * rayleigh_phase + beta_m.z * mie_phase)
            };
            
            // Proper physically-based in-scattering equation
            Vector3f sky = {
                (scatter.x / beta_sum.x) * (1.0f - scatter_attenuation.x),
                (scatter.y / beta_sum.y) * (1.0f - scatter_attenuation.y),
                (scatter.z / beta_sum.z) * (1.0f - scatter_attenuation.z)
            };
            
            // The sky itself also receives less light as the light source sets
            Vector3f l_tau = { beta_sum.x * opt_depth_s * 5.0f, beta_sum.y * opt_depth_s * 5.0f, beta_sum.z * opt_depth_s * 5.0f };
            Vector3f l_attenuation = { std::exp(-l_tau.x), std::exp(-l_tau.y), std::exp(-l_tau.z) };
            
            return {
                sky.x * l_attenuation.x * intensity.x,
                sky.y * l_attenuation.y * intensity.y,
                sky.z * l_attenuation.z * intensity.z
            };
        };

        Color3f sun_sky = calc_sky_for_light(sun_dir, {20.0f, 20.0f, 20.0f});
        Color3f moon_sky = calc_sky_for_light(moon_dir, {0.2f, 0.24f, 0.3f}); // Much dimmer moonlight

        // Shio: Sun and Moon disc evaluation for primary background rays.
        // The raw base emission of the sun is 2000.0, but to see it turn orange
        // as it sets, it *must* undergo the same transmission attenuation as the scattered sky!
        float sun_angular_radius = 0.045f;
        float cos_theta_sun = view_dir.dot(sun_dir);
        if (cos_theta_sun > std::cos(sun_angular_radius) && sun_dir.y > 0.0f) {
            float s_y = std::max(0.001f, sun_dir.y);
            Vector3f s_tau = { beta_sum.x * (1.0f / s_y) * 5.0f, beta_sum.y * (1.0f / s_y) * 5.0f, beta_sum.z * (1.0f / s_y) * 5.0f };
            Vector3f sun_attenuation = { std::exp(-s_tau.x), std::exp(-s_tau.y), std::exp(-s_tau.z) };
            
             sun_sky += Color3f{
                 2000.0f * sun_attenuation.x, 
                 1000.0f * sun_attenuation.y, 
                  500.0f * sun_attenuation.z
             };
        }

        float moon_angular_radius = 0.040f;
        float cos_theta_moon = view_dir.dot(moon_dir);
        if (cos_theta_moon > std::cos(moon_angular_radius) && moon_dir.y > 0.0f) {
            
            // The moon is in a different position than the sun, so it has its own unique atmospheric thickness
            float opt_depth_m = 1.0f / std::max(0.001f, moon_dir.y);
            Vector3f m_tau = { beta_sum.x * opt_depth_m * 5.0f, beta_sum.y * opt_depth_m * 5.0f, beta_sum.z * opt_depth_m * 5.0f };
            Vector3f m_attenuation = { std::exp(-m_tau.x), std::exp(-m_tau.y), std::exp(-m_tau.z) };

            moon_sky += Color3f{
                15.0f * m_attenuation.x, 
                18.0f * m_attenuation.y, 
                22.0f * m_attenuation.z
            };
        }

        Color3f total_sky = sun_sky + moon_sky;

        // Add a tiny ambient floor to prevent division-by-zero or pure black
        total_sky.x += 0.001f;
        total_sky.y += 0.002f;
        total_sky.z += 0.005f;

        return total_sky;
    }

    std::optional<HitRecord> intersect(
        const Vector3f & o, const Vector3f & d, float t_max) const
    {
        if (bvh_nodes.empty()) return std::nullopt;

        Vector3f inv_d = {
            d.x == 0.0f ? 1e30f : 1.0f / d.x,
            d.y == 0.0f ? 1e30f : 1.0f / d.y,
            d.z == 0.0f ? 1e30f : 1.0f / d.z
        };

        HitRecord rec;
        rec.t = t_max;
        bool hit_anything = false;

        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;

        while (stack_ptr > 0) {
            int node_idx = stack[--stack_ptr];
            const auto& node = bvh_nodes[node_idx];

            if (!node.bounds.intersect(o, inv_d, 0.001f, rec.t)) {
                continue;
            }

            if (node.is_leaf()) {
                for (int i = 0; i < node.prim_count; ++i) {
                    const auto& prim = bvh_prims[node.first_prim + i];
                    if (prim.type == 0) {
                        const auto& s = spheres[prim.index];
                        if (auto t = s.intersect(o, d, 0.001f, rec.t)) {
                            rec.t              = *t;
                            rec.point          = o + d * rec.t;
                            rec.normal         = (rec.point - s.center).normalize(); // STRICTLY OUTWARD
                            rec.material_index = s.material_index;
                            rec.obj_center     = s.center;
                            hit_anything       = true;
                        }
                    } else {
                        const auto& b = boxes[prim.index];
                        if (auto t = b.intersect(o, d, 0.001f, rec.t)) {
                            rec.t              = *t;
                            rec.point          = o + d * rec.t;
                            rec.normal         = b.get_normal(rec.point); // STRICTLY OUTWARD
                            rec.material_index = b.material_index;
                            rec.obj_center     = b.min + (b.max - b.min) * 0.5f;
                            hit_anything       = true;
                        }
                    }
                }
            } else {
                stack[stack_ptr++] = node.right_child;
                stack[stack_ptr++] = node.left_child;
            }
        }

        if(hit_anything) return rec;
        return std::nullopt;
    }

    bool sample_light(SamplerPCG32& rng, Vector3f& p, Normal3f& n, Color3f& emission, float& pdf) const {
        int light_count = 0;
        for (const auto& s : spheres) if (materials[s.material_index].type == MaterialType::Light) light_count++;
        for (const auto& b : boxes) if (materials[b.material_index].type == MaterialType::Light) light_count++;
        
        if (light_count == 0) return false;

        int light_idx = std::min((int)(rng.next_float() * light_count), light_count - 1);
        
        int current_idx = 0;
        for (const auto& s : spheres) {
            if (materials[s.material_index].type == MaterialType::Light) {
                if (current_idx++ == light_idx) {
                    s.sample_surface(rng, p, n, pdf);
                    pdf /= light_count; // Scale pdf by uniform choice probability
                    
                    Color3f base_emission = evaluate_emission(materials[s.material_index], p, s.center);
                    
                    // Atmosphere attenuation for BDPT light path origins
                    // Shio: p is the position on the sun/moon. The direction FROM the origin TO the light is simply p.normalize()
                    Vector3f dir_to_light = p.normalize();
                    float l_y = std::max(0.001f, dir_to_light.y);
                    float opt_depth_l = 1.0f / l_y;
                    Vector3f tau = {
                        (0.0038f + 0.0210f) * opt_depth_l * 5.0f,
                        (0.0135f + 0.0210f) * opt_depth_l * 5.0f,
                        (0.0331f + 0.0210f) * opt_depth_l * 5.0f
                    };
                    Vector3f attenuation = { std::exp(-tau.x), std::exp(-tau.y), std::exp(-tau.z) };

                    emission = { base_emission.x * attenuation.x, base_emission.y * attenuation.y, base_emission.z * attenuation.z };
                    return true;
                }
            }
        }
        for (const auto& b : boxes) {
            if (materials[b.material_index].type == MaterialType::Light) {
                if (current_idx++ == light_idx) {
                    b.sample_surface(rng, p, n, pdf);
                    pdf /= light_count;
                    Vector3f center = b.min + (b.max - b.min) * 0.5f;
                    emission = evaluate_emission(materials[b.material_index], p, center);
                    return true;
                }
            }
        }
        return false;
    }

    float get_light_area() const {
        float area = 0.0f;
        for (const auto& s : spheres) {
            if (materials[s.material_index].type == MaterialType::Light) {
                area += 4.0f * PI * s.radius * s.radius;
            }
        }
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
    using ReadService    = Usagi::ComponentList<ServiceScheduler, ServiceRenderState, ServiceScene, ServiceCamera>;

    void update(auto && entities, auto && services)
    {
        auto & state_svc = services.template get<ServiceRenderState>();
        if (state_svc.pass != 0 && state_svc.pass != 2) return;

        auto & canvas    = services.template get<ServiceGDICanvasProvider>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & scene     = services.template get<ServiceScene>();
        auto & camera    = services.template get<ServiceCamera>();

        size_t total_pixels = entities.size();
        int actual_count = std::min((int)total_pixels, std::max(1000, state_svc.ray_budget));

        if (state_svc.pass == 0) {
            canvas.frame_count++;
            
            // Randomly select actual_count pixels across the whole screen.
            // This natively prevents "strobe strips" and creates a uniform white-noise
            // visual resolution field when rendering sparsely during fast movement!
            state_svc.active_pixels.resize(actual_count);
            for(int i = 0; i < actual_count; ++i) {
                 state_svc.active_pixels[i] = state_svc.next_u32() % total_pixels;
            }
        }

        ray_queue.active_rays = state_svc.active_pixels;
        size_t count = state_svc.active_pixels.size();

        Vector3f cam_pos      = camera.position;
        Vector3f cam_fwd      = camera.forward();
        Vector3f cam_right    = camera.right();
        Vector3f cam_up       = camera.up();
        float    aspect_ratio = static_cast<float>(canvas.width) / static_cast<float>(canvas.height);

        auto pixels = entities.template get_array<ComponentPixel>();
        auto rays   = entities.template get_array<ComponentRay>();
        auto states = entities.template get_array<ComponentPathState>();
        auto cpaths = entities.template get_array<ComponentCameraPath>();
        auto lpaths = entities.template get_array<ComponentLightPath>();

        bool is_cam = (state_svc.pass == 0);

        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                uint32_t entity_idx = state_svc.active_pixels[i];
                auto & pixel = pixels[entity_idx];
                auto & ray   = rays[entity_idx];
                auto & state = states[entity_idx];

                // Initialize RNG once per pixel
                if(state.rng.inc == 0)
                {
                    // Shio: O(1) temporal seeding instead of looping. The frame count guarantees
                    // a unique temporal stream of stochastic samples over time.
                    state.rng.seed(pixel.y * canvas.width + pixel.x + 1, 1337 + canvas.frame_count.load());
                    // Generate a random pixel-specific offset for the Halton sequence
                    // to prevent structural grid patterns across neighboring pixels
                    state.sample_index = state.rng.next_u32();
                }

                // Advance sample index to guarantee a unique sequence point each frame
                uint32_t current_sample = state.sample_index + canvas.frame_count.load();

                state.throughput = { 1.0f, 1.0f, 1.0f };
                state.radiance   = { 0.0f, 0.0f, 0.0f };
                state.depth      = 0;
                state.active     = true;
                state.last_bounce_specular = true;

                if (is_cam) {
                    // Jitter for anti-aliasing (Base-2 / Base-3 Radical Inverse)
                    float j_x = radical_inverse(current_sample);
                    float j_y = halton(current_sample, 3);
                    state.sample_x = pixel.x + j_x;
                    state.sample_y = pixel.y + j_y;

                    float u = state.sample_x / (float)canvas.width;
                    float v = state.sample_y / (float)canvas.height;

                    // NDC to Camera Space
                    float px = (2.0f * u - 1.0f) * aspect_ratio;
                    float py = 1.0f - 2.0f * v;

                    ray.origin    = cam_pos;
                    ray.direction = (cam_fwd + cam_right * px + cam_up * py).normalize();
                    ray.t_max     = 1000.0f;

                    cpaths[entity_idx].count = 0;
                    cpaths[entity_idx].direct_emission = {0,0,0};

                    // Note: In BDPT, camera is theoretically vertex 0.
                    // We only record surface hits for now.
                } else {
                    lpaths[entity_idx].count = 0;
                    Vector3f p; Normal3f n; Color3f e; float pdf;
                    if (scene.sample_light(state.rng, p, n, e, pdf)) {
                        state.throughput = e * (1.0f / pdf);
                        ONB onb; onb.build_from_w(n);
                        Vector3f dir = onb.local(cosine_sample_hemisphere(state.rng.next_float(), state.rng.next_float()));
                        
                        // First light vertex is the point on the light source.
                        // Setting albedo to PI ensures the 1/PI lambertian factor correctly cancels when connecting directly.
                        lpaths[entity_idx].vertices[0] = {p, n, state.throughput, {PI,PI,PI}, false};
                        lpaths[entity_idx].count = 1;
                        
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
    using ReadService    = Usagi::ComponentList<ServiceScene, ServiceRenderState>;
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler, ServiceFilm>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & film      = services.template get<ServiceFilm>();
        auto & state_svc = services.template get<ServiceRenderState>();

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

        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
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
                    
                    // Shio: Robust front_face resolution
                    // The 'normal' field coming from primitives is strictly OUTWARD-pointing.
                    hit_c.hit.set_face_normal(ray.direction, hit_c.hit.normal);

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
                    // Background Sky
                    hit_c.did_hit = false;
                    state.active = false;
                    if(state_svc.pass == 1) { // Camera paths
                        Color3f sky = scene.evaluate_sky(ray.direction);
                        
                        Color3f total_sky_L = state.radiance + state.throughput * sky;
                        // If this is the primary camera ray, it's direct (cleared every frame). 
                        // If it bounced, it's indirect (EMA decayed).
                        film.add_sample(state.sample_x, state.sample_y, total_sky_L, state.depth == 0);
                    }
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

        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
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

                if (!hit.front_face) {
                    // Shio: Beer's Law (Volumetric Absorption for Jelly/SSS)
                    float dist = hit.t;
                    Vector3f absorption = { 1.0f - mat.albedo.x, 1.0f - mat.albedo.y, 1.0f - mat.albedo.z };
                    state.throughput.x *= std::exp(-absorption.x * dist * mat.density);
                    state.throughput.y *= std::exp(-absorption.y * dist * mat.density);
                    state.throughput.z *= std::exp(-absorption.z * dist * mat.density);
                }

                // Determine if we are entering or exiting the medium using explicit front_face flag
                // hit.normal is guaranteed by SystemIntersectRays to point AGAINST ray.direction
                float eta = hit.front_face ? (1.0f / mat.ior) : mat.ior;
                float cos_theta_i = std::min(hit.normal.dot(ray.direction * -1.0f), 1.0f);

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
                    // Reflect (hit.normal is against ray, so standard reflection works)
                    scatter_dir = ray.direction + hit.normal * 2.0f * cos_theta_i;
                } else {
                    // Refract
                    float cos_theta_t = std::sqrt(1.0f - sin_theta_t_sq);
                    // Since hit.normal is against ray, inward normal is -hit.normal
                    scatter_dir = ray.direction * eta + hit.normal * (eta * cos_theta_i - cos_theta_t);
                }
                
                if (mat.roughness > 0.0f) {
                    scatter_dir = scatter_dir + random_in_unit_sphere(state.rng) * mat.roughness;
                }
                
                scatter_dir = scatter_dir.normalize();

                // Shio: Bias origin OUTWARD from the surface if reflecting, INWARD if refracting.
                // Since hit.normal points against the incoming ray (OUTWARD from the surface we hit):
                // If reflecting, scatter_dir is on the same side as hit.normal. dot > 0.
                // If refracting, scatter_dir is on the opposite side. dot < 0.
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

        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
            std::vector<uint32_t> loc_next;

            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.q_lambert[i];
                auto & ray   = rays[id];
                auto & hit   = hits[id].hit;
                auto & state = states[id];
                const auto & mat = scene.materials[hit.material_index];

                if (is_cam && cpaths[id].count < MAX_PATH_DEPTH) {
                    // Shio: Store the BRDF value (albedo / PI) in the vertex for the BDPT connector to evaluate!
                    cpaths[id].vertices[cpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo * (1.0f / PI), false};
                } else if (!is_cam && lpaths[id].count < MAX_PATH_DEPTH) {
                    lpaths[id].vertices[lpaths[id].count++] = {hit.point, hit.normal, state.throughput, mat.albedo * (1.0f / PI), false};
                }

                ONB onb;
                onb.build_from_w(hit.normal);
                
                // Shio: Independent PCG32 samples correctly map to uniform points 
                // in the 2D domain without Weyl sequence correlation patterns.
                float u1 = state.rng.next_float();
                float u2 = state.rng.next_float();
                
                Vector3f scatter_dir = onb.local(cosine_sample_hemisphere(u1, u2));

                ray.origin    = hit.point + hit.normal * 0.001f;
                ray.direction = scatter_dir.normalize();
                ray.t_max     = 1000.0f;

                // Because PDF = cos(theta) / PI and BRDF = albedo / PI, they cancel out exactly
                // ONLY for the indirect recursive throughput!
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

        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
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
    using WriteService   = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler, ServiceFilm>;

    void update(auto && entities, auto && services)
    {
        auto & scene     = services.template get<ServiceScene>();
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();
        auto & film      = services.template get<ServiceFilm>();

        size_t count = ray_queue.q_light.size();
        if(count == 0) return;

        auto hits   = entities.template get_array<ComponentRayHit>();
        auto states = entities.template get_array<ComponentPathState>();
        auto cpaths = entities.template get_array<ComponentCameraPath>();

        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                uint32_t id  = ray_queue.q_light[i];
                auto & hit   = hits[id].hit;
                auto & state = states[id];
                const auto & mat = scene.materials[hit.material_index];

                if (state_svc.pass == 1) { // Removed last_bounce_specular check to cover diffuse direct hits
                    auto & cpath = cpaths[id];
                    
                    Color3f raw_emission = scene.evaluate_emission(mat, hit.point, hit.obj_center);

                    // Compute atmospheric attenuation from the viewer to the celestial body
                    // Shio: hit.point is on the sun. Direction TO the light is hit.point.normalize()
                    Vector3f dir_to_light = hit.point.normalize();
                    float l_y = std::max(0.001f, dir_to_light.y);
                    float opt_depth_l = 1.0f / l_y;
                    Vector3f tau = {
                        (0.0038f + 0.0210f) * opt_depth_l * 5.0f,
                        (0.0135f + 0.0210f) * opt_depth_l * 5.0f,
                        (0.0331f + 0.0210f) * opt_depth_l * 5.0f
                    };
                    Vector3f attenuation = { std::exp(-tau.x), std::exp(-tau.y), std::exp(-tau.z) };
                    Color3f emission = { raw_emission.x * attenuation.x, raw_emission.y * attenuation.y, raw_emission.z * attenuation.z };

                    if (state.depth == 0) {
                        // Pure direct hits bypass MIS and go straight to the direct buffer!
                        film.add_sample(state.sample_x, state.sample_y, state.radiance + state.throughput * emission, true);
                    } else {
                        PathVertex path[MAX_PATH_DEPTH + 1];
                        // Shio: Light source vertices MUST have an effective BRDF of 1.0 (so albedo = PI to cancel the /PI in ConnectPaths).
                        path[0] = {hit.point, hit.normal, {1,1,1}, {PI,PI,PI}, false};
                        for (int j = 0; j < cpath.count; ++j) {
                            path[1 + j] = cpath.vertices[cpath.count - 1 - j];
                        }
                        
                        float light_area = scene.get_light_area();
                        float mis_weight = compute_mis_weight(path, cpath.count + 1, 0, light_area);
                        
                        cpath.direct_emission = state.throughput * emission * mis_weight;
                    }
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

        size_t count = state_svc.active_pixels.size();
        auto cpaths = entities.template get_array<ComponentCameraPath>();
        auto lpaths = entities.template get_array<ComponentLightPath>();
        auto states = entities.template get_array<ComponentPathState>();

        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                uint32_t entity_idx = state_svc.active_pixels[i];
                auto & cpath = cpaths[entity_idx];
                auto & lpath = lpaths[entity_idx];
                auto & state = states[entity_idx];

                float light_area = scene.get_light_area();
                Color3f indirect_L = {0,0,0};
                Color3f direct_L = {0,0,0};

                // Explicit direct hit (bounced via material evaluations hitting light source directly).
                if (cpath.count > 0 && (cpath.direct_emission.x > 0.0f || cpath.direct_emission.y > 0.0f || cpath.direct_emission.z > 0.0f)) {
                     indirect_L += cpath.direct_emission;
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
                                Color3f brdf_c = cv.albedo;
                                Color3f brdf_l = lv.albedo;
                                Color3f contrib = cv.beta * brdf_c * G * brdf_l * lv.beta;
                                
                                PathVertex path[MAX_PATH_DEPTH * 2];
                                for (int j = 0; j < s; ++j) path[j] = lpath.vertices[j];
                                for (int j = 0; j < t; ++j) path[s + j] = cpath.vertices[t - 1 - j];
                                
                                float mis_weight = compute_mis_weight(path, s + t, s, light_area);
                                
                                // All connection paths are treated as indirect (EMA decayed) since they 
                                // represent scattered light. Only 0-bounce direct eye hits go to direct_L.
                                indirect_L += contrib * mis_weight;
                            }
                        }
                    }
                }

                if (direct_L.x > 0 || direct_L.y > 0 || direct_L.z > 0) film.add_sample(state.sample_x, state.sample_y, direct_L, true);
                if (indirect_L.x > 0 || indirect_L.y > 0 || indirect_L.z > 0) film.add_sample(state.sample_x, state.sample_y, indirect_L, false);
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
    using WriteService = Usagi::ComponentList<ServiceRayQueue, ServiceScheduler, ServiceRenderState, ServiceCamera, ServiceTime>;

    void update(auto && entities, auto && services)
    {
        [[maybe_unused]]
        auto & ray_queue = services.template get<ServiceRayQueue>();
        auto & scheduler = services.template get<ServiceScheduler>();
        auto & state_svc = services.template get<ServiceRenderState>();

        // Start pass
        if (state_svc.pass == 0) {
            auto & scene    = services.template get<ServiceScene>();
            auto & film     = services.template get<ServiceFilm>();
            auto & canvas   = services.template get<ServiceGDICanvasProvider>();
            auto & camera   = services.template get<ServiceCamera>();
            auto & time_svc = services.template get<ServiceTime>();

            // Read the dynamically driven time
            float time = time_svc.current_time;

            // Track if time is actively advancing (e.g., normal playback or forced fast-forward)
            static float last_time = time;
            bool time_moved = (time != last_time);
            last_time = time;
            
            bool camera_moved = camera.moved.exchange(false);
            static bool was_moving = false;

            // Energy compensation for dynamic sparse rendering scaling!
            float fraction = (float)state_svc.ray_budget / (canvas.width * canvas.height);
            film.sample_weight_multiplier = 1.0f / std::max(0.001f, fraction);
            
            // Align expired pixel decay dynamically to the ray spawn rate!
            // If we spawn 10% of rays, fraction = 0.1. Decay base = 0.9.
            // A tunable multiplier applies a 25% faster decay rate (0.9 ^ 1.25) so it aggressively fades out.
            double expired_decay = std::clamp<double>(std::pow(1.0f - fraction, film.expired_decay_scalar), 0.0, 0.99);

            if (camera_moved || time_moved) {
                // Shio: Keep a trace going so user isn't blind during heavy movement
                film.apply_ema_decay(0.6, 0.0); // Wipe expired immediately when moving
                canvas.frame_count = 0;
                was_moving = true;
            } else {
                if (was_moving) {
                    was_moving = false;
                    // The exact moment movement stops:
                    // 1. Shove the entire dirty motion-blurred frame into the rapidly decaying `expired` buffer.
                    // 2. Clear the active buffer to 0.
                    // 3. From now on, the clean active buffer builds up pure sparse samples and replaces the ghost perfectly!
                    film.swap_to_expired();
                    canvas.frame_count = 0;
                }
                
                if (time_svc.is_paused) {
                    // Infinite accumulation (decay = 1.0) merges all frames perfectly when paused and completely still
                    film.apply_ema_decay(1.0, expired_decay);
                } else {
                    film.apply_ema_decay(0.85, expired_decay); // Normal standing decay (0.85). Smooths out the 4-SPP noise while keeping shadows responsive!
                }
            }
            
            // Sun orbits in YZ plane. Z > 0 is out the back wall.
            scene.sun_dir = Vector3f{ 0.0f, std::cos(time), std::sin(time) }.normalize();
            // Moon orbits offset in phase and tilted slightly in X so they never collide
            scene.moon_dir = Vector3f{ 0.3f, std::cos(time + PI), std::sin(time + PI) }.normalize();
            
            if (scene.spheres.size() >= 2) {
                // Sphere 0 is the Sun
                scene.spheres[0].center = scene.sun_dir * 1000.0f;
                
                // Set the base sun emission extremely bright and raw. 
                // The physical scattering equations in evaluate_sky and evaluate_light will naturally
                // scatter blue light away, turning the *remaining* transmitted beam orange without
                // artificially hacking the source color.
                if (scene.sun_dir.y > 0.0f) {
                    scene.materials[scene.spheres[0].material_index].emission = { 2000.0f, 1000.0f, 500.0f };
                } else {
                    scene.materials[scene.spheres[0].material_index].emission = { 0.0f, 0.0f, 0.0f };
                }

                // Sphere 1 is the Moon
                scene.spheres[1].center = scene.moon_dir * 1000.0f;
                
                // Base moon color (pale blue/white). Moon is very dim compared to the sun.
                if (scene.moon_dir.y > 0.0f) {
                    scene.materials[scene.spheres[1].material_index].emission = { 15.0f, 18.0f, 22.0f };
                } else {
                    scene.materials[scene.spheres[1].material_index].emission = { 0.0f, 0.0f, 0.0f };
                }
            }

            scene.optimize_bvh(); // Shio: Rebalance and optimize BVH as dynamic objects have moved!

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

        // Shio: Fast 3x3 Edge-Avoiding Bilateral filter pass using luminance differentials
        // to smooth noise without destroying structural detail.
        size_t chunk_size = std::max<size_t>(1, count / (std::thread::hardware_concurrency() * 4)); scheduler.host->parallel_for(count, chunk_size, [&](size_t start, size_t end) {
            for(size_t i = start; i < end; ++i)
            {
                auto & pixel = pixels[i];
                int idx = pixel.y * canvas.width + pixel.x;

                double d_w = film.pixels[idx].direct_w.load(std::memory_order_relaxed) + film.expired_pixels[idx].direct_w.load(std::memory_order_relaxed);
                double i_w = film.pixels[idx].indirect_w.load(std::memory_order_relaxed) + film.expired_pixels[idx].indirect_w.load(std::memory_order_relaxed);
                double total_w = d_w + i_w;

                Color3f color_center = {0, 0, 0};
                if (total_w > 0.0) {
                    double inv = 1.0 / total_w;
                    
                    double active_sum_r = film.pixels[idx].direct_r.load(std::memory_order_relaxed) + film.pixels[idx].indirect_r.load(std::memory_order_relaxed);
                    double active_sum_g = film.pixels[idx].direct_g.load(std::memory_order_relaxed) + film.pixels[idx].indirect_g.load(std::memory_order_relaxed);
                    double active_sum_b = film.pixels[idx].direct_b.load(std::memory_order_relaxed) + film.pixels[idx].indirect_b.load(std::memory_order_relaxed);
                    
                    double exp_sum_r = film.expired_pixels[idx].direct_r.load(std::memory_order_relaxed) + film.expired_pixels[idx].indirect_r.load(std::memory_order_relaxed);
                    double exp_sum_g = film.expired_pixels[idx].direct_g.load(std::memory_order_relaxed) + film.expired_pixels[idx].indirect_g.load(std::memory_order_relaxed);
                    double exp_sum_b = film.expired_pixels[idx].direct_b.load(std::memory_order_relaxed) + film.expired_pixels[idx].indirect_b.load(std::memory_order_relaxed);

                    color_center.x = static_cast<float>((active_sum_r + exp_sum_r) * inv);
                    color_center.y = static_cast<float>((active_sum_g + exp_sum_g) * inv);
                    color_center.z = static_cast<float>((active_sum_b + exp_sum_b) * inv);
                }

                float lum_center = color_center.x * 0.2126f + color_center.y * 0.7152f + color_center.z * 0.0722f;

                Color3f c_blur = {0, 0, 0};
                float w_blur_sum = 0.0f;
                
                for(int dy = -1; dy <= 1; ++dy) {
                    for(int dx = -1; dx <= 1; ++dx) {
                        int nx = std::clamp(pixel.x + dx, 0, canvas.width - 1);
                        int ny = std::clamp(pixel.y + dy, 0, canvas.height - 1);
                        int nidx = ny * canvas.width + nx;

                        Color3f neighbor = {0, 0, 0};
                        
                        double n_d_w = film.pixels[nidx].direct_w.load(std::memory_order_relaxed) + film.expired_pixels[nidx].direct_w.load(std::memory_order_relaxed);
                        double n_i_w = film.pixels[nidx].indirect_w.load(std::memory_order_relaxed) + film.expired_pixels[nidx].indirect_w.load(std::memory_order_relaxed);
                        double n_total_w = n_d_w + n_i_w;
                        
                        if (n_total_w > 0.0) {
                            double inv = 1.0 / n_total_w;
                            
                            double n_active_sum_r = film.pixels[nidx].direct_r.load(std::memory_order_relaxed) + film.pixels[nidx].indirect_r.load(std::memory_order_relaxed);
                            double n_active_sum_g = film.pixels[nidx].direct_g.load(std::memory_order_relaxed) + film.pixels[nidx].indirect_g.load(std::memory_order_relaxed);
                            double n_active_sum_b = film.pixels[nidx].direct_b.load(std::memory_order_relaxed) + film.pixels[nidx].indirect_b.load(std::memory_order_relaxed);

                            double n_exp_sum_r = film.expired_pixels[nidx].direct_r.load(std::memory_order_relaxed) + film.expired_pixels[nidx].indirect_r.load(std::memory_order_relaxed);
                            double n_exp_sum_g = film.expired_pixels[nidx].direct_g.load(std::memory_order_relaxed) + film.expired_pixels[nidx].indirect_g.load(std::memory_order_relaxed);
                            double n_exp_sum_b = film.expired_pixels[nidx].direct_b.load(std::memory_order_relaxed) + film.expired_pixels[nidx].indirect_b.load(std::memory_order_relaxed);

                            neighbor.x = static_cast<float>((n_active_sum_r + n_exp_sum_r) * inv);
                            neighbor.y = static_cast<float>((n_active_sum_g + n_exp_sum_g) * inv);
                            neighbor.z = static_cast<float>((n_active_sum_b + n_exp_sum_b) * inv);
                        }

                        float lum_neighbor = neighbor.x * 0.2126f + neighbor.y * 0.7152f + neighbor.z * 0.0722f;
                        
                        // Edge-stopping function based on luminance difference
                        float l_diff = lum_neighbor - lum_center;
                        float w_l = std::exp(-(l_diff * l_diff) / 0.1f); // Range sensitivity tuning
                        
                        float w_s = std::exp(-(dx*dx + dy*dy) / 2.0f);
                        float w = w_s * w_l;

                        c_blur += neighbor * w;
                        w_blur_sum += w;
                    }
                }
                
                Color3f color = w_blur_sum > 0.0f ? c_blur * (1.0f / w_blur_sum) : color_center;

                // Protect against negative colors
                color.x = std::max(0.0f, color.x);
                color.y = std::max(0.0f, color.y);
                color.z = std::max(0.0f, color.z);

                // Shio: Robust ACES Filmic Tone Mapping Curve
                // This gracefully handles extremely bright 2000.0+ HDR pixels (Sun Core) 
                auto ACESFilm = [](float x) {
                    float a = 2.51f;
                    float b = 0.03f;
                    float c = 2.43f;
                    float d = 0.59f;
                    float e = 0.14f;
                    return std::clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0f, 1.0f);
                };

                // Exposure adjustment. Lower exposure maps the intense sun/sky into SDR range without blowout
                float exposure = 0.1f;
                color.x *= exposure;
                color.y *= exposure;
                color.z *= exposure;

                color.x = ACESFilm(color.x);
                color.y = ACESFilm(color.y);
                color.z = ACESFilm(color.z);

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
