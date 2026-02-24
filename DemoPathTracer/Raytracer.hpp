#pragma once

#include <algorithm>
#include <cmath>

#include "UsagiCore.hpp"

namespace RT
{

/*
 * Shio: Standard 3D vector structure used for positions, directions, and colors.
 * Includes basic arithmetic operator overloading.
 */
struct Vector3f
{
    float x, y, z;

    Vector3f operator+(const Vector3f & o) const
    {
        return { x + o.x, y + o.y, z + o.z };
    }

    Vector3f operator-(const Vector3f & o) const
    {
        return { x - o.x, y - o.y, z - o.z };
    }

    Vector3f operator*(float s) const { return { x * s, y * s, z * s }; }

    float dot(const Vector3f & o) const { return x * o.x + y * o.y + z * o.z; }

    Vector3f normalize() const
    {
        float len = std::sqrt(dot(*this));
        if(len > 0.000001f) return { x / len, y / len, z / len };
        return { 0.0f, 0.0f, 0.0f };
    }
};

typedef Vector3f Normal3f;
typedef Vector3f Color3f;

// -----------------------------------------------------------------------------
// Components
// -----------------------------------------------------------------------------

/*
 * Shio: Represents a ray in 3D space.
 * Stored in a structure-of-arrays format by ComponentGroup.
 */
struct ComponentRay
{
    Vector3f origin;
    Normal3f direction;
    Color3f  color;   // Accumulates color during traversal
    float    t_max;   // Max distance for intersection
};

/*
 * Shio: Screen-space coordinates for a ray.
 * Used to map the ray back to the display buffer.
 */
struct ComponentPixel
{
    int x, y;
};

// -----------------------------------------------------------------------------
// Services
// -----------------------------------------------------------------------------

/*
 * Shio: Service to provide access to the raw GDI pixel buffer.
 * Systems use this to write the final rendered frame.
 */
struct ServiceGDICanvasProvider
{
    uint32_t * pixel_buffer;
    int        width;
    int        height;
};

// -----------------------------------------------------------------------------
// Systems
// -----------------------------------------------------------------------------

/*
 * Shio: Initializes rays for each pixel based on a simple pinhole camera model.
 */
struct SystemGenerateCameraRays
{
    void update(auto && entities, auto && services)
    {
        auto & canvas = services.template get<ServiceGDICanvasProvider>();

        // Hardcoded camera position
        Vector3f cam_pos      = { 0.0f, 5.0f, -8.0f };
        float    aspect_ratio = static_cast<float>(canvas.width) /
            static_cast<float>(canvas.height);

        // Iterate over all entities with Pixel and Ray components
        entities.template query<ComponentPixel, ComponentRay>()(
            [&](ComponentPixel & pixel, ComponentRay & ray) {
                // Map pixel coordinates to Normalized Device Coordinates (NDC) [-1, 1]
                float px = (2.0f * ((pixel.x + 0.5f) / canvas.width) - 1.0f) *
                    aspect_ratio;
                float py = 1.0f - 2.0f * ((pixel.y + 0.5f) / canvas.height);

                ray.origin    = cam_pos;
                // Simple perspective projection (z = 1.0f is the image plane)
                ray.direction = Vector3f { px, py, 1.0f }.normalize();
                ray.color     = { 0.0f, 0.0f, 0.0f };
                ray.t_max     = 1000.0f; // Far clip plane
            });
    }
};

/*
 * Shio: Performs intersection tests against the scene geometry.
 * The scene is currently hardcoded within the system for simplicity.
 * Calculates simple diffuse shading based on a fixed light direction.
 */
struct SystemEvaluatePhysicalMaterial
{
    void update(auto && entities, auto && services)
    {
        entities.template query<ComponentRay>()([](ComponentRay & ray) {
            [[maybe_unused]]
            float    t_min     = 0.0f;
            float    t_max     = ray.t_max;
            Vector3f normal    = { 0, 0, 0 };
            Color3f  hit_color = { 0, 0, 0 };

            // Floor (y = 0)
            if(ray.direction.y < 0.0f)
            {
                float t = (0.0f - ray.origin.y) / ray.direction.y;
                if(t > 0 && t < t_max)
                {
                    t_max     = t;
                    normal    = { 0, 1, 0 };
                    hit_color = { 0.8f, 0.8f, 0.8f };
                }
            }

            // Left Wall (x = -5)
            if(ray.direction.x < 0.0f)
            {
                float t = (-5.0f - ray.origin.x) / ray.direction.x;
                if(t > 0 && t < t_max)
                {
                    t_max     = t;
                    normal    = { 1, 0, 0 };
                    hit_color = { 0.8f, 0.2f, 0.2f };
                }
            }

            // Right Wall (x = 5)
            if(ray.direction.x > 0.0f)
            {
                float t = (5.0f - ray.origin.x) / ray.direction.x;
                if(t > 0 && t < t_max)
                {
                    t_max     = t;
                    normal    = { -1, 0, 0 };
                    hit_color = { 0.2f, 0.2f, 0.8f };
                }
            }

            // Back Wall (z = 10)
            if(ray.direction.z > 0.0f)
            {
                float t = (10.0f - ray.origin.z) / ray.direction.z;
                if(t > 0 && t < t_max)
                {
                    t_max     = t;
                    normal    = { 0, 0, -1 };
                    hit_color = { 0.8f, 0.8f, 0.8f };
                }
            }

            // Shading
            if(t_max < ray.t_max)
            {
                // Hardcoded directional light
                Vector3f light_dir = Vector3f { 0.0f, 1.0f, -1.0f }.normalize();
                // Lambertian reflection
                float    ndotl     = std::max(0.1f, normal.dot(light_dir));
                ray.color          = hit_color * ndotl;
            }
            else
            {
                // Ambient background color
                ray.color = { 0.05f, 0.05f, 0.05f };
            }
        });
    }
};

/*
 * Shio: Converts the floating-point colors to 32-bit integers (0xRRGGBB)
 * and writes them to the canvas buffer for display.
 */
struct SystemRenderGDICanvas
{
    void update(auto && entities, auto && services)
    {
        auto & canvas = services.template get<ServiceGDICanvasProvider>();

        entities.template query<ComponentPixel, ComponentRay>()(
            [&](ComponentPixel & pixel, ComponentRay & ray) {
                // Tone mapping / clamping
                uint8_t r = static_cast<uint8_t>(
                    std::min(ray.color.x * 255.0f, 255.0f));
                uint8_t g = static_cast<uint8_t>(
                    std::min(ray.color.y * 255.0f, 255.0f));
                uint8_t b = static_cast<uint8_t>(
                    std::min(ray.color.z * 255.0f, 255.0f));

                // Write to linear buffer: y * width + x
                canvas.pixel_buffer[pixel.y * canvas.width + pixel.x] =
                    (r << 16) | (g << 8) | b;
            });
    }
};
} // namespace RT
