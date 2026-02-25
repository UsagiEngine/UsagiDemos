#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#include <atomic>
#include <chrono>
#include <thread>

#include "Raytracer.hpp"

// Global window handle for GDI calls
HWND              g_hWnd = nullptr;
std::atomic<bool> g_RenderActive { true };

/*
 * Shio: Standard Win32 Window Procedure.
 */
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    if(uMsg == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

struct SystemPresentToGDI
{
    void update(auto && entities, auto && services)
    {
        auto & canvas = services.template get<RT::ServiceGDICanvasProvider>();
        HDC    hdc    = GetDC(g_hWnd);
        if(!hdc) return; // Basic validation

        BITMAPINFO bmi              = { };
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = canvas.width;
        bmi.bmiHeader.biHeight      = -canvas.height;
        bmi.bmiHeader.biPlanes      = 1;
        bmi.bmiHeader.biBitCount    = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        SetDIBitsToDevice(hdc,
            0,
            0,
            canvas.width,
            canvas.height,
            0,
            0,
            0,
            canvas.height,
            canvas.pixel_buffer,
            &bmi,
            DIB_RGB_COLORS);
        ReleaseDC(g_hWnd, hdc);
    }
};

/*
 * Shio: Setup the Cornell Box scene geometry.
 */
void SetupCornellBox(RT::ServiceScene & scene)
{
    using namespace RT;

    // Materials
    // 0: White
    scene.materials.push_back(
        { MaterialType::Lambert, { 0.73f, 0.73f, 0.73f }, { 0, 0, 0 }, 0 });
    // 1: Red
    scene.materials.push_back(
        { MaterialType::Lambert, { 0.65f, 0.05f, 0.05f }, { 0, 0, 0 }, 0 });
    // 2: Green
    scene.materials.push_back(
        { MaterialType::Lambert, { 0.12f, 0.45f, 0.15f }, { 0, 0, 0 }, 0 });
    // 3: Light
    scene.materials.push_back({ MaterialType::Light,
        { 1.0f, 1.0f, 1.0f },
        { 15.0f, 15.0f, 15.0f },
        0 });
    // 4: Metal (Slightly rough mirror)
    scene.materials.push_back({ MaterialType::Metal,
        { 0.8f, 0.85f, 0.88f },
        { 0, 0, 0 },
        0.02f });
    // 5: Translucent (Pale Purple Jelly SSS)
    scene.materials.push_back({ MaterialType::Translucent,
        { 0.95f, 0.85f, 0.98f }, // Very pale, bright lilac/purple
        { 0, 0, 0 },
        0.1f, 1.3f, false, 0.3f }); // Lower density to make it significantly more translucent

    // 6: The Sun (Massive light sphere far away)
    scene.materials.push_back({ MaterialType::Light, { 1.0f, 1.0f, 1.0f }, { 15.0f, 15.0f, 15.0f }, 0.0f, 1.0f, false });
    
    // 7: The Moon (Textured secondary light sphere directly opposed to the Sun)
    scene.materials.push_back({ MaterialType::Light, { 1.0f, 1.0f, 1.0f }, { 2.0f, 2.2f, 2.5f }, 0.0f, 1.0f, true });

    // --- Minecraft-style World Materials ---
    // 8: Grass (Lambertian Vivid Green)
    scene.materials.push_back({ MaterialType::Lambert, { 0.3f, 0.7f, 0.2f }, { 0, 0, 0 }, 0.0f });
    // 9: Dirt (Lambertian Brown)
    scene.materials.push_back({ MaterialType::Lambert, { 0.4f, 0.25f, 0.15f }, { 0, 0, 0 }, 0.0f });
    // 10: Wood Log (Lambertian Dark Brown)
    scene.materials.push_back({ MaterialType::Lambert, { 0.25f, 0.15f, 0.08f }, { 0, 0, 0 }, 0.0f });
    // 11: Leaves (Subsurface Translucent Green)
    scene.materials.push_back({ MaterialType::Translucent, { 0.2f, 0.8f, 0.2f }, { 0, 0, 0 }, 0.3f, 1.1f, false, 0.5f });
    // 12: Snow (Highly reflective rough Lambertian with slight blue tint)
    scene.materials.push_back({ MaterialType::Lambert, { 0.9f, 0.95f, 1.0f }, { 0, 0, 0 }, 0.0f });
    // 13: Frozen Lake (Highly glossy Ice)
    scene.materials.push_back({ MaterialType::Translucent, { 0.7f, 0.85f, 0.95f }, { 0, 0, 0 }, 0.005f, 1.31f, false, 0.1f });

    // Cornell Box Walls (approximated with large boxes or planes, using Boxes here)
    float s = 10.0f; // Scale

    // Floor (White)
    scene.boxes.push_back({ { -s, -s - 1.0f, -s }, { s, -s, s }, 0 });
    // Ceiling (White)
    scene.boxes.push_back({ { -s, s, -s }, { s, s + 1.0f, s }, 0 });
    // Left Wall (Metal mirror)
    scene.boxes.push_back({ { -s - 1.0f, -s, -s }, { -s, s, s }, 4 });
    // Right Wall (Green)
    scene.boxes.push_back({ { s, -s, -s }, { s + 1.0f, s, s }, 2 });

    // The celestial bodies
    scene.spheres.push_back({ { 0.0f, 0.0f, 1000.0f }, 45.0f, 6 });
    scene.spheres.push_back({ { 0.0f, 0.0f, -1000.0f }, 40.0f, 7 });

    // Boxes inside
    // Short box
    scene.boxes.push_back(
        { { -3.0f, -s, 2.0f }, { 0.0f, -s + 3.0f, 5.0f }, 0 });
    // Tall box (Jelly)
    scene.boxes.push_back(
        { { 2.0f, -s, -3.0f }, { 5.0f, -s + 6.0f, 0.0f }, 5 });

    // --- Minecraft World Generation ---
    float bs = 2.0f; // Block Size
    float floor_y = -s - 1.0f; // Align with bottom of Cornell Box

    auto add_block = [&](float x, float y, float z, int mat_idx, float scale = 1.0f) {
        float gap = bs * (1.0f - scale) * 0.5f;
        scene.boxes.push_back({
            { x * bs + gap, floor_y + y * bs + gap, z * bs + gap },
            { (x + 1) * bs - gap, floor_y + (y + 1) * bs - gap, (z + 1) * bs - gap },
            mat_idx
        });
    };

    auto get_height = [](int x, int z) {
        float fx = x * 0.15f;
        float fz = z * 0.15f;
        float h = std::sin(fx) * std::cos(fz) * 3.0f + std::sin(fx * 2.5f + fz * 1.5f) * 1.0f;
        return (int)std::floor(h);
    };

    // Massive Underground Base to catch escaping rays efficiently without 1000s of dirt blocks
    scene.boxes.push_back({ { -100.0f, floor_y - 10.0f, -100.0f }, { 100.0f, floor_y - 3.0f, 100.0f }, 9 }); 

    for (int x = -15; x <= 15; ++x) {
        for (int z = -15; z <= 25; ++z) {
            // Leave space for the Cornell Box
            if (x >= -6 && x <= 6 && z >= -6 && z <= 6) continue;

            int h = get_height(x, z);
            
            if (h < 0) {
                // Frozen Lake
                bool is_hole = (std::sin(x * 3.14f) * std::cos(z * 2.71f)) > 0.6f;
                if (!is_hole) {
                    add_block(x, -1, z, 13); // Ice Surface
                } else {
                    add_block(x, -2, z, 13); // Sunken Ice (Fracture/Hole)
                }
            } else {
                // Land
                if (h > 1) {
                    add_block(x, h, z, 12); // Snow capped peaks
                } else {
                    add_block(x, h, z, 8);  // Grassy lowlands
                }
                
                // Add exposed cliff faces (only down to neighbor minimum to save polygons)
                int min_neighbor = std::min({get_height(x-1, z), get_height(x+1, z), get_height(x, z-1), get_height(x, z+1), h});
                for (int y = std::max(-2, min_neighbor); y < h; ++y) {
                    add_block(x, y, z, 9); // Dirt cliff
                }

                // Random Trees
                if (h <= 1 && (std::abs(x * 73 + z * 37) % 45 == 0)) {
                    int tree_h = 3 + (std::abs(x+z) % 3);
                    // Trunk
                    for (int ty = 1; ty <= tree_h; ++ty) {
                        add_block(x, h + ty, z, 10, 0.8f);
                    }
                    // Leaves with physical gaps allowing light transmission!
                    for (int lx = -2; lx <= 2; ++lx) {
                        for (int ly = 0; ly <= 2; ++ly) {
                            for (int lz = -2; lz <= 2; ++lz) {
                                if (std::abs(lx) == 2 && std::abs(lz) == 2 && ly == 2) continue; // Round canopy corners
                                if ((std::abs(lx * 11 + ly * 13 + lz * 17)) % 4 == 0) continue; // Procedural gaps for dappled light
                                add_block(x + lx, h + tree_h - 1 + ly, z + lz, 11, 0.85f);
                            }
                        }
                    }
                }
            }
        }
    }
}

int WINAPI WinMain(
    HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Resolution
    const int    WIDTH       = 800; // Lower resolution for faster debug
    const int    HEIGHT      = 600;
    const size_t PIXEL_COUNT = WIDTH * HEIGHT;

    // Window Setup
    WNDCLASS wc      = { };
    wc.lpfnWndProc   = WindowProc;
    wc.hInstance     = hInstance;
    wc.lpszClassName = L"UsagiEngineApp";
    RegisterClass(&wc);

    RECT wr = { 0, 0, WIDTH, HEIGHT };
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    g_hWnd = CreateWindowEx(0,
        wc.lpszClassName,
        L"Usagi Engine - Cornell Box Path Tracer",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        wr.right - wr.left,
        wr.bottom - wr.top,
        nullptr,
        nullptr,
        hInstance,
        nullptr);
    ShowWindow(g_hWnd, nCmdShow);

    // Engine Core
    // Shio: Increased heap size to 2GB to accommodate full BDPT vertex history arrays for 800x600 pixels.
    // Each pixel needs multiple Vector3f/Color3f vertices.
    Usagi::MappedHeap primary_heap(2ULL * 1024 * 1024 * 1024); // 2GB

    // Systems
    // Note: Systems are stateless or have trivial state for now, so copying or
    // reconstructing them is cheap. However, ComponentGroup and Services must
    // be shared.

    Usagi::Services services;

    // Components
    // Note: Added ComponentPathState
    Usagi::ComponentGroup<RT::ComponentPixel,
        RT::ComponentRay,
        RT::ComponentPathState,
        RT::ComponentRayHit,
        RT::ComponentCameraPath,
        RT::ComponentLightPath>
        primary_group(primary_heap, PIXEL_COUNT);

    // Services setup
    RT::ServiceGDICanvasProvider gdi_canvas;
    gdi_canvas.width        = WIDTH;
    gdi_canvas.height       = HEIGHT;
    gdi_canvas.pixel_buffer = primary_heap.allocate_pod<uint32_t>(PIXEL_COUNT)
                                  .resolve(primary_heap.get_base());
    services.register_service(&gdi_canvas);

    RT::ServiceCamera camera;
    services.register_service(&camera);

    RT::ServiceTime time_svc;
    services.register_service(&time_svc);

    RT::ServiceRenderState render_state;
    services.register_service(&render_state);

    RT::ServiceFilm film;
    film.init(WIDTH, HEIGHT, 2.0f, 2.0f);
    services.register_service(&film);

    RT::ServiceScene scene;
    SetupCornellBox(scene);
    services.register_service(&scene);

    RT::ServiceRayQueue ray_queue;
    services.register_service(&ray_queue);

    RT::ServiceScheduler scheduler;
    services.register_service(&scheduler);

    // Init Entities
    for(int y = 0; y < HEIGHT; ++y)
    {
        for(int x = 0; x < WIDTH; ++x)
        {
            Usagi::EntityId id = primary_group.spawn();
            auto * pixels      = primary_group.get_array<RT::ComponentPixel>();
            pixels[id]         = { x, y };
            // Zero init rng state
            primary_group.get_array<RT::ComponentPathState>()[id].rng.inc = 0;
            // Zero init hit state
            primary_group.get_array<RT::ComponentRayHit>()[id].did_hit = false;
        }
    }

    SystemPresentToGDI sys_present;

    // Shio: Launch the render thread
    std::thread render_thread([&]() {
        Usagi::TaskGraphExecutionHost host(std::thread::hardware_concurrency() * 2);
        scheduler.host = &host;

        RT::SystemPathTracingCoordinator sys_coord;
        RT::SystemRenderGDICanvas        sys_render;

        host.register_system(sys_coord, primary_group, services);

        host.build_graph();

        while(g_RenderActive)
        {
            // Execute the BDPT state machine which orchestrates the sub-passes
            host.execute();
            
            // Render the splatted film explicitly after the BDPT cycle completes
            sys_render.update(primary_group, services);
        }
    });

    // Main Message Loop
    MSG msg = { };
    auto last_time = std::chrono::high_resolution_clock::now();
    bool first_mouse = true;
    POINT last_mouse_pos = {0, 0};

    while(true)
    {
        while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if(msg.message == WM_QUIT)
            {
                g_RenderActive = false;
                render_thread.join();
                return 0;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        bool moved = false;

        bool r_held = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
        bool l_held = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;

        // Unreal Editor Style Mouse Input
        if (l_held && r_held) {
            // Both L and R: Pan vertically (World Y) and horizontally (Camera Right)
            POINT current_mouse;
            GetCursorPos(&current_mouse);
            if (first_mouse) {
                last_mouse_pos = current_mouse;
                first_mouse = false;
            } else {
                float dx = (current_mouse.x - last_mouse_pos.x) * 0.05f;
                float dy = (current_mouse.y - last_mouse_pos.y) * 0.05f;
                
                if (dx != 0.0f || dy != 0.0f) {
                    camera.position += camera.right() * dx; // standard X drag
                    camera.position.y += -dy; // standard Y drag (mouse up = negative dy = positive world Y)
                    moved = true;
                    SetCursorPos(last_mouse_pos.x, last_mouse_pos.y);
                }
            }
        } else if (r_held) {
            // Right Click Only: Mouse Look (FPS style)
            POINT current_mouse;
            GetCursorPos(&current_mouse);
            if (first_mouse) {
                last_mouse_pos = current_mouse;
                first_mouse = false;
            } else {
                float dx = (current_mouse.x - last_mouse_pos.x) * 0.005f;
                float dy = (current_mouse.y - last_mouse_pos.y) * 0.005f;
                
                if (dx != 0.0f || dy != 0.0f) {
                    camera.yaw += dx; // Positive dx means looking right (+X)
                    camera.pitch += dy;
                    camera.pitch = std::clamp(camera.pitch, -1.5f, 1.5f); // Prevent gimbal lock loops
                    moved = true;
                    SetCursorPos(last_mouse_pos.x, last_mouse_pos.y);
                }
            }
        } else if (l_held) {
            // Left Click Only: UE Style (Mouse X = Yaw, Mouse Y = Move Forward/Backward)
            POINT current_mouse;
            GetCursorPos(&current_mouse);
            if (first_mouse) {
                last_mouse_pos = current_mouse;
                first_mouse = false;
            } else {
                float dx = (current_mouse.x - last_mouse_pos.x) * 0.005f;
                float dy = (current_mouse.y - last_mouse_pos.y) * 0.05f;
                
                if (dx != 0.0f || dy != 0.0f) {
                    // X movement turns the camera (Yaw)
                    camera.yaw += dx; 
                    
                    // Y movement pushes camera along the flat ground vector
                    RT::Vector3f flat_fwd = camera.forward();
                    flat_fwd.y = 0.0f;
                    if (flat_fwd.length_squared() > 0.0001f) {
                        flat_fwd = flat_fwd.normalize();
                    } else {
                        flat_fwd = {0, 0, 1};
                    }
                    
                    // Dragging mouse down (positive dy) moves backward, up moves forward
                    camera.position += flat_fwd * -dy;
                    
                    moved = true;
                    SetCursorPos(last_mouse_pos.x, last_mouse_pos.y);
                }
            }
        } else {
            first_mouse = true;
        }

        // Colemak (WARS physical) Keyboard movement
        float speed = 10.0f * dt;
        if (GetAsyncKeyState(VK_SHIFT) & 0x8000) speed *= 3.0f; // Sprint

        RT::Vector3f fwd = camera.forward();
        RT::Vector3f right = camera.right();
        RT::Vector3f delta = {0, 0, 0};

        if (GetAsyncKeyState('W') & 0x8000) delta += fwd;
        if (GetAsyncKeyState('R') & 0x8000) delta -= fwd;
        if (GetAsyncKeyState('A') & 0x8000) delta -= right;
        if (GetAsyncKeyState('S') & 0x8000) delta += right;
        
        if (delta.x != 0 || delta.y != 0 || delta.z != 0) {
            camera.position += delta.normalize() * speed;
            moved = true;
        }

        if (moved) camera.moved = true;

        // Shio: Time Control
        // Physical P (Virtual Key VK_OEM_1 or ';' in Colemak layout) toggles pause
        static bool was_pause_held = false;
        bool pause_held = (GetAsyncKeyState(VK_OEM_1) & 0x8000) != 0;
        if (pause_held && !was_pause_held) {
            time_svc.is_paused = !time_svc.is_paused;
        }
        was_pause_held = pause_held;

        // Physical R (Virtual Key 'P' in Colemak layout) unconditionally adds +1.5x time speed
        // Physical E (Virtual Key 'F' in Colemak layout) unconditionally adds -1.5x time speed
        float time_speed = 0.0f;
        if (!time_svc.is_paused) {
            time_speed += 1.0f; // Base speed when running normally
        }
        if (GetAsyncKeyState('P') & 0x8000) {
            time_speed += 1.5f; // Fast forward
        }
        if (GetAsyncKeyState('F') & 0x8000) {
            time_speed -= 1.5f; // Rewind
        }

        // Apply smooth time advancement
        // 0.3f effectively matches the old 60fps * 0.005f pacing
        if (time_speed != 0.0f) {
            time_svc.current_time += dt * 0.3f * time_speed;
        }

        // Present to Window (Main Thread)
        // We present as fast as the main thread can, or we could limit this.
        // It reads from the buffer being written by the render thread.
        // Tearing may occur, but UI stays responsive.
        sys_present.update(primary_group, services);

        // Yield slightly to prevent 100% CPU usage on the UI thread effectively
        // busy-waiting for messages if the queue is empty.
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}

int main()
{
    return WinMain(
        GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOWDEFAULT);
}
