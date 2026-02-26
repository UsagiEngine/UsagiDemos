/*
 * Shio:
 * =============================================================================
 * Usagi Engine - Demo Path Tracer Entry Point
 * =============================================================================
 *
 * File: main.cpp
 *
 * Logic & Design Decisions:
 * - Application lifecycle and window management are decoupled from the core
 * render loop.
 * - Win32 message pump runs exclusively on the main UI thread to remain
 * responsive.
 * - The actual BDPT engine runs on a dedicated background thread, maximizing
 * CPU utilization.
 * - Interaction:
 *   - Unreal Engine / FPS style camera controls (WASD, Mouse look, Drag to
 * Pan).
 *   - Time control: Spacebar/Pause to freeze time, Forward/Rewind mapped to
 * specific keys.
 *   - Time advancement governs the position of the celestial bodies (Sun/Moon),
 * which forces the BVH to dynamically rebalance each frame.
 * - Scene Setup:
 *   - SetupCornellBox initially populated a standard Cornell Box, but was
 * expanded to generate procedurally generated voxel terrain (mountains, trees,
 * frozen lakes, caves) to benchmark the BVH and the BDPT integrator's caustics.
 *
 * Caveats & Technical Information:
 * - Uses raw GDI SetDIBitsToDevice for zero-dependency blitting of the
 * software-rendered framebuffer to the window surface.
 * - High-resolution clock delta time (dt) dynamically adjusts the frame's
 * ray_budget to guarantee 90+ FPS interactive feedback, shifting computational
 * weight from spatial density to temporal accumulation when the camera moves.
 * - Hardware scan codes (GetAsyncKeyState / WM_KEYDOWN) bypass OS keyboard
 * layouts for precise physical key mapping.
 *
 * Development History & Experiments (Git Log Summary):
 * - Implemented non-blocking OS message pumping alongside background rendering
 * threads to prevent UI lockups during intensive trace frames (shio: move
 * window smoothly while rendering).
 * - Added comprehensive Minecraft-style terrain generation spanning thousands
 * of AABBs to stress-test acceleration structures (comprehensive minecraft
 * terrain).
 * - Fine-tuned the input mechanics (mouse capture, WASD velocity) to match
 * standard editor navigation schemes (impl mouse panning, impl interactive map
 * exploration).
 * - Added dynamic time acceleration/rewind to observe physically-based
 * atmospheric scattering dynamically as the sun sets (add time accel/rewind).
 * =============================================================================
 */
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
std::atomic<bool> g_KeyStates[256];

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
    if(uMsg == WM_KEYDOWN || uMsg == WM_KEYUP || uMsg == WM_SYSKEYDOWN ||
        uMsg == WM_SYSKEYUP)
    {
        bool isDown   = (uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN);
        UINT scanCode = (lParam >> 16) & 0xFF;
        g_KeyStates[scanCode].store(isDown, std::memory_order_relaxed);
    }
    if(uMsg == WM_KILLFOCUS)
    {
        for(int i = 0; i < 256; ++i)
            g_KeyStates[i].store(false, std::memory_order_relaxed);
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
    scene.materials.push_back(
        { MaterialType::Metal, { 0.8f, 0.85f, 0.88f }, { 0, 0, 0 }, 0.02f });
    // 5: Translucent (Pale Purple Jelly SSS)
    scene.materials.push_back({ MaterialType::Translucent,
        { 0.95f, 0.85f, 0.98f }, // Very pale, bright lilac/purple
        { 0, 0, 0 },
        0.1f,
        1.3f,
        false,
        0.3f }); // Lower density to make it significantly more translucent

    // 6: The Sun (Massive light sphere far away)
    scene.materials.push_back({ MaterialType::Light,
        { 1.0f, 1.0f, 1.0f },
        { 15.0f, 15.0f, 15.0f },
        0.0f,
        1.0f,
        false });

    // 7: The Moon (Textured secondary light sphere directly opposed to the Sun)
    scene.materials.push_back({ MaterialType::Light,
        { 1.0f, 1.0f, 1.0f },
        { 2.0f, 2.2f, 2.5f },
        0.0f,
        1.0f,
        true });

    // --- Minecraft-style World Materials ---
    // 8: Grass (Lambertian Vivid Green)
    scene.materials.push_back(
        { MaterialType::Lambert, { 0.3f, 0.7f, 0.2f }, { 0, 0, 0 }, 0.0f });
    // 9: Dirt (Lambertian Brown)
    scene.materials.push_back(
        { MaterialType::Lambert, { 0.4f, 0.25f, 0.15f }, { 0, 0, 0 }, 0.0f });
    // 10: Wood Log (Lambertian Dark Brown)
    scene.materials.push_back(
        { MaterialType::Lambert, { 0.25f, 0.15f, 0.08f }, { 0, 0, 0 }, 0.0f });
    // 11: Leaves (Subsurface Translucent Green)
    scene.materials.push_back({ MaterialType::Translucent,
        { 0.2f, 0.8f, 0.2f },
        { 0, 0, 0 },
        0.3f,
        1.1f,
        false,
        0.5f });
    // 12: Snow (Highly reflective rough Lambertian with slight blue tint)
    scene.materials.push_back(
        { MaterialType::Lambert, { 0.9f, 0.95f, 1.0f }, { 0, 0, 0 }, 0.0f });
    // 13: Frozen Lake (Highly glossy Ice)
    scene.materials.push_back({ MaterialType::Translucent,
        { 0.7f, 0.85f, 0.95f },
        { 0, 0, 0 },
        0.005f,
        1.31f,
        false,
        0.1f });

    // Cornell Box Walls (approximated with large boxes or planes, using Boxes
    // here)
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
    float bs      = 2.0f;      // Block Size
    float floor_y = -s - 1.0f; // Align with bottom of Cornell Box

    auto add_block =
        [&](float x, float y, float z, int mat_idx, float scale = 1.0f) {
            float gap = bs * (1.0f - scale) * 0.5f;
            scene.boxes.push_back(
                { { x * bs + gap, floor_y + y * bs + gap, z * bs + gap },
                    { (x + 1) * bs - gap,
                        floor_y + (y + 1) * bs - gap,
                        (z + 1) * bs - gap },
                    mat_idx });
        };

    auto get_height = [](int x, int z) {
        // Shio: Procedural pseudo-random heightmap generation using overlapping
        // trigonometric functions. Creating non-uniform frequencies breaks up
        // grid-like repetition in the Minecraft terrain.
        float fx = x * 0.15f;
        float fz = z * 0.15f;
        float h  = std::sin(fx) * std::cos(fz) * 3.0f +
            std::sin(fx * 2.5f + fz * 1.5f) * 1.0f;
        return (int)std::floor(h);
    };

    // Massive Underground Base to catch escaping rays efficiently without 1000s
    // of dirt blocks
    scene.boxes.push_back({ { -100.0f, floor_y - 10.0f, -100.0f },
        { 100.0f, floor_y - 3.0f, 100.0f },
        9 });

    for(int x = -15; x <= 15; ++x)
    {
        for(int z = -15; z <= 25; ++z)
        {
            // Leave space for the Cornell Box
            if(x >= -6 && x <= 6 && z >= -6 && z <= 6) continue;

            int h = get_height(x, z);

            if(h < 0)
            {
                // Frozen Lake
                bool is_hole =
                    (std::sin(x * 3.14f) * std::cos(z * 2.71f)) > 0.6f;
                if(!is_hole)
                {
                    add_block(x, -1, z, 13); // Ice Surface
                }
                else
                {
                    add_block(x, -2, z, 13); // Sunken Ice (Fracture/Hole)
                }
            }
            else
            {
                // Land
                if(h > 1)
                {
                    add_block(x, h, z, 12); // Snow capped peaks
                }
                else
                {
                    add_block(x, h, z, 8); // Grassy lowlands
                }

                // Add exposed cliff faces (only down to neighbor minimum to
                // save polygons)
                int min_neighbor = std::min({ get_height(x - 1, z),
                    get_height(x + 1, z),
                    get_height(x, z - 1),
                    get_height(x, z + 1),
                    h });
                for(int y = std::max(-2, min_neighbor); y < h; ++y)
                {
                    add_block(x, y, z, 9); // Dirt cliff
                }

                // Random Trees
                if(h <= 1 && (std::abs(x * 73 + z * 37) % 45 == 0))
                {
                    int tree_h = 3 + (std::abs(x + z) % 3);
                    // Trunk
                    for(int ty = 1; ty <= tree_h; ++ty)
                    {
                        add_block(x, h + ty, z, 10, 0.8f);
                    }
                    // Leaves with physical gaps allowing light transmission!
                    for(int lx = -2; lx <= 2; ++lx)
                    {
                        for(int ly = 0; ly <= 2; ++ly)
                        {
                            for(int lz = -2; lz <= 2; ++lz)
                            {
                                if(std::abs(lx) == 2 && std::abs(lz) == 2 &&
                                    ly == 2)
                                    continue; // Round canopy corners
                                if((std::abs(lx * 11 + ly * 13 + lz * 17)) %
                                        4 ==
                                    0)
                                    continue; // Procedural gaps for dappled
                                              // light
                                add_block(x + lx,
                                    h + tree_h - 1 + ly,
                                    z + lz,
                                    11,
                                    0.85f);
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
    // Shio: Increased heap size to 2GB to accommodate full BDPT vertex history
    // arrays for 800x600 pixels. Each pixel needs multiple Vector3f/Color3f
    // vertices.
    Usagi::MappedHeap primary_heap(2ULL * 1'024 * 1'024 * 1'024); // 2GB

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
    scene.optimize_bvh(); // Shio: Initial BVH build.
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
    // This true game loop runs completely decoupled from the OS message pump,
    // ensuring consistent input polling, time advancement, and rendering rates.
    std::thread render_thread([&]() {
        Usagi::TaskGraphExecutionHost host(
            std::thread::hardware_concurrency() * 2);
        scheduler.host = &host;

        RT::SystemPathTracingCoordinator sys_coord;
        RT::SystemRenderGDICanvas        sys_render;

        host.register_system(sys_coord, primary_group, services);
        host.build_graph();

        auto  last_time      = std::chrono::high_resolution_clock::now();
        bool  first_mouse    = true;
        POINT last_mouse_pos = { 0, 0 };

        while(g_RenderActive)
        {
            // --- TIME AND INPUT UPDATE ---
            auto  current_time = std::chrono::high_resolution_clock::now();
            float dt =
                std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;

            // --- Dynamic Ray Budget (Target ~90 FPS for maximum fluidity) ---
            float target_dt = 1.0f / 90.0f;
            // Shio: Feedback control loop. If the previous frame took too long,
            // aggressively drop the ray budget by 20%. If it was fast,
            // cautiously increase by 15%. This guarantees interactivity even
            // when moving into complex, heavy BVH views.
            if(dt > target_dt * 1.05f)
            {
                render_state.ray_budget =
                    std::max(20'000, int(render_state.ray_budget * 0.8f));
            }
            else if(dt < target_dt * 0.95f)
            {
                render_state.ray_budget = std::min(
                    (int)PIXEL_COUNT, int(render_state.ray_budget * 1.15f));
            }

            bool moved      = false;
            HWND foreground = GetForegroundWindow();
            bool is_focused = (foreground == g_hWnd);

            bool r_held =
                is_focused && (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
            bool l_held =
                is_focused && (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;

            static bool is_dragging_viewport = false;

            if(is_focused && (l_held || r_held))
            {
                if(!is_dragging_viewport)
                {
                    POINT pt;
                    GetCursorPos(&pt);
                    ScreenToClient(g_hWnd, &pt);
                    RECT clientRect;
                    GetClientRect(g_hWnd, &clientRect);
                    if(PtInRect(&clientRect, pt))
                    {
                        is_dragging_viewport = true;
                        first_mouse = true; // reset delta to prevent snapping
                    }
                }
            }
            else
            {
                is_dragging_viewport = false;
            }

            if(is_dragging_viewport)
            {
                // Unreal Editor Style Mouse Input
                if(l_held && r_held)
                {
                    // Both L and R: Pan vertically (World Y) and horizontally
                    // (Camera Right)
                    POINT current_mouse;
                    GetCursorPos(&current_mouse);
                    if(first_mouse)
                    {
                        last_mouse_pos = current_mouse;
                        first_mouse    = false;
                    }
                    else
                    {
                        float dx = (current_mouse.x - last_mouse_pos.x) * 0.05f;
                        float dy = (current_mouse.y - last_mouse_pos.y) * 0.05f;

                        if(dx != 0.0f || dy != 0.0f)
                        {
                            camera.position +=
                                camera.right() * dx;  // standard X drag
                            camera.position.y += -dy; // standard Y drag
                            moved = true;
                            SetCursorPos(last_mouse_pos.x, last_mouse_pos.y);
                        }
                    }
                }
                else if(r_held)
                {
                    // Right Click Only: Mouse Look (FPS style)
                    POINT current_mouse;
                    GetCursorPos(&current_mouse);
                    if(first_mouse)
                    {
                        last_mouse_pos = current_mouse;
                        first_mouse    = false;
                    }
                    else
                    {
                        float dx =
                            (current_mouse.x - last_mouse_pos.x) * 0.005f;
                        float dy =
                            (current_mouse.y - last_mouse_pos.y) * 0.005f;

                        if(dx != 0.0f || dy != 0.0f)
                        {
                            camera.yaw +=
                                dx; // Positive dx means looking right (+X)
                            camera.pitch += dy;
                            camera.pitch = std::clamp(camera.pitch,
                                -1.5f,
                                1.5f); // Prevent gimbal lock loops
                            moved        = true;
                            SetCursorPos(last_mouse_pos.x, last_mouse_pos.y);
                        }
                    }
                }
                else if(l_held)
                {
                    // Left Click Only: UE Style (Mouse X = Yaw, Mouse Y = Move
                    // Forward/Backward)
                    POINT current_mouse;
                    GetCursorPos(&current_mouse);
                    if(first_mouse)
                    {
                        last_mouse_pos = current_mouse;
                        first_mouse    = false;
                    }
                    else
                    {
                        float dx =
                            (current_mouse.x - last_mouse_pos.x) * 0.005f;
                        float dy = (current_mouse.y - last_mouse_pos.y) * 0.05f;

                        if(dx != 0.0f || dy != 0.0f)
                        {
                            camera.yaw += dx;

                            RT::Vector3f flat_fwd = camera.forward();
                            flat_fwd.y            = 0.0f;
                            if(flat_fwd.length_squared() > 0.0001f)
                            {
                                flat_fwd = flat_fwd.normalize();
                            }
                            else
                            {
                                flat_fwd = { 0, 0, 1 };
                            }

                            camera.position += flat_fwd * -dy;
                            moved = true;
                            SetCursorPos(last_mouse_pos.x, last_mouse_pos.y);
                        }
                    }
                }
                else
                {
                    first_mouse = true;
                }
            }
            else
            {
                first_mouse =
                    true; // Prevents sudden jerks when regaining focus
            }

            if(is_focused)
            {
                // Shio: Raw hardware scan codes completely bypass Windows
                // layout translation! 0x11 = Physical W, 0x1E = Physical A,
                // 0x1F = Physical S, 0x20 = Physical D This ensures the
                // physical positions of WASD remain the same on
                // AZERTY/Dvorak/etc.
                bool key_W = g_KeyStates[0x11].load(std::memory_order_relaxed);
                bool key_A = g_KeyStates[0x1E].load(std::memory_order_relaxed);
                bool key_S = g_KeyStates[0x1F].load(std::memory_order_relaxed);
                bool key_D = g_KeyStates[0x20].load(std::memory_order_relaxed);
                bool key_Shift =
                    g_KeyStates[0x2A].load(std::memory_order_relaxed);

                float speed = 10.0f * dt;
                if(key_Shift) speed *= 3.0f;

                RT::Vector3f fwd   = camera.forward();
                RT::Vector3f right = camera.right();
                RT::Vector3f delta = { 0, 0, 0 };

                if(key_W) delta += fwd;
                if(key_S) delta -= fwd;
                if(key_A) delta -= right;
                if(key_D) delta += right;

                if(delta.x != 0 || delta.y != 0 || delta.z != 0)
                {
                    camera.position += delta.normalize() * speed;
                    moved = true;
                }
            }

            if(moved) camera.moved = true;

            // Shio: Time Control
            static bool was_pause_held = false;
            // Scan Code 0x19 is the physical 'P' key
            bool        pause_held =
                is_focused && g_KeyStates[0x19].load(std::memory_order_relaxed);
            if(pause_held && !was_pause_held)
            {
                time_svc.is_paused = !time_svc.is_paused;
            }
            was_pause_held = pause_held;

            float time_speed = 0.0f;
            if(!time_svc.is_paused)
            {
                time_speed += 1.0f; // Base speed when running normally
            }
            // Scan Code 0x13 is physical 'R' key
            if(is_focused && g_KeyStates[0x13].load(std::memory_order_relaxed))
            {
                time_speed += 1.5f; // Fast forward
            }
            // Scan Code 0x12 is physical 'E' key
            if(is_focused && g_KeyStates[0x12].load(std::memory_order_relaxed))
            {
                time_speed -= 1.5f; // Rewind
            }
            if(time_speed != 0.0f)
            {
                time_svc.current_time += dt * 0.3f * time_speed;
            }

            // --- RENDER PIPELINE ---
            // Execute the BDPT state machine which orchestrates the sub-passes
            host.execute();

            // Render the splatted film explicitly after the BDPT cycle
            // completes
            sys_render.update(primary_group, services);

            // Present to Window asynchronously inside the worker thread to free
            // up the OS message pump
            sys_present.update(primary_group, services);
        }
    });

    // Main Message Loop (UI Thread only!)
    MSG msg = { };
    while(true)
    {
        // Shio: GetMessage blocks, allowing the UI thread to sleep peacefully
        // at 0% CPU while the background render_thread spins at maximum
        // framerate!
        if(GetMessage(&msg, nullptr, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            g_RenderActive = false;
            render_thread.join();
            return 0;
        }
    }
    return 0;
}

int main()
{
    return WinMain(
        GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOWDEFAULT);
}
