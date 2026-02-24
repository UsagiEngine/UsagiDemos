#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#include <atomic>
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

    // Cornell Box Walls (approximated with large boxes or planes, using Boxes
    // here)
    float s = 10.0f; // Scale

    // Floor (White)
    scene.boxes.push_back({ { -s, -s - 1.0f, -s }, { s, -s, s }, 0 });
    // Ceiling (White)
    scene.boxes.push_back({ { -s, s, -s }, { s, s + 1.0f, s }, 0 });
    // Back Wall (White)
    scene.boxes.push_back({ { -s, -s, s }, { s, s, s + 1.0f }, 0 });
    // Left Wall (Red)
    scene.boxes.push_back({ { -s - 1.0f, -s, -s }, { -s, s, s }, 1 });
    // Right Wall (Green)
    scene.boxes.push_back({ { s, -s, -s }, { s + 1.0f, s, s }, 2 });

    // Light (Ceiling patch)
    float l = 2.0f;
    scene.boxes.push_back({ { -l, s - 0.1f, -l }, { l, s, l }, 3 });

    // Boxes inside
    // Short box
    scene.boxes.push_back(
        { { -3.0f, -s, 2.0f }, { 0.0f, -s + 3.0f, 5.0f }, 0 });
    // Tall box
    scene.boxes.push_back(
        { { 2.0f, -s, -3.0f }, { 5.0f, -s + 6.0f, 0.0f }, 0 });
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
    Usagi::MappedHeap primary_heap(128 * 1'024 * 1'024); // 128MB

    // Systems
    // Note: Systems are stateless or have trivial state for now, so copying or
    // reconstructing them is cheap. However, ComponentGroup and Services must
    // be shared.

    Usagi::Services services;

    // Components
    // Note: Added ComponentPathState
    Usagi::ComponentGroup<RT::ComponentPixel,
        RT::ComponentRay,
        RT::ComponentPathState>
        primary_group(primary_heap, PIXEL_COUNT);

    // Services setup
    RT::ServiceGDICanvasProvider gdi_canvas;
    gdi_canvas.width        = WIDTH;
    gdi_canvas.height       = HEIGHT;
    gdi_canvas.pixel_buffer = primary_heap.allocate_pod<uint32_t>(PIXEL_COUNT)
                                  .resolve(primary_heap.get_base());
    services.register_service(&gdi_canvas);

    RT::ServiceScene scene;
    SetupCornellBox(scene);
    services.register_service(&scene);

    // Init Entities
    for(int y = 0; y < HEIGHT; ++y)
    {
        for(int x = 0; x < WIDTH; ++x)
        {
            Usagi::EntityId id = primary_group.spawn();
            auto * pixels      = primary_group.get_array<RT::ComponentPixel>();
            pixels[id]         = { x, y };
            // Zero init rng state
            primary_group.get_array<RT::ComponentPathState>()[id].rng.state = 0;
        }
    }

    SystemPresentToGDI sys_present;

    // Shio: Launch the render thread
    std::thread render_thread([&]() {
        Usagi::Executive             executive(1);
        RT::SystemGenerateCameraRays sys_gen_rays;
        RT::SystemPathBounce         sys_bounce;
        RT::SystemRenderGDICanvas    sys_render;

        while(g_RenderActive)
        {
            // 1. Generate Primary Rays
            executive.dispatch(sys_gen_rays, primary_group, services);

            // 2. Iterative Bounces
            for(int i = 0; i < 5; ++i)
            {
                executive.dispatch(sys_bounce, primary_group, services);
            }

            // 3. Render to Buffer
            executive.dispatch(sys_render, primary_group, services);
        }
    });

    // Main Message Loop
    MSG msg = { };
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

        // Present to Window (Main Thread)
        // We present as fast as the main thread can, or we could limit this.
        // It reads from the buffer being written by the render thread.
        // Tearing may occur, but UI stays responsive.
        sys_present.update(primary_group, services);

        // Yield slightly to prevent 100% CPU usage on the UI thread effectively
        // busy-waiting for messages if the queue is empty.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}

int main()
{
    return WinMain(
        GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOWDEFAULT);
}
