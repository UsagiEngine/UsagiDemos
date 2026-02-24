#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#include "Raytracer.hpp"

// Global window handle for GDI calls
HWND g_hWnd = nullptr;

/*
 * Shio: Standard Win32 Window Procedure.
 * Handles window destruction to exit the application loop.
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

/*
 * Shio: System responsible for blitting the rendered buffer to the window.
 * Uses SetDIBitsToDevice for direct memory copy to GDI surface.
 */
struct SystemPresentToGDI
{
    void update(auto && entities, auto && services)
    {
        auto & canvas = services.template get<RT::ServiceGDICanvasProvider>();

        HDC hdc = GetDC(g_hWnd);

        // Configure the bitmap header to describe our raw pixel buffer
        BITMAPINFO bmi              = { };
        bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth       = canvas.width;
        bmi.bmiHeader.biHeight      = -canvas.height; // Negative for top-down
        bmi.bmiHeader.biPlanes      = 1;
        bmi.bmiHeader.biBitCount    = 32;             // 0xRRGGBB format
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
 * Shio: Application Entry Point.
 * Sets up the window, initializes the Usagi engine, and enters the main loop.
 */
int WINAPI WinMain(
    HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Resolution Configuration
    const int    WIDTH       = 1'280;
    const int    HEIGHT      = 720;
    const size_t PIXEL_COUNT = WIDTH * HEIGHT;

    // Window Registration and Creation
    WNDCLASS wc      = { };
    wc.lpfnWndProc   = WindowProc;
    wc.hInstance     = hInstance;
    wc.lpszClassName = L"UsagiEngineApp";
    RegisterClass(&wc);

    RECT wr = { 0, 0, WIDTH, HEIGHT };
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
    g_hWnd = CreateWindowEx(0,
        wc.lpszClassName,
        L"Usagi Engine - Minimal Raytracer",
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

    // -------------------------------------------------------------------------
    // Engine Initialization
    // -------------------------------------------------------------------------

    // 64 MB Heap mapped via NTAPI for component storage
    Usagi::MappedHeap primary_heap(64 * 1'024 * 1'024);

    // Single-threaded executive for demonstration
    Usagi::Executive executive(1);
    Usagi::Services  services;

    /*
     * Shio: Create the main component group.
     * This allocates SoA storage for Pixel and Ray components for all pixels.
     * We use a fixed capacity equal to the pixel count.
     */
    Usagi::ComponentGroup<RT::ComponentPixel, RT::ComponentRay> primary_group(
        primary_heap, PIXEL_COUNT);

    // Setup the Canvas Service
    RT::ServiceGDICanvasProvider gdi_canvas;
    gdi_canvas.width  = WIDTH;
    gdi_canvas.height = HEIGHT;

    // Allocate the pixel buffer from the managed heap
    auto pixel_buffer_handle = primary_heap.allocate_pod<uint32_t>(PIXEL_COUNT);
    gdi_canvas.pixel_buffer =
        pixel_buffer_handle.resolve(primary_heap.get_base());

    // Shio: Register the canvas service so systems can access it
    services.register_service(&gdi_canvas);

    // -------------------------------------------------------------------------
    // Entity Spawning
    // -------------------------------------------------------------------------

    /*
     * Shio: Initialize one entity per pixel.
     * We populate the ComponentPixel data here, which remains constant.
     */
    for(int y = 0; y < HEIGHT; ++y)
    {
        for(int x = 0; x < WIDTH; ++x)
        {
            Usagi::EntityId id = primary_group.spawn();
            primary_group.get_array<RT::ComponentPixel>()[id] = { x, y };
        }
    }

    // -------------------------------------------------------------------------
    // System Instantiation
    // -------------------------------------------------------------------------
    RT::SystemGenerateCameraRays       sys_gen_rays;
    RT::SystemEvaluatePhysicalMaterial sys_eval_mat;
    RT::SystemRenderGDICanvas          sys_render_canvas;
    SystemPresentToGDI                 sys_present_gdi;

    // -------------------------------------------------------------------------
    // Main Loop
    // -------------------------------------------------------------------------
    MSG msg = { };
    while(true)
    {
        // Process Window Messages (Non-blocking)
        while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if(msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        /*
         * Shio: Dispatch the system pipeline.
         * 1. Generate Rays: Reset rays and set direction based on camera.
         * 2. Evaluate Material: Intersect scene and calculate color.
         * 3. Render Canvas: Convert colors to integer buffer.
         * 4. Present: Blit buffer to window.
         */
        executive.dispatch(sys_gen_rays, primary_group, services);
        executive.dispatch(sys_eval_mat, primary_group, services);
        executive.dispatch(sys_render_canvas, primary_group, services);
        executive.dispatch(sys_present_gdi, primary_group, services);
    }

    return 0;
}

int main()
{
    return WinMain(
        GetModuleHandle(NULL), NULL, GetCommandLineA(), SW_SHOWDEFAULT);
}
