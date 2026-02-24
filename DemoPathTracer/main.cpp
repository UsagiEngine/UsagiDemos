#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#include "Raytracer.hpp"

HWND g_hWnd = nullptr;

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

        HDC hdc = GetDC(g_hWnd);

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

int WINAPI WinMain(
    HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    const int    WIDTH       = 1'280;
    const int    HEIGHT      = 720;
    const size_t PIXEL_COUNT = WIDTH * HEIGHT;

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

    // 64 MB Heap mapped via NTAPI
    Usagi::MappedHeap primary_heap(64 * 1'024 * 1'024);

    // Single-threaded executive for demonstration
    Usagi::Executive executive(1);
    Usagi::Services  services;

    Usagi::ComponentGroup<RT::ComponentPixel, RT::ComponentRay> primary_group(
        primary_heap, PIXEL_COUNT);

    RT::ServiceGDICanvasProvider gdi_canvas;
    gdi_canvas.width  = WIDTH;
    gdi_canvas.height = HEIGHT;

    auto pixel_buffer_handle = primary_heap.allocate_pod<uint32_t>(PIXEL_COUNT);
    gdi_canvas.pixel_buffer =
        pixel_buffer_handle.resolve(primary_heap.get_base());

    // Shio: Now using the fully implemented registry
    services.register_service(&gdi_canvas);

    for(int y = 0; y < HEIGHT; ++y)
    {
        for(int x = 0; x < WIDTH; ++x)
        {
            Usagi::EntityId id = primary_group.spawn();
            primary_group.get_array<RT::ComponentPixel>()[id] = { x, y };
        }
    }

    RT::SystemGenerateCameraRays       sys_gen_rays;
    RT::SystemEvaluatePhysicalMaterial sys_eval_mat;
    RT::SystemRenderGDICanvas          sys_render_canvas;
    SystemPresentToGDI                 sys_present_gdi;

    MSG msg = { };
    while(true)
    {
        while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if(msg.message == WM_QUIT) return 0;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

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
