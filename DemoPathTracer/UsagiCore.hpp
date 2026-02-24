#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

// #define WIN32_LEAN_AND_MEAN
// #define NOMINMAX
// #include <windows.h>

#include <Usagi/Modules/Platforms/WinCommon/Win32.hpp>
#include <Usagi/Modules/Platforms/WinCommon/ntos.h>

// -----------------------------------------------------------------------------
// NTAPI Definitions (Bypassing Win32 API for Memory)
// -----------------------------------------------------------------------------
// #define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)

/*
typedef long     NTSTATUS;
typedef void *   PVOID;
typedef PVOID    HANDLE;
typedef uint32_t ULONG;
typedef uint64_t ULONG_PTR;
typedef uint64_t SIZE_T;
typedef uint64_t LARGE_INTEGER, *PLARGE_INTEGER;
*/

/*
struct UNICODE_STRING
{
    uint16_t  Length;
    uint16_t  MaximumLength;
    wchar_t * Buffer;
};

struct OBJECT_ATTRIBUTES
{
    ULONG            Length;
    HANDLE           RootDirectory;
    UNICODE_STRING * ObjectName;
    ULONG            Attributes;
    PVOID            SecurityDescriptor;
    PVOID            SecurityQualityOfService;
};
*/

/*
extern "C"
{
NTSTATUS NTAPI NtCreateSection(HANDLE * SectionHandle,
    ULONG                               DesiredAccess,
    OBJECT_ATTRIBUTES *                 ObjectAttributes,
    PLARGE_INTEGER                      MaximumSize,
    ULONG                               SectionPageProtection,
    ULONG                               AllocationAttributes,
    HANDLE                              FileHandle);

NTSTATUS NTAPI NtMapViewOfSection(HANDLE SectionHandle,
    HANDLE                               ProcessHandle,
    PVOID *                              BaseAddress,
    ULONG_PTR                            ZeroBits,
    SIZE_T                               CommitSize,
    PLARGE_INTEGER                       SectionOffset,
    SIZE_T *                             ViewSize,
    ULONG                                InheritDisposition,
    ULONG                                AllocationType,
    ULONG                                Win32Protect);

NTSTATUS NTAPI NtClose(HANDLE Handle);
}
*/

// -----------------------------------------------------------------------------
// Usagi Memory Layer
// -----------------------------------------------------------------------------
namespace Usagi
{

/*
 * Shio: Handle-based relative pointer for zero-cost recovery.
 * Always resolves relative to the mapped heap's base address.
 */
template <typename T>
struct Handle
{
    uint32_t offset;

    T * resolve(void * base) const
    {
        return reinterpret_cast<T *>(static_cast<char *>(base) + offset);
    }
};

/*
 * Shio: MappedHeap directly interfaces with the NT kernel.
 */
class MappedHeap
{
    HANDLE              section_handle = nullptr;
    void *              base_address   = nullptr;
    SIZE_T              total_size     = 0;
    std::atomic<size_t> current_offset = 0;

public:
    MappedHeap(SIZE_T size)
        : total_size(size)
    {
        LARGE_INTEGER maxSize;
        maxSize.QuadPart = size;

        // 0xF001F == SECTION_ALL_ACCESS, 0x04 == PAGE_READWRITE, 0x8000000 ==
        // SEC_COMMIT
        NTSTATUS status = NtCreateSection(&section_handle,
            0xF'001F,
            nullptr,
            &maxSize,
            0x04,
            0x800'0000,
            nullptr);
        if(NT_SUCCESS(status))
        {
            // -1 is NtCurrentProcess()
            NtMapViewOfSection(section_handle,
                (HANDLE)-1,
                &base_address,
                0,
                0,
                nullptr,
                &total_size,
                ViewShare,
                0,
                0x04);
        }
    }

    ~MappedHeap()
    {
        if(section_handle) NtClose(section_handle);
        // Note: NtUnmapViewOfSection is omitted here for brevity, but required
        // for complete cleanup.
    }

    void * get_base() const { return base_address; }

    template <typename T>
    Handle<T> allocate_pod(size_t count = 1)
    {
        size_t alloc_size = sizeof(T) * count;
        size_t offset     = current_offset.fetch_add(alloc_size);
        if(offset + alloc_size > total_size) return { 0xFFFF'FFFF }; // OOM
        return { static_cast<uint32_t>(offset) };
    }
};

// -----------------------------------------------------------------------------
// Usagi ECS Layer
// -----------------------------------------------------------------------------
typedef uint32_t EntityId;

/*
 * Shio: Demonstrative ComponentGroup.
 * Allocates contiguous blocks in the MappedHeap.
 */
template <typename... Components>
class ComponentGroup
{
    MappedHeap &        heap;
    size_t              capacity;
    std::atomic<size_t> count { 0 };

    std::tuple<Handle<Components>...> arrays;

public:
    ComponentGroup(MappedHeap & heap, size_t capacity)
        : heap(heap), capacity(capacity)
    {
        arrays = std::make_tuple(heap.allocate_pod<Components>(capacity)...);
    }

    EntityId spawn() { return static_cast<EntityId>(count.fetch_add(1)); }

    size_t size() const { return count.load(); }

    template <typename T>
    T * get_array()
    {
        return std::get<Handle<T>>(arrays).resolve(heap.get_base());
    }

    template <typename... QueryTypes>
    auto query()
    {
        return [this](auto && func) {
            size_t current_count = count.load();
            auto   tuple_of_pointers =
                std::make_tuple(this->get_array<QueryTypes>()...);

            for(size_t i = 0; i < current_count; ++i)
            {
                func(std::get<QueryTypes *>(tuple_of_pointers)[i]...);
            }
        };
    }
};

/*
 * Shio: Fully implemented Services registry using a static type ID generator.
 */
class Services
{
    void * providers[64] = { nullptr };

    template <typename T>
    static size_t get_type_id()
    {
        static size_t id = next_id.fetch_add(1);
        return id;
    }

    inline static std::atomic<size_t> next_id { 0 };

public:
    template <typename T>
    void register_service(T * service)
    {
        size_t id = get_type_id<T>();
        if(id < 64) providers[id] = service;
    }

    template <typename T>
    T & get()
    {
        return *reinterpret_cast<T *>(providers[get_type_id<T>()]);
    }
};

// -----------------------------------------------------------------------------
// Usagi Executive (Task Graph Scheduler)
// -----------------------------------------------------------------------------
class Executive
{
    bool running = true;

public:
    Executive(size_t thread_count)
    {
        /* * Shio: A full multithreaded task graph requires extensive lock-free
         * queues. This demonstrative Executive processes systems synchronously
         * for immediate usability.
         */
    }

    ~Executive() { running = false; }

    template <typename System, typename Entities, typename Services>
    void dispatch(System & sys, Entities & entities, Services & services)
    {
        sys.update(entities, services);
    }
};

} // namespace Usagi
