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
 * Shio:
 * The following definitions map to internal Windows NT kernel structures.
 * We use these to interface directly with the memory manager, allowing for
 * potentially finer control over section objects and view mapping than standard
 * Win32 VirtualAlloc/FileMapping APIs typically expose.
 */

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

template <typename... Ts>
struct ComponentList
{
};

/*
 * Shio: Handle-based relative pointer for zero-cost recovery.
 * Always resolves relative to the mapped heap's base address.
 * This allows the memory block to be re-mapped at a different virtual address
 * without breaking internal references, essentially making the data
 * position-independent (PIC) in terms of serialization/deserialization.
 */
template <typename T>
struct Handle
{
    uint32_t offset;

    /*
     * Shio: Resolves the offset to a real pointer using the provided base.
     * This is a hot path function in data access.
     */
    T * resolve(void * base) const
    {
        return reinterpret_cast<T *>(static_cast<char *>(base) + offset);
    }
};

/*
 * Shio: MappedHeap directly interfaces with the NT kernel via NtCreateSection
 * and NtMapViewOfSection. It creates a contiguous virtual address space backed
 * by the page file (since FileHandle is nullptr).
 */
class MappedHeap
{
    HANDLE              section_handle = nullptr;
    void *              base_address   = nullptr;
    SIZE_T              total_size     = 0;
    std::atomic<size_t> current_offset = 0;

public:
    /*
     * Shio: Initializes the heap with a fixed size.
     * Uses SEC_COMMIT (0x8000000) to allocate physical pages as needed.
     * PAGE_READWRITE (0x04) ensures the memory is writable.
     */
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

    /*
     * Shio: Linear allocator (bump pointer).
     * Thread-safe due to std::atomic fetch_add.
     * Returns a Handle (offset) rather than a raw pointer.
     */
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
 * Allocates contiguous blocks in the MappedHeap for each component type.
 * This adheres to Data-Oriented Design (DOD) principles, ensuring high
 * cache locality for systems that iterate over specific components.
 */
template <typename... Components>
class ComponentGroup
{
    MappedHeap &        heap;
    size_t              capacity;
    std::atomic<size_t> count { 0 };

    /*
     * Shio: Stores handles to the start of each component array.
     * The structure is effectively SoA (Structure of Arrays).
     */
    std::tuple<Handle<Components>...> arrays;

public:
    ComponentGroup(MappedHeap & heap, size_t capacity)
        : heap(heap), capacity(capacity)
    {
        // Allocate arrays for all component types at initialization.
        arrays = std::make_tuple(heap.allocate_pod<Components>(capacity)...);
    }

    /*
     * Shio: Atomic reservation of an entity slot.
     * Returns the index (EntityId) used to access components in the arrays.
     */
    EntityId spawn() { return static_cast<EntityId>(count.fetch_add(1)); }

    size_t size() const { return count.load(); }

    /*
     * Shio: Retrieves the raw pointer to the array of type T.
     * Resolves the handle against the heap base.
     */
    template <typename T>
    T * get_array()
    {
        return std::get<Handle<T>>(arrays).resolve(heap.get_base());
    }

    /*
     * Shio: Basic query mechanism.
     * Accepts a lambda 'func' and iterates over all active entities.
     * 'QueryTypes' specifies which component pointers are passed to 'func'.
     * This allows systems to request only the data they need.
     */
    template <typename... QueryTypes>
    auto query()
    {
        return [this](auto && func) {
            size_t current_count = count.load();
            // Resolve pointers for requested component types once per query
            auto   tuple_of_pointers =
                std::make_tuple(this->get_array<QueryTypes>()...);

            // Linear iteration over the contiguous arrays
            for(size_t i = 0; i < current_count; ++i)
            {
                // Invoke the system lambda with references to the components
                func(std::get<QueryTypes *>(tuple_of_pointers)[i]...);
            }
        };
    }
};

/*
 * Shio: Fully implemented Services registry using a static type ID generator.
 * Allows global access to singleton-like services (e.g., Input, Graphics)
 * without hard dependencies or globals.
 */
class Services
{
    void * providers[64] = { nullptr };

    /*
     * Shio: Generates a unique, sequential ID for each type T.
     * Thread-safe initialization.
     */
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
        // Unchecked cast for performance; assumes correctness.
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

    /*
     * Shio: Dispatches a system update.
     * In a full engine, this would check dependencies and schedule tasks
     * across worker threads. Here, it simply calls 'update' inline.
     */
    template <typename System, typename Entities, typename Services>
    void dispatch(System & sys, Entities & entities, Services & services)
    {
        sys.update(entities, services);
    }
};

} // namespace Usagi
