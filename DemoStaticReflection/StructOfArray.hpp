#pragma once

#include <array>
#include <iostream>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

template <typename T, size_t N>
struct struct_of_arrays_impl
{
    struct impl;

    consteval
    {
        auto ctx = std::meta::access_context::current();

        std::vector<std::meta::info> old_members =
            nonstatic_data_members_of(^^T, ctx);
        std::vector<std::meta::info> new_members = {};
        for(std::meta::info member : old_members)
        {
            auto array_type = substitute(^^std::array,
                {
                    type_of(member),
                    std::meta::reflect_constant(N),
                });
            auto mem_descr =
                data_member_spec(array_type, { .name = identifier_of(member) });
            new_members.push_back(mem_descr);
        }

        define_aggregate(^^impl, new_members);
    }
};

template <typename T, size_t N>
using struct_of_arrays = struct_of_arrays_impl<T, N>::impl;

struct point
{
    float x;
    float y;
    float z;
};

inline void print_struct_of_arrays()
{
    using points = struct_of_arrays<point, 2>;

    points p = {
        .x = { 1.1, 2.2 },
        .y = { 3.3, 4.4 },
        .z = { 5.5, 6.6 }
    };
    static_assert(p.x.size() == 2);
    static_assert(p.y.size() == 2);
    static_assert(p.z.size() == 2);

    for(size_t i = 0; i != 2; ++i)
    {
        std::cout << "p[" << i << "] = (" << p.x[i] << ", " << p.y[i] << ", "
                  << p.z[i] << ")\n";
    }
}
