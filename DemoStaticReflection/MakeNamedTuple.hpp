#pragma once

#include <algorithm>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

consteval auto make_named_tuple(std::meta::info                         type,
    std::initializer_list<std::pair<std::meta::info, std::string_view>> members)
{
#ifdef __EDG__
    // EDG's implementation is not yet updated with the new names
    std::vector<std::meta::nsdm_description> nsdms;
    for(auto [type, name] : members)
    {
        nsdms.push_back(std::meta::nsdm_description(type, { .name = name }));
    }
    return define_class(type, nsdms);
#else
    std::vector<std::meta::info> nsdms;
    for(auto [type, name] : members)
    {
        nsdms.push_back(data_member_spec(type, { .name = name }));
    }
    return define_aggregate(type, nsdms);
#endif
}

struct R;
consteval
{
    make_named_tuple(^^R,
        {
            {    ^^int, "x" },
            { ^^double, "y" }
    });
}

namespace
{
constexpr auto ctx2 = std::meta::access_context::current();
static_assert(type_of(nonstatic_data_members_of(^^R, ctx2)[0]) == ^^int);
static_assert(type_of(nonstatic_data_members_of(^^R, ctx2)[1]) == ^^double);

auto make_named_tuple()
{
    return R { .x = 1, .y = 2.0 };
}
} // namespace
