#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <string>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

namespace __impl
{
template <auto... vals>
struct replicator_type
{
    template <typename F>
    constexpr auto operator>>(F body) const -> decltype(auto)
    {
        return body.template operator()<vals...>();
    }
};

template <auto... vals>
replicator_type<vals...> replicator = {};
} // namespace __impl

template <typename R>
consteval auto expand_all(R range)
{
    std::vector<std::meta::info> args;
    for(auto r : range)
    {
        args.push_back(reflect_constant(r));
    }
    return substitute(^^__impl::replicator, args);
}

template <typename T>
constexpr auto struct_to_tuple(T const &t)
{
    constexpr auto ctx = std::meta::access_context::current();
    return [:expand_all(nonstatic_data_members_of(^^T, ctx)):] >>
        [&]<auto... members> { return std::make_tuple(t.[:members:]...); };
}

struct S
{
    int         x;
    char        y;
    std::string z;
};

inline auto struct_to_tuple()
{
    auto parts = struct_to_tuple(S { .x = 1, .y = 'x', .z = "hello" });
    static_assert(
        std::same_as<decltype(parts), std::tuple<int, char, std::string>>);
    assert(std::get<0>(parts) == 1);
    assert(std::get<1>(parts) == 'x');
    assert(std::get<2>(parts) == "hello");
    return parts;
}
