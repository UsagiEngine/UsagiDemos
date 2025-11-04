#pragma once

#include <array>
#include <cassert>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

template <typename... Ts>
struct Tuple
{
    struct storage;
    consteval
    {
        define_aggregate(^^storage,
            {
                data_member_spec(^^Ts)... });
    }
    storage data;

    Tuple() : data {} {}

    Tuple(Ts const &...vs) : data { vs... } {}
};

template <typename... Ts>
struct std::tuple_size<Tuple<Ts...>>
    : public integral_constant<size_t, sizeof...(Ts)>
{
};

template <std::size_t I, typename... Ts>
struct std::tuple_element<I, Tuple<Ts...>>
{
    static constexpr std::array types = { ^^Ts... };
    using type                        = [:types[I]:];
};

consteval std::meta::info get_nth_nsdm(std::meta::info r, std::size_t n)
{
    return nonstatic_data_members_of(r,
        std::meta::access_context::current())[n];
}

template <std::size_t I, typename... Ts>
constexpr auto get(Tuple<Ts...> &t) noexcept
    -> std::tuple_element_t<I, Tuple<Ts...>> &
{
    return t.data.[:get_nth_nsdm(^^decltype(t.data), I):];
}

template <std::size_t I, typename... Ts>
constexpr auto get(Tuple<Ts...> const &t) noexcept
    -> std::tuple_element_t<I, Tuple<Ts...>> const &
{
    return t.data.[:get_nth_nsdm(^^decltype(t.data), I):];
}

template <std::size_t I, typename... Ts>
constexpr auto get(Tuple<Ts...> &&t) noexcept
    -> std::tuple_element_t<I, Tuple<Ts...>> &&
{
    return std::move(t).data.[:get_nth_nsdm(^^decltype(t.data), I):];
}

inline void test_tuple()
{
    auto [x, y, z] = Tuple { 1, 'c', 3.14 };
    assert(x == 1);
    assert(y == 'c');
    assert(z == 3.14);
}
