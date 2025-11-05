#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <new>
#include <string>
#include <variant>

#include <boost/mp11.hpp>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

using namespace boost::mp11;

template <typename... Ts>
class Variant
{
    union Storage;

    struct Empty
    {
    };

    consteval
    {
        define_aggregate(^^Storage,
            {
                data_member_spec(^^Empty,
                    {
                        .name = "empty" }),
                data_member_spec(^^Ts)... });
    }

    static consteval std::meta::info get_nth_nsdm(std::size_t n)
    {
        auto ctx = std::meta::access_context::current();
        return nonstatic_data_members_of(^^Storage, ctx)[n + 1];
    }

    Storage storage_;
    int     index_ = -1;

    // cheat: use libstdc++'s implementation
    template <typename T>
    static constexpr size_t accepted_index =
        std::__find_detail::__find_index<T &&, Ts &&...>();

    template <class F>
    constexpr auto with_index(F &&f) const -> decltype(auto)
    {
        return mp_with_index<sizeof...(Ts)>(index_, (F &&)f);
    }

public:
    using first_type = Ts...[0];

    constexpr Variant()
        requires std::is_default_constructible_v<first_type>
        // TODO: should this work: storage_{. [: get_nth_nsdm(0) :]{} }
        : storage_ { .empty = {} }
        , index_(0)
    {
        std::construct_at(&storage_.[:get_nth_nsdm(0):]);
    }

    constexpr ~Variant()
        requires(std::is_trivially_destructible_v<Ts> and ...)
    = default;

    constexpr ~Variant()
    {
        if(index_ != -1)
        {
            with_index([&](auto I) {
                std::destroy_at(&storage_.[:get_nth_nsdm(I):]);
            });
        }
    }

    template <typename T, size_t I = accepted_index<T>>
        requires(!std::is_base_of_v<Variant, std::decay_t<T>>)
    constexpr Variant(T &&t) : storage_ { .empty = {} }
                             , index_((int)I)
    {
        std::construct_at(&storage_.[:get_nth_nsdm(I):], (T &&)t);
    }

    // you can't actually express this constraint nicely until P2963
    constexpr Variant(Variant const &)
        requires(std::is_trivially_copyable_v<Ts> and ...)
    = default;

    constexpr Variant(Variant const &rhs)
        requires((std::is_copy_constructible_v<Ts> and ...) and
                    not(std::is_trivially_copyable_v<Ts> and ...))
        : storage_ { .empty = {} }
        , index_(-1)
    {
        rhs.with_index([&](auto I) {
            constexpr auto nsdm = get_nth_nsdm(I);
            std::construct_at(&storage_.[:nsdm:], rhs.storage_.[:nsdm:]);
            index_ = I;
        });
    }

    constexpr auto index() const -> int
    {
        return index_;
    }

    template <class F>
    constexpr auto visit(F &&f) const -> decltype(auto)
    {
        if(index_ == -1)
        {
            throw std::bad_variant_access();
        }

        return mp_with_index<sizeof...(Ts)>(index_,
            [&](auto I) -> decltype(auto) {
                return std::invoke((F &&)f, storage_.[:get_nth_nsdm(I):]);
            });
    }
};

inline void print_variants()
{
    Variant<int, char> v1;

    static_assert(std::is_trivially_copyable_v<decltype(v1)>);

    Variant<int, char> v2 = 5;
    Variant<int, char> v3 = v2;
    Variant<int, char> v4 = 'h';

    auto show = [](auto v) {
        std::cout << "holding value " << v << " of type '"
                  << display_string_of(type_of(^^v)) << "'\n";
    };

    v1.visit(show);
    v2.visit(show);
    v3.visit(show);
    v4.visit(show);
}
