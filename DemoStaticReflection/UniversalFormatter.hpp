#pragma once

#include <format>
#include <print>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

struct universal_formatter
{
    constexpr auto parse(auto &ctx)
    {
        return ctx.begin();
    }

    template <typename T>
    auto format(T const &t, auto &ctx) const
    {
        std::string_view type_label = "(unnamed-type)";
        if constexpr(has_identifier(^^T)) type_label = identifier_of(^^T);
        auto out = std::format_to(ctx.out(), "{}{{", type_label);

        auto delim = [first = true, &out]() mutable {
            if(!first)
            {
                *out++ = ',';
                *out++ = ' ';
            }
            first = false;
        };

        constexpr auto access_ctx = std::meta::access_context::unchecked();

        template for(constexpr auto base :
            define_static_array(bases_of(^^T, access_ctx)))
        {
            delim();
            out = std::format_to(
                out, "{}", (typename[:type_of(base):] const &)(t));
        }

        template for(constexpr auto mem :
            define_static_array(nonstatic_data_members_of(^^T, access_ctx)))
        {
            delim();

            std::string_view mem_label = "unnamed-member";
            if constexpr(has_identifier(mem)) mem_label = identifier_of(mem);

            if constexpr(is_bit_field(mem) && !has_identifier(mem))
                out = std::format_to(out, "(unnamed-bitfield)");
            else
                out = std::format_to(out, ".{}={}", mem_label, t.[:mem:]);
        }

        *out++ = '}';
        return out;
    }
};

struct B
{
    int m0 = 0;
};

struct X : B
{
    int m1 = 1;
};

struct Y : B
{
    int m2 = 2;
};

class Z
    : public X
    , private Y
{
    struct
    {
        int : 0;
    } s;

    int m3 = 3;
    int m4 = 4;
};

template <>
struct std::formatter<B> : universal_formatter
{
};

template <>
struct std::formatter<X> : universal_formatter
{
};

template <>
struct std::formatter<Y> : universal_formatter
{
};

template <>
struct std::formatter<Z> : universal_formatter
{
};

template <>
struct std::formatter<decltype(Z::s)> : universal_formatter
{
};

inline void print_z()
{
    std::println("{}", Z());
    // Z{X{B{.m0=0}, .m1 = 1}, Y{{.m0=0}, .m2 = 2}, .m3 = 3, .m4 = 4}
}
