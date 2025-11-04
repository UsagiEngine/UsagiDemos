#pragma once

#include <print>
#include <string>
#include <type_traits>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

template <typename E, bool Enumerable = is_enumerable_type(^^E)>
    requires std::is_enum_v<E>
constexpr std::string_view enum_to_string(E value)
{
    if constexpr(Enumerable)
        template for(constexpr auto e : define_static_array(enumerators_of(
                         ^^E))) if(value == [:e:]) return identifier_of(e);

    return "<unnamed>";
}

inline void print_enum_to_string()
{
    enum Color : int;
    static_assert(enum_to_string(Color(0)) == "<unnamed>");
    std::println("Color 0: {}", enum_to_string(Color(0))); // prints '<unnamed>'

    enum Color : int
    {
        red,
        green,
        blue
    };

    static_assert(enum_to_string(Color::red) == "red");
    static_assert(enum_to_string(Color(42)) == "<unnamed>");
    std::println("Color 0: {}", enum_to_string(Color(0))); // prints 'red'
}
