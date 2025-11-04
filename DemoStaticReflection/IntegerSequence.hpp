#pragma once

#include <utility>
#include <vector>

#include <Usagi/Library/Reflection/StaticReflection.hpp>

template <typename T>
consteval std::meta::info make_integer_seq_refl(T N)
{
    std::vector args { ^^T };
    for(T k = 0; k < N; ++k)
    {
        args.push_back(std::meta::reflect_constant(k));
    }
    return substitute(^^std::integer_sequence, args);
}

template <typename T, T N>
using _make_integer_sequence = [:make_integer_seq_refl<T>(N):];

static_assert(std::same_as<_make_integer_sequence<int, 10>,
    std::make_integer_sequence<int, 10>>);
