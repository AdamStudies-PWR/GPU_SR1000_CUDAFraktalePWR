#pragma once

namespace ConstexprMath {

template <typename T>
constexpr T
abs(const T x) noexcept
{
    return x == T(0) ? T(0) : x < T(0) ? -x : x;
}

template <typename T>
constexpr T
divUp(const T a, const T b) noexcept
{
    return a % b != 0 ? (a / b) + 1 : (a / b);
}

}
