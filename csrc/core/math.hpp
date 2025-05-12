#pragma once

#include <climits>
#include <iostream>

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename T>
inline constexpr std::enable_if_t<std::is_integral_v<T>, T> ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// Compute the next multiple of a that is greater than or equal to b
template <typename A, typename B>
static inline constexpr auto next_multiple_of(A a, B b) {
  return ceil_div(b, a) * a;
}

// Compute the largest multiple of a that is less than or equal to b
template <typename A, typename B>
static inline constexpr auto prev_multiple_of(A a, B b) {
  return (b / a) * a;
}
