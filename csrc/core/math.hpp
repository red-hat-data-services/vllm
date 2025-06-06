#pragma once

#include <climits>
#include <iostream>

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// Round a down to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

// Round a up to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
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
