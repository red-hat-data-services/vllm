#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"

/**
 * This file defines Gemm kernel configurations for SM90 (fp8) based on the Gemm
 * shape.
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

#define CALL_CUTLASS_GEMM                                                 \
  cutlass_gemm_caller<                                                    \
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape, \
                      KernelSchedule, EpilogueSchedule>>(                 \
      out, a, b, std::forward<EpilogueArgs>(args)...);

struct sm90_fp8_config_M64 {
  // M in [1, 64]
  using ClusterShape = Shape<_1, _8, _1>;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;

  template <typename InType, typename OutType,
            template <typename, typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    uint32_t const n = out.size(1);

    if (n < 8 * 1024) {
      using TileShape = Shape<_64, _64, _128>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      CALL_CUTLASS_GEMM

    } else if (n < 16 * 1024) {
      using TileShape = Shape<_64, _128, _128>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      CALL_CUTLASS_GEMM

    } else {
      using TileShape = Shape<_64, _128, _128>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
      CALL_CUTLASS_GEMM
    }
  }
};

struct sm90_fp8_config_M128 {
  // M in (64, 128]

  template <typename InType, typename OutType,
            template <typename, typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    uint32_t const n = out.size(1);

    if (n <= 4 * 1024) {
      using TileShape = Shape<_64, _64, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      using EpilogueSchedule =
          typename cutlass::epilogue::TmaWarpSpecializedCooperative;

      CALL_CUTLASS_GEMM

    } else if (n <= 8 * 1024) {
      using TileShape = Shape<_64, _128, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      using EpilogueSchedule =
          typename cutlass::epilogue::TmaWarpSpecializedCooperative;
      CALL_CUTLASS_GEMM

    } else if (n <= 16 * 1024) {
      using TileShape = Shape<_128, _128, _128>;
      using ClusterShape = Shape<_1, _8, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM

    } else if (n <= 24 * 1024) {
      using TileShape = Shape<_128, _64, _128>;
      using ClusterShape = Shape<_1, _2, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM

    } else {
      using TileShape = Shape<_128, _64, _128>;
      using ClusterShape = Shape<_1, _8, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM
    }
  }
};

struct sm90_fp8_config_M256 {
  // M in (128, 256]

  template <typename InType, typename OutType,
            template <typename, typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    uint32_t const n = out.size(1);

    if (n <= 4 * 1024) {
      using TileShape = Shape<_64, _128, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      using EpilogueSchedule =
          typename cutlass::epilogue::TmaWarpSpecializedCooperative;
      CALL_CUTLASS_GEMM

    } else if (n <= 8 * 1024) {
      using TileShape = Shape<_128, _128, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM

    } else if (n <= 16 * 1024) {
      using TileShape = Shape<_128, _256, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
      using EpilogueSchedule =
          typename cutlass::epilogue::TmaWarpSpecializedCooperative;
      CALL_CUTLASS_GEMM

    } else if (n <= 24 * 1024) {
      using TileShape = Shape<_128, _128, _128>;
      using ClusterShape = Shape<_2, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM

    } else {
      using TileShape = Shape<_256, _128, _64>;
      using ClusterShape = Shape<_1, _8, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
      using EpilogueSchedule =
          typename cutlass::epilogue::TmaWarpSpecializedCooperative;
      CALL_CUTLASS_GEMM
    }
  }
};

struct sm90_fp8_config_M3072 {
  // M in (256, 3072]

  template <typename InType, typename OutType,
            template <typename, typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    uint32_t const n = out.size(1);

    if (n <= 4 * 1024) {
      using TileShape = Shape<_128, _128, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM

    } else if (n <= 8 * 1024) {
      using TileShape = Shape<_128, _256, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
      using EpilogueSchedule =
          typename cutlass::epilogue::TmaWarpSpecializedCooperative;
      CALL_CUTLASS_GEMM

    } else if (n <= 16 * 1024) {
      using TileShape = Shape<_128, _128, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM

    } else if (n <= 24 * 1024) {
      using TileShape = Shape<_128, _256, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
      using EpilogueSchedule =
          typename cutlass::epilogue::TmaWarpSpecializedCooperative;
      CALL_CUTLASS_GEMM

    } else {
      using TileShape = Shape<_64, _256, _128>;
      using ClusterShape = Shape<_1, _1, _1>;
      using KernelSchedule =
          cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
      using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
      CALL_CUTLASS_GEMM
    }
  }
};

struct sm90_fp8_config_default {
  // M in (3072, inf)

  template <typename InType, typename OutType,
            template <typename, typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_2, _1, _1>;
    using KernelSchedule =
        cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
    CALL_CUTLASS_GEMM
  }
};

#undef CALL_CUTLASS_GEMM

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm90_fp8_dispatch(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  uint32_t const m = a.size(0);

  if (m <= 64) {
    // m in [1, 64]
    return sm90_fp8_config_M64::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 128) {
    // m in (64, 128]
    return sm90_fp8_config_M128::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 256) {
    // m in (128, 256]
    return sm90_fp8_config_M256::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 3072) {
    // m in (256, 3072]
    return sm90_fp8_config_M3072::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (3072, inf]
    return sm90_fp8_config_default::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm90_fp8_epilogue(torch::Tensor& out,
                                         torch::Tensor const& a,
                                         torch::Tensor const& b,
                                         EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                          cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                          cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace vllm