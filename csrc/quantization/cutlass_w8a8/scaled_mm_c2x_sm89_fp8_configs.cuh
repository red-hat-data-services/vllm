template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_0 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<16, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 4;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_1 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 4;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_2 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 5;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_3 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<32, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 4;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_4 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 4;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_5 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<16, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 3;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_6 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<16, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 3;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle = typename cutlass::gemm::threadblock::
      GemmSplitKHorizontalThreadblockSwizzle;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_7 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<16, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 2;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle = typename cutlass::gemm::threadblock::
      GemmSplitKHorizontalThreadblockSwizzle;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_8 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<16, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 3;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_9 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 2;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle = typename cutlass::gemm::threadblock::
      GemmSplitKHorizontalThreadblockSwizzle;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_10 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 4;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_11 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 3;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_12 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 4;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle = typename cutlass::gemm::threadblock::
      GemmSplitKHorizontalThreadblockSwizzle;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_13 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 4;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<
          1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_14 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<32, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 3;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_15 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 3;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_16 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 3;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemm;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_config_17 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int32_t MainLoopStages = 2;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  using ThreadBlockSwizzle = typename cutlass::gemm::threadblock::
      GemmSplitKHorizontalThreadblockSwizzle;
  static constexpr cutlass::gemm::GemmUniversalMode GemmMode =
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel;

  using Cutlass2xGemm =
      vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                            InType, OutType, Epilogue, TileShape, WarpShape,
                            InstructionShape, MainLoopStages, FP8MathOperator,
                            ThreadBlockSwizzle, GemmMode>;
};