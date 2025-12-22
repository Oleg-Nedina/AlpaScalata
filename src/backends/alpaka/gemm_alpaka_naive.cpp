// src/backends/alpaka/gemm_alpaka_naive.cpp
#include "gemm/gemm.hpp"

#include <alpaka/alpaka.hpp>
#include <cstring>
#include <cstddef>

namespace gemm
{
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    using TagAcc = alpaka::TagGpuCudaRt;
#else
    using TagAcc = alpaka::TagCpuSerial;
#endif

using TagHost = alpaka::TagCpuSerial;

struct GemmNaiveKernel
{
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        float const* A,
        float const* B,
        float* C,
        int m, int n, int k) const
    {
        // 2D global thread index: x = col, y = row
        auto const g = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        int col = static_cast<int>(g[0]);
        int row = static_cast<int>(g[1]);

        if(row >= m || col >= n) return;

        float sum = 0.0f;
        for(int kk = 0; kk < k; ++kk)
        {
            // row-major
            sum = alpaka::math::fma(acc, A[row * k + kk], B[kk * n + col], sum);
        }
        C[row * n + col] = sum;
    }
};

void gemm_alpaka_naive(float const* Ah, float const* Bh, float* Ch, GemmShape s)
{
    using Idx = std::size_t;
    using Dim1 = alpaka::DimInt<1>;
    using Dim2 = alpaka::DimInt<2>;

    // Devices + queue
    auto devAcc  = alpaka::getDevByIdx<alpaka::Pltf<TagAcc>>(0u);
    auto queue   = alpaka::Queue<TagAcc, alpaka::Blocking>{devAcc};
    auto devHost = alpaka::getDevByIdx<alpaka::Pltf<TagHost>>(0u);

    // -------- linear sizes --------
    Idx sizeA = static_cast<Idx>(s.m) * static_cast<Idx>(s.k);
    Idx sizeB = static_cast<Idx>(s.k) * static_cast<Idx>(s.n);
    Idx sizeC = static_cast<Idx>(s.m) * static_cast<Idx>(s.n);

    alpaka::Vec<Dim1, Idx> extA{sizeA};
    alpaka::Vec<Dim1, Idx> extB{sizeB};
    alpaka::Vec<Dim1, Idx> extC{sizeC};

    // -------- device buffers (1D) --------
    auto A_d = alpaka::allocBuf<float, Idx>(devAcc,  extA);
    auto B_d = alpaka::allocBuf<float, Idx>(devAcc,  extB);
    auto C_d = alpaka::allocBuf<float, Idx>(devAcc,  extC);

    // -------- host staging buffers (1D) --------
    auto A_h = alpaka::allocBuf<float, Idx>(devHost, extA);
    auto B_h = alpaka::allocBuf<float, Idx>(devHost, extB);
    auto C_h = alpaka::allocBuf<float, Idx>(devHost, extC);

    // fill staging
    std::memcpy(alpaka::getPtrNative(A_h), Ah, sizeof(float) * sizeA);
    std::memcpy(alpaka::getPtrNative(B_h), Bh, sizeof(float) * sizeB);

    // H->D
    alpaka::memcpy(queue, A_d, A_h, extA);
    alpaka::memcpy(queue, B_d, B_h, extB);

    // -------- launch (2D like CUDA) --------
    using Acc = alpaka::Acc<TagAcc, Dim2, Idx>;

    alpaka::Vec<Dim2, Idx> threads{16u, 16u};
    alpaka::Vec<Dim2, Idx> elems{static_cast<Idx>(s.n), static_cast<Idx>(s.m)};

    alpaka::KernelCfg<Acc> cfg{elems, threads};
    GemmNaiveKernel kernel{};
    auto workDiv = alpaka::getValidWorkDiv(cfg, devAcc, kernel,
                                           alpaka::getPtrNative(A_d),
                                           alpaka::getPtrNative(B_d),
                                           alpaka::getPtrNative(C_d),
                                           s.m, s.n, s.k);

    alpaka::exec<Acc>(queue, workDiv, kernel,
                      alpaka::getPtrNative(A_d),
                      alpaka::getPtrNative(B_d),
                      alpaka::getPtrNative(C_d),
                      s.m, s.n, s.k);

    alpaka::wait(queue);

    // D->H
    alpaka::memcpy(queue, C_h, C_d, extC);
    alpaka::wait(queue);

    // staging -> output
    std::memcpy(Ch, alpaka::getPtrNative(C_h), sizeof(float) * sizeC);
}

} // namespace gemm

