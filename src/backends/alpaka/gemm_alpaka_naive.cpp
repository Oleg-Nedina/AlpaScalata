#include "gemm/gemm.hpp"
#include <alpaka/alpaka.hpp>

namespace gemm {

using Idx = std::size_t;
using Dim2 = alpaka::DimInt<2>;

// ... Kernel struct rimane uguale ...

// Modifica la firma per accettare la queue per riferimento
template <typename TQueue>
void gemm_alpaka_naive(TQueue &queue, float const *A, float const *B, float *C,
                       GemmShape shape) {

  // Nota: Acc lo ricaviamo dalla queue se necessario, oppure usiamo quello
  // definito globalmente Se TQueue è la coda Alpaka corretta, ha già il device
  // associato.

  using Acc = typename TQueue::Acc; // Recuperiamo l'acceleratore dalla coda

  constexpr Idx TX = 16;
  constexpr Idx TY = 16;

  Idx blocksY = (Idx)((shape.m + (int)TY - 1) / (int)TY);
  Idx blocksX = (Idx)((shape.n + (int)TX - 1) / (int)TX);

  auto const gridThreadExtent = alpaka::Vec<Dim2, Idx>{blocksY, blocksX};
  auto const blockThreadExtent = alpaka::Vec<Dim2, Idx>{TY, TX};
  auto const elemExtent = alpaka::Vec<Dim2, Idx>{1u, 1u};

  alpaka::WorkDivMembers<Dim2, Idx> workDiv(gridThreadExtent, blockThreadExtent,
                                            elemExtent);

  GemmNaiveKernel kernel;

  // Usiamo la queue passata come argomento
  alpaka::exec<Acc>(queue, workDiv, kernel, A, B, C, shape.m, shape.n, shape.k);

  // Non fare wait qui se vuoi misurare il tempo asincrono fuori,
  // ma per ora lasciamo wait per semplicità di debug.
  alpaka::wait(queue);
}

// Istanziazione esplicita se necessario, o sposta tutto nell'header se usi
// template. Per semplicità nel tuo setup attuale senza template nell'header,
// mantieni i tipi fissi:
/*
void gemm_alpaka_naive(alpaka::Queue<Acc, alpaka::Blocking>& queue, float const*
A, ...){
   // implementazione come sopra senza template TQueue
}
*/

} // namespace gemm
