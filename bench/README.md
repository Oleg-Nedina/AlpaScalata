
## bench/

Benchmark delle prestazioni.

I benchmark misurano:
- tempo di esecuzione
- throughput (GFLOP/s)
- scalabilità forte e debole

Ogni benchmark utilizza l’API comune GEMM e può essere eseguito
con backend CUDA o Alpaka, garantendo confronti riproducibili.
