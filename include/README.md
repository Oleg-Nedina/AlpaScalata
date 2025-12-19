
## include/

Header pubblici del progetto.

Questa cartella definisce l’**API comune** per il prodotto tra matrici dense (GEMM),
indipendente dal backend di esecuzione (CUDA o Alpaka).

Contiene:
- interfacce (`gemm.hpp`)
- tipi comuni (layout, datatype, configurazioni)
- viste non-owning su matrici (tensor views)

Tutti i backend devono implementare questa interfaccia senza modificarne la semantica,
così da permettere confronti equi di prestazioni.
