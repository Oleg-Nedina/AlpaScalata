
## src/common/

Codice condiviso tra tutti i backend.

Include:
- implementazione di riferimento CPU (baseline di correttezza)
- generatori e loader di matrici dense
- utility comuni (timer, parsing CLI, gestione errori)

Questo codice **non contiene ottimizzazioni GPU** ed è usato
per verificare correttezza e riproducibilità dei risultati.
