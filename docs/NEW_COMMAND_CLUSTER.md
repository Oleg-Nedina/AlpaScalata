# 🧠 MEGA-RIASSUNTO — Git, Cluster, Rete, Trasferimento Dati (NO Python)

## Contesto

Progetto **AlpaScalata / Alpaka** su cluster PoliMi (login01 / gpu01).
Sviluppo in **locale**, esecuzione e benchmark su **cluster**, analisi **in locale**.

---

## 1️⃣ Problema — Git ha smesso di funzionare sul cluster

### Sintomi

* `git clone / pull / push` non funzionano
* Errori di timeout su:

  * porta 22 (SSH)
  * porta 443 (HTTPS)

### Verifiche fatte

```bash
getent hosts github.com     # DNS OK
nc -vz github.com 22        # timeout
nc -vz github.com 443       # timeout
curl https://github.com     # timeout
```

### Conclusione

* Il cluster **non ha accesso a Internet**
* Cambiamento di **policy di rete**
* Non è un problema di Git o di chiavi

### Decisione

❌ Niente GitHub dal cluster
✅ Git **solo in locale**

---

## 2️⃣ Problema — Come portare il codice sul cluster senza Git

### Soluzione adottata

Usare **`rsync` dal PC locale verso il cluster**
(il PC è l’unico nodo con accesso esterno).

### Comando ESATTO (PC → cluster)

```bash
rsync -av --delete -e "ssh" \
  --exclude .git/ \
  --exclude build/ \
  ~/UNI/AMSC/AlpaScalata/ \
  u10905938@10.78.18.100:~/AlpaScalata/
```

📌 Effetto:

* il cluster viene **allineato** al locale
* sovrascrittura completa
* `.git` escluso

---

## 3️⃣ Problema — rsync / ssh non funzionano da gpu01

### Sintomo

```text
Permission denied (publickey)
```

### Diagnosi

* `gpu01` è un **compute node**
* non ha (e non deve avere) le chiavi SSH
* non è pensato per fare connessioni in uscita

### Regola fondamentale

❌ Mai fare rsync / ssh **da gpu01**
✅ Tutti i trasferimenti passano dal **PC locale**

---

## 4️⃣ Problema — `git status` non funziona sul cluster

### Motivo

* `.git` non viene copiato sul cluster (scelta voluta)
* la directory sul cluster **non è un repository Git**

### Soluzione

Nessuna azione richiesta:

* Git serve solo in locale
* sul cluster servono solo i sorgenti per build/run

---

## 5️⃣ Problema — Come riportare i risultati in locale

### Errore iniziale

Tentare di fare:

```text
gpu01 → login → PC
```

❌ Fallisce (chiavi / policy)

### Soluzione corretta

Scaricare i risultati **direttamente dal PC locale**

### Comando ESATTO (PC ← cluster)

```bash
mkdir -p ~/UNI/AMSC/AlpaScalata/data/results
rsync -av -e "ssh" \
  u10905938@10.78.18.100:~/AlpaScalata/data/results/ \
  ~/UNI/AMSC/AlpaScalata/data/results/
```

📌 Questo comando:

* va eseguito **dal PC**
* è l’unico modo corretto
* non richiede nulla dal cluster

---

## 6️⃣ WORKFLOW FINALE (definitivo)

```text
[PC] modifica codice
[PC] git commit / push
[PC] rsync → cluster
[CLUSTER] build / run / benchmark
[PC] rsync ← results
```

---

## 7️⃣ CHEAT-SHEET FINALE (solo comandi utili)

### 🔼 PC → cluster (codice)

```bash
rsync -av --delete -e "ssh" \
  --exclude .git/ --exclude build/ \
  ~/UNI/AMSC/AlpaScalata/ \
  u10905938@10.78.18.100:~/AlpaScalata/
```

### 🔽 PC ← cluster (risultati)

```bash
rsync -av -e "ssh" \
  u10905938@10.78.18.100:~/AlpaScalata/data/results/ \
  ~/UNI/AMSC/AlpaScalata/data/results/
```

---

