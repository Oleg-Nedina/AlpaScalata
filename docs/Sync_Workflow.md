# Fast Recap — Git, Cluster, Net, Data Transfer (NO Python)

## Context

Project **AlpaScalata / Alpaka** over cluster PoliMi (login01 / gpu01).
 **Local** development, execution and benchmark over **cluster**, **Local** analysis.

---

## Problem — Git stop to work on cluster

### Symptoms

* `git clone / pull / push` does not work
* Timeout errors on:

  * port 22 (SSH)
  * port 443 (HTTPS)

### Checks

```bash
getent hosts github.com     # DNS OK
nc -vz github.com 22        # timeout
nc -vz github.com 443       # timeout
curl https://github.com     # timeout
```

### Conclusion

* The cluster **does not have Internet access**
* Change of **net policy**
* Not a git or key problem

### Decision

No GitHub on cluster
Git **local only**

---

## Problem — How to get code on cluster without Git

### Adopted solution

Using **`rsync` on local devide toward the cluster**
(the device is the only node with external access).

### EXACT command (device → cluster)

```bash
rsync -av --delete -e "ssh" \
  --exclude .git/ \
  --exclude build/ \
  ~/UNI/AMSC/AlpaScalata/ \
  u10905938@10.78.18.100:~/AlpaScalata/
```

Effect:

* the cluster gets **aligned** with local
* complet overwrite
* `.git` excluded

---

## Problem — rsync / ssh does not work on gpu01

### Symptom

```text
Permission denied (publickey)
```

### Diagnosis

* `gpu01` is a **compute node**
* it does not have (and does not have to) SSH keys
* it isn't projected for connections toward the outside

### Fundamental rule

Never do rsync / ssh **from gpu01**
Every transfer pass through **local device**

---

##  Problem — `git status` does not work over cluster

### Reason

* `.git` does not get compiled over cluster (intentional choice)
* the cluster directory **is not a Git repository**

### Solution

No action required:

* Git only needed in local
* on the cluster only needed source codes for build/run

---

##  Problem — How to bring results back in local

### Initial error

Trying to do:

```text
gpu01 → login → PC
```

Fails (keys / policy)

### Correct solution

Download results **directly from local device**

### EXACT command (PC ← cluster)

```bash
mkdir -p ~/UNI/AMSC/AlpaScalata/data/results
rsync -av -e "ssh" \
  u10905938@10.78.18.100:~/AlpaScalata/data/results/ \
  ~/UNI/AMSC/AlpaScalata/data/results/
```

This command:

* has to be executed **on the local device**
* is the only correct procedure
* does not require anything from the cluster-side

---

## FINAL WORKFLOW 

```text
[DEVICE] code modify
[DEVICE] git commit / push
[DEVICE] rsync → cluster
[CLUSTER] build / run / benchmark
[DEVICE] rsync ← results
```

---

## FINAL CHEAT-SHEET (useful commands only)

### Device → cluster (code)

```bash
rsync -av --delete -e "ssh" \
  --exclude .git/ --exclude build/ \
  ~/UNI/AMSC/AlpaScalata/ \
  u10905938@10.78.18.100:~/AlpaScalata/
```

### Device ← cluster (results)

```bash
rsync -av -e "ssh" \
  u10905938@10.78.18.100:~/AlpaScalata/data/results/ \
  ~/UNI/AMSC/AlpaScalata/data/results/
```

---

