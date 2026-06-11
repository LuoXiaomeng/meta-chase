# Meta-Chase (AFF) Engine

This repository accompanies the paper ***From Chase to Meta-Chase***. It contains a
minimal, runnable implementation of the **Meta-Chase** engine (the
Agentized-Fishing-Fort, "AFF").

## Paper

The submitted version is page-limited; the **full version** is bundled here as
[`paper.pdf`](paper.pdf). 

## What this code does

Given a property graph and a set of Graph Association Rules (GARs), the engine runs a
task-aware **chase**: it decomposes the rules into tasks, schedules them under a
policy over a shared overlay, and derives the decision facts `P_out`. It also resolves
competing values on a functional decision key via **canonicalization** (and a
two-phase protocol).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy (required); tqdm (optional)
```

## Run the demo

```bash
python demo.py
```

This loads a tiny self-contained graph + rule set (`sample_data/demo/`), runs the
chase, and prints the derived `P_out` facts — both the raw (multi-valued) output and
the canonicalized output, where two competing rules disagree on a functional decision
key and max-PCA canonicalization resolves the conflict. Expected tail:

```
RESULT: engine derived 7 P_out facts (6 after canonicalization).
```

## Package layout

```
demo.py                          end-to-end runnable demo
paper.pdf                        full version of the paper
sample_data/demo/                the bundled demo dataset
metachase/                       the engine, layered low -> high
  graph/      graph (CSR) + processed-graph loader + GAR/pattern + derived overlay
  match/      VF2 subgraph matching (+ semi-naive delta)
  chase/      round-based chase coordinator + fixpoint semantics
  confluence/ canonicalization, two-phase protocol, community check
  policy.py   task-scheduling policies
  partition.py  task-construction strategies (SCC-merge / random)
  aff/        AFFConfig + AFFRunner (the single chase entry point) + FF baselines
  rules/      AMIE rule loading, SCC clustering, P_out config, dataset registry
  session.py  the shared `Context` (dataset -> graph + rules + tasks)
  metrics.py  evaluation metrics
```
