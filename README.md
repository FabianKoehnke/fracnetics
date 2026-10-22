
![Run Tests](https://github.com/FabianKoehnke/fracnetics/actions/workflows/tests.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/fracnetics.svg)](https://pypi.org/project/fracnetics/)
[![Documentation Status](https://readthedocs.org/projects/fracnetics/badge/?version=latest)](https://fracnetics.readthedocs.io/en/latest/?badge=latest)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fabian-koehnke/)

# Fracnetics

**Fracnetics** is a high-performance **C++ library with Python bindings** for **Genetic Network Programming (GNP)** — an evolutionary approach that evolves **directed graph structures** for decision-making and control tasks.

> Bridging evolution, networks, and fractals into one framework.

---

## 🔬 Overview

**Fracnetics** extends **Genetic Network Programming (GNP)** with **variable-size** networks and **fractal geometry**, designed to model adaptive, recursive, and self-similar network structures inspired by biological evolution and natural growth patterns.

It is designed for researchers and developers exploring:

- Evolutionary network design
- Obtain fully interpretable networks
- Training agents in dynamic environments
- Directed multigraph network architecture
- variable-size networks based on [Fabian Köhnke and Christian Borgelt, 2025](https://doi.org/10.1007/978-3-031-90062-4_18)
- Complex systems with self-similar topology

---

## 🚀 Installation

```bash
pip install fracnetics
```

---
## 📓 Colab Tutorials

Small Tutorials Using Fracnetics

This notebooks demonstrate how to use the Fracnetics library to solve the CartPole environment problem from Gymnasium (a fork of OpenAI Gym) and the simple Irisdata classification.

**CartPole Tutorial**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FabianKoehnke/fracnetics/blob/main/examples/example_CartPole.ipynb)

**Iris Tutorial**  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FabianKoehnke/fracnetics/blob/main/examples/example_IRIS.ipynb)
 
---

## 🦾 Features

- **Population Management**  
  - Create and manage a full population of `Network` individuals.  
  - Graph-Based Evolution: Networks evolve as directed graphs with judgment nodes (conditional decisions) and processing nodes (actions), enabling node reuse and cyclic execution paths

- **Fitness Evaluation**  
    - Built-in support for OpenAI Gymnasium environments
    - Traverse a network path via ``callTraversePath`` and calculate any own fitness/ target function

- **Selection & Elitism**: Tournament selection with configurable size.

- **Mutation Operators**  
  - **Edge mutation** (altering connections between nodes).  
  - **Boundary mutations** (multiple variants):  
    - Uniform  
    - Normally distributed  
    - Network-size dependent sigma scaling  
    - Edge-count dependent sigma scaling  
    - Fractal-based mutation  

- **Crossover**: Node exchange between individuals.

- **Innovative Opreators**  
    - **Add/Delete Nodes** – dynamic structural changes in networks.
    - **Fractal Geometry Integration** – hierarchical boundary generation via production rules (L-systems-style subdivision).

---


## 🔁 One Generation

The steps of a generation must run in this order; every step is optional:

```python
pop.gymnasium(env, seeds=seeds, dMax=10, maxSteps=500, maxConsecutiveP=5)  # evaluation
pop.callAgeUnusedNodes()                       # once
pop.tournamentSelection(N=5, E=1)              # or lexicaseSelection(), once
pop.updateAdaptiveK()
pop.crossover(probability, type="caseSpecialistAppend")  # once, before callAddDelNodes
pop.callAddDelNodes(minFeatures, maxFeatures, junk=0.1)
pop.crossover(probability, type="semantic")
pop.callEdgeMutation(0.05, 0.05)               # mutations in any order
pop.callBoundaryMutationNormal(0.05, 0.2)
pop.capLineageAfterMutation()
```

The evaluation writes the node usage flags and the fitness that the later steps read, and
every change to the networks makes them stale: `callAddDelNodes()` after a mutation could
delete a node that has just become active, a selection after a crossover compares fitness
values of networks that no longer exist. The population therefore raises a `RuntimeError`
for a step out of order until the next evaluation. `gymnasium()`, `callTraversePath()`,
`accuracy()` and `ParallelEvaluator.evaluate()` count as evaluations; after an evaluation of
your own call `pop.markEvaluated()`. To call single operators on purpose (e.g. in tests), set
`pop.checkOrder = False`.

---
