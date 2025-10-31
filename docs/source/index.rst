.. Fracnetics documentation master file, created by
   sphinx-quickstart on Mon Oct  6 21:09:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fracnetics Documentation
========================

**Fracnetics** is a high-performance **C++ library with Python bindings** for **Genetic Network Programming (GNP)** — an evolutionary approach that evolves **directed graph structures** for decision-making and control tasks.

It extends GNP with **variable-size** networks and **fractal-based edge patterns**, inspired by **L-systems** and biological growth,  
to create hierarchical, self-similar decision boundaries.

It is designed for researchers and developers exploring:

- Evolutionary network design
- Directed multigraph network architecture
- variable-size networks
- Complex systems with self-similar topology

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   cpp_api
   python_api
   examples

Installation
---------------------

```
pip install fracnetics
```

Architecture Overview
---------------------

Fracnetics is organized into three core components:

1. **Population** – Manages selection, mutation, crossover, and generations
2. **Network** – Directed multigraph with evolutionary operators and fitness functions  
3. **Node** – Basic unit (start, judgment, processing)  

Key Features
------------

**Population Management**

- Create and manage a full population of ``Network`` individuals.
- Graph-Based Evolution: Networks evolve as directed graphs with judgment nodes (conditional decisions) and processing nodes (actions), enabling node reuse and cyclic execution paths

**Fitness Evaluation**

- Built-in support for OpenAI Gymnasium environments
- Traverse a network path via ``callTraversePath`` and calculate any own fitness/ target function

**Selection & Elitism**: Tournament selection with configurable size.

**Mutation Operators**

- **Edge mutation** – altering connections between nodes.
- **Boundary mutations** (multiple variants):
  - Uniform
  - Normally distributed
  - Network-size dependent sigma scaling
  - Edge-count dependent sigma scaling
  - Fractal-based mutation

**Crossover**: Node exchange between individuals.

**Innovative Operators**

- **Add/Delete Nodes** – dynamic structural changes in networks.
- **Fractal Geometry Integration** – hierarchical boundary generation via production rules (L-systems-style subdivision).


Applications
------------

Fracnetics targets domains requiring **adaptive, interpretable graph structures**:

- Real-time control and robotics  
- Classification
- Reinforcement learning problems
- Combinatorial and parameter optimization  
- Time series prediction with recurrent connections

