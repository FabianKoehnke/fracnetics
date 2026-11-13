.. Fracnetics documentation master file, created by
   sphinx-quickstart on Mon Oct  6 21:09:46 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fracnetics Documentation
========================

**Fracnetics** is a high-performance **C++ library with Python bindings** for **Genetic Network Programming (GNP)** — an evolutionary approach that evolves **directed graph structures** for decision-making and control tasks.

The Library extends **Genetic Network Programming (GNP)** with **variable-size** networks and **fractal geometry**, designed to model adaptive, recursive, and self-similar network structures inspired by biological evolution and natural growth patterns.


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


One Generation
--------------

The steps of a generation must run in this order; every step is optional:

.. code-block:: python

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

The evaluation writes the node usage flags and the fitness that the later steps read, and
every change to the networks makes them stale: ``callAddDelNodes()`` after a mutation could
delete a node that has just become active, a selection after a crossover compares fitness
values of networks that no longer exist. Since version 4.0.0 the population therefore raises a
``RuntimeError`` for a step out of order until the next evaluation (see ``Population::Step``).
``gymnasium()``, ``callTraversePath()``, ``accuracy()`` and ``ParallelEvaluator.evaluate()``
count as evaluations; after an evaluation of your own call ``pop.markEvaluated()``. To call
single operators on purpose (e.g. in tests), set ``pop.checkOrder = False``.

Hall of fame for noisy evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With stochastic episodes the training fitness on a few cases ranks the population poorly.
``fracnetics.hall_of_fame.HallOfFame`` re-scores the best candidates and the current champion
on the same fresh cases every generation; the winner becomes the champion, and the round order
fills the elite slots after the selection (``update()`` between evaluation and selection,
``placeElites()`` right after it). With ``memory=True`` the champion accumulates its scores,
and a round winner must confirm on further fresh cases before it takes over.

Data-guided boundary mutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every evaluation records, per judgment node, the range of the values it judged
(``seenMin``/``seenMax``) and over the node's lifetime (``everSeenMin``/``everSeenMax``).
With ``dataGuided=True`` the uniform boundary mutation redraws a boundary only inside its
gap ∩ that range, and the normal mutation scales its step with the range instead of the gap;
``everSeen=True`` uses the lifetime range. The size-adaptive normal mutations take the same
options, and the fractal mutation lays its partition over the range (outermost boundaries
stay at the feature's min/max). With ``startInRange=True`` a normal step starts from the
nearest point of the range, so a boundary outside the judged values enters at their edge
instead of wandering without direction.


Applications
------------

Fracnetics targets domains requiring **adaptive, interpretable graph structures**:

- Real-time control and robotics  
- Classification
- Reinforcement learning problems
- Combinatorial and parameter optimization  
- Time series prediction with recurrent connections

