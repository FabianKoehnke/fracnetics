---
title: 'Fracnetics: A Python library with C++ core engine for Genetic Network Programming (GNP)'
tags:
  - evolutionary elgorithms 
  - genetic network programming 
  - fractals 
authors:
  - name: Fabian Köhnke 
    orcid: 0009-0007-2726-5980 
    affiliation: 1 
  - name: Christian Borgelt 
    orcid: XXXX
    affiliation: 1
affiliations:
 - name: University of Salzburg, Austria, Dept. of Artificial Intelligence 
   index: 1
   ror: 00hx57361
date: 2 November 2025
bibliography: paper.bib
---

# Summary 

'Fracnetics' is Python library with C++ core engine for Genetic Network Programming (GNP) — 
an evolutionary algorithms that evolves directed graph structures. 

Evolutionary algorithms are based on the theory of biological evolution developed by Charles Darwin. 
The basic principle of biological evolution can be formulated as follows:
"Beneficial traits resulting from random variation are favored by natural selection" [@Kruse.2015]. 

Fracnetics is a portmanteau composed of fractals, network, and genetic.
It refers to Genetic Network Programming as the underlying algorithm and, with fractals, 
additionally hints at an extension of GNP that enhances the architecture of the networks with fractal properties.

With Fracnetics, GNP can easily be applied to any problem through its Python API, allowing agents/models to be trained.
For the evolutionary process and its network-specific topology, the API provides classes and methods, 
such as the initialization of populations, tournament selection, 
and several operators for recombination and modifications of the gene structure (crossover and mutation).

# Genetic Network Programming and Extensions

Genetic Network Programming (GNP) is an extension of Genetic Programming
(GP), which has been developed and researched since around 2000 [@Katagiri.2000] and has
been used to solve optimization problems with large search spaces. Due to the
network structure of individuals and the possibility of reusing nodes, an implicit
memory can develop. As a consequence, GNP targets dynamic environments
and has been applied successfully in fields such as robotics and financial market
analysis [@Chen.2009II][@Mabu.2007].

We extended the Genetic Network Programming by a new mutation operator,
which allows for a variable number of nodes and edges per individual [@kohnke2025variable].
With this operator, the search space is significantly extended, but without the risk of incurring the bloat problem. 
The operator is fitness neutral and has no hyper-parameter. 
Due to higher flexibility, it is now possible for GNP to automatically adapt to the complexity of a given task
and to find suitable features, especially for high dimensional data sets.

Future research will now also explore fractal structures in the data space, for which Fracnetics was developed.

# Statement of Need

Fracnetics is the first open-source library that provides an API for Genetic Network Programming.
By enabling easy application to any individual optimization task, it simplifies and intensifies research on GNP.
Of course, Fracnetics can also be used for all kinds of industrial optimizations and is particularly suited
for dynamic environments such as finance and robotics.

Included is the novel mutation operator introduced in [@kohnke2025variable], which can modify the topology of a network
by growing or shrinking networks, thereby can automatically adapt to the complexity of a given task. With this innovative 
operator, solving a complex optimization task such as portfolio optimization is feasible with a minimal initial network size of only three nodes.

The fractal extension of the network topology additionally provides a method for dividing the feature space
into self-similar intervals. This is especially useful for datasets where information exists at multiple scales.
Financial markets, for example, often exhibit fractal-like behavior: price movements at different time scales [@Mandelbrot.2005],
or environmental interactions in robotics often display recursive or repeating patterns.

The trained models (networks) remain fully interpretable, thus making a fundamental contribution to interpretable AI.

# Key Features

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

- **Innovative Operators**  
    - **Add/Delete Nodes** – structural changes in network topology.
    - **Fractal Geometry Integration** – hierarchical boundary generation via production rules (L-systems-style subdivision).

# References 
