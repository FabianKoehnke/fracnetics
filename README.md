# Fracnetics

**Fracnetics** is a Python library for **Genetic Network Programming (GNP)** enhanced with **fractal geometry**, designed to model scalable, recursive, and self-similar network structures inspired by biological evolution and natural growth patterns.

> Bridging evolution, networks, and fractal recursion into one framework.

---

## 🔬 Overview

Fracnetics extends traditional **Genetic Network Programming** with **fractal expansion**, enabling dynamic and hierarchical structure generation. It is designed for researchers and developers exploring:

- Evolutionary network design
- Recursive architecture formation
- Bio-inspired optimization and simulation
- Complex systems with self-similar topology

---

## 🚀 Installation

```bash
pip install fracnetics
```

---
## 📓 Colab Tutorial

Small Tutorial Using Fracnetics

This notebook demonstrates how to use the Fracnetics library to solve the CartPole environment problem from Gymnasium (a fork of OpenAI Gym).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/FabianKoehnke/fracnetics/blob/main/notebooks/minExampleCartPole.ipynb)
 
---

## 🦾 Features

- **Population Management**  
  - Create and manage a full population of `Network` individuals.  
  - Random initialization with reproducible seed support.  

- **Fitness Evaluation**  
  - Generic fitness evaluation via template functions (`applyFitness`).  
  - Built-in evaluation modes:
    - Classification accuracy (`accuracy`)  
    - Reinforcement Learning environments of OpenAI Gym (`gymnasium`)  

- **Selection & Elitism**  
  - Tournament selection with configurable size.  
  - Automatic tracking and preservation of **elite individuals**.  
  - Population statistics: `bestFit`, `meanFitness`, `minFitness`.  

- **Mutation Operators**  
  - **Edge mutation** (altering connections between nodes).  
  - **Boundary mutations** (multiple variants):  
    - Uniform  
    - Normally distributed  
    - Network-size dependent sigma scaling  
    - Edge-count dependent sigma scaling  
    - Fractal-based mutation  

- **Crossover**  
  - Node exchange between individuals with configurable probability.  
  - Automatic handling of dangling edges.  
  - Elites excluded from crossover.  

- **Structural Variation**  
  - **Add/Delete Nodes**: dynamic structural changes in networks.  
  - Support for fractal judgment nodes with special production rules.

---

