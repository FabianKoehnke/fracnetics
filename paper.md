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
an evolutionary algorithm that evolves directed graph structures. 

They are based on the theory of biological evolution developed by Charles Darwin. 
The basic principle of biological evolution can be formulated as follows:
Beneficial traits resulting from random variation are favored by natural selection [@Kruse.2015]. 

Fracnetics ist ein Portmanteau zusammengesetzt aus Fractals, Network und Genetic. 
Er deutet auf die Genetic Network Programming als zugrundeliegenden Algorithmus hin und
gibt mit Fractals einen zusätzlich Hinweis auf eine Erweiterung der GNP, die die
Architektur der Netzwerke um fraktale Eigenschaften erweitert. 

Mit Fracnetics lässt sich auf einfache Weise mittels API die GNP auf jegliche Problemstellungen anwenden und Agenten trainieren.
Für das evolutionäre Verfahren und deren netzwerkspezifische Topologie stellt die API mehrere Methoden wie z.B. 
die Erzeugung von Populationen, Turnierauswahl und mehrere Operatoren für die Rekombination und Veränderungen der Genstruktur zur Verfügung.

# Genetic Network Programming and Extensions

Genetic Network Programming (GNP) is an extension of Genetic Programming
(GP), which has been developed and researched since around 2000 [@Katagiri.2000] and has
been used to solve optimization problems with large search spaces. Due to the
network structure of individuals and the possibility of reusing nodes, an implicit
memory can develop. As a consequence, GNP targets dynamic environments
and has been applied successfully in fields such as robotics and financial market
analysis.

We extended the Genetic Network Programming by a new mutation operator,
which allows for a variable number of nodes and edges per individual [@kohnke2025variable].
With this operator, the search space is significantly extended, but without the risk of incurring the bloat problem. 
The operator is fitness neutral and has no hyper-parameter. 
Due to higher flexibility, it is now possible for GNP to automatically adapt to the complexity of a given task
and to find suitable features, especially for high dimensional data sets.

Zukünftige Forschung soll nun auch Fraktale Strukturen im Datenraum untersuchen, wofür Fracnetics entwickelt wurde.

# Statement of Need

Fracnetics ist die erste open-source library, die gezielt auf die Genetic Network Programming eine API zur Verfügung stellt. 
Durch die einfache Anwendung auf jegliche individuelle Problemstellungen, lässt sich die Forschung der GNP so erleichtern und intensivieren.
Selbstverständlich kann Fracnetics auch für jegliche industrielle Optimierungen verwendet werden und ist besonders geeignet 
für dynamische Umgebungen wie dem Finanzbereich und der Robotik.

Inkludiert ist auch der neue in [@kohnke2025variable] vorgestellt Mutationsoperator, der die Topologie eines Netzwerks durch 
Wachstum und Schrumpfen verändern kann, um sie an die Komplexität eines gegeben Problems automatisch anzupassen.

Die Fraktale Erweiterung der Netzstrukturen stellt zusätzlich eine Methode zur Verfügung den Datenraum (Features) in 
selbstähnliche Intervalle einzuteilen. This is especially useful for datasets where information exists at multiple resolutions.
Financial markets, e.g., often exhibit fractal-like behavior: price movements at different time scales [@Mandelbrot.2005] or 
environmental interactions in robotics often display recursive or repeating patterns. 

Die angelernten Modelle (Netzwerke) bleiben dabei vollständig interpretierbar und leisten so einen fundamentalen Beitrag zu 
interpretierbarer AI.

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

- **Innovative Opreators**  
    - **Add/Delete Nodes** – dynamic structural changes in networks.
    - **Fractal Geometry Integration** – hierarchical boundary generation via production rules (L-systems-style subdivision).


- TODO Vorgehen/Funktionen teasern

# References 
