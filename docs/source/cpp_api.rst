C++ API Reference
=================

The C++ core implements the data structures and algorithms that power **Fracnetics**.  
All classes are defined in the `include/` directory.

.. contents::
   :local:
   :depth: 1

Population
----------

.. doxygenclass:: Population
   :project: Fracnetics
   :members:
   :protected-members:
   :undoc-members:

Network
-------

.. doxygenclass:: Network
   :project: Fracnetics
   :members:
   :protected-members:
   :undoc-members:

Node
----

.. doxygenclass:: Node
   :project: Fracnetics
   :members:
   :protected-members:
   :undoc-members:

Fractal
-------

.. doxygenfile:: Fractal.hpp
   :project: Fracnetics
   :path: ../include/Fractal.hpp


Gymnasium evaluator
-------------------

The only part that talks to Python: runs episodes of any Gymnasium environment and writes
the results into the networks (``Population.gymnasium()`` in Python).

.. doxygenfile:: GymnasiumWrapper.hpp
   :project: Fracnetics
   :path: ../include/GymnasiumWrapper.hpp
