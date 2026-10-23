# fracnetics (GNPfrac)

C++ library for **Genetic Network Programming (GNP)** with variable network size and fractal
expansion, exposed to Python via pybind11. Evolves directed decision graphs for control tasks
(LunarLander, CartPole) and classification.

## Language

**Write code, comments and docs in English.** Identifiers, inline comments and Doxygen blocks
alike. Chat with the maintainer stays German.

**Keep comments short** — one to three lines stating *why*, not *what*. 

## Layout

| Path | Contents |
|---|---|
| `include/*.hpp` | **The entire core, header-only.** `Population.hpp` (selection, crossover, mutation), `Network.hpp` (graph, traversal, `act()`/`recordStep()` for episodes), `Node.hpp` (nodes, edges, boundaries), `Fractal.hpp` -- plain C++, no Python. `GymnasiumWrapper.hpp` is the only pybind11 part: the generic episode loop behind `Population.gymnasium()` |
| `bindings/bindings.cpp` | pybind11 bindings — the **only** translation unit |
| `fracnetics/` | Python package. `__init__.py`, `utilsModel.py` and `parallel_eval.py` (process-parallel evaluation) are versioned, `_core*.so` is built |
| `tests/` | `*.cpp` GoogleTest, `test_*.py` pytest; `Data.hpp`/`PrintHelper.hpp` only serve the `iris.cpp` example |
| `examples/` | Notebooks (CartPole, IRIS, MountainCar) |
| `benchmarks/` | `bench_lunarlander.py`: times the first N generations of the LunarLander script and checks them bit for bit against the reference run |

This repository is the library only, and it is **problem-independent**: nothing in it may
know about a particular environment (observation layout, action meaning, reward scale,
success thresholds). Evaluation produces one score per case (`Network::caseScores`; an
episode per seed, a sample per row); problem-specific criteria enter through the
`episodeObjectives` hook, reward shaping and start-state changes through a
`gymnasium.Wrapper`, per-episode settings through `resetOptions`.

The LunarLander experiment, the studies, the runs and the reports live in the dissertation
repository under `../dissertation/paper/LunarLander - Interval Boundaries/` and import the
installed `fracnetics`. Everything LunarLander-specific (curriculum, survival/potential reward
modes, the five lexicase objectives, landing thresholds, hold action 4) is in its
`code/lander.py`. Library changes must keep its `code/lunarlander.py` bit-identical to
`runs/run_20260915_025833` (first 60 generations, all 8 columns of `curves.csv`) with
`seedCount = 10`, which the benchmark sets.

Everything is header-only, so **any** header change forces a full recompile of
`bindings.cpp`. There are no incremental partial builds.

## Building — mandatory

```bash
source venv/bin/activate
pip install .
```

That is the complete procedure. No further steps are needed.

**Rules to follow:**

- **Do not create a new build directory.** Use `pip install .` only. `build/` already exists
  (CMake, `BUILD_TESTS=ON`) for the C++ tests — that is enough.
- **Never delete `fracnetics/_core*.so`.** The `fracnetics/` directory is the source package
  and shadows the site-packages install whenever Python starts from the project root. Without
  the local `.so` the import fails with
  `ModuleNotFoundError: No module named 'fracnetics._core'`. `pip install .` recreates it
  automatically, byte-identical to the installed copy.
- **Without an activated venv, always use `./venv/bin/pip` and `./venv/bin/python`.** The PATH
  also contains Anaconda (3.8.5) and a framework Python 3.11. Only the venv has gymnasium 1.2.0
  with `LunarLander-v3`; the framework Python knows `v2` only.
- **Do not verify via timestamps.** pip takes mtimes from the wheel metadata, so a freshly
  installed `.so` can carry an old date. Use:
  ```bash
  shasum -a 256 fracnetics/_core*.so venv/lib/python3.11/site-packages/fracnetics/_core*.so
  ```
  Both hashes must match. If they diverge (typically after a CMake build, which writes only the
  local copy), run `pip install .` again — do not delete either copy.

## Testing

```bash
./venv/bin/python -m pytest tests/ -q     # Python tests (same as CI)
./build/runTests                          # C++ tests (GoogleTest)
```

The C++ tests are commented out in CI but run locally from the existing `build/`. After header
changes: `cmake --build build`.

Performance changes are checked on a short LunarLander run, not the full 2000 generations:

```bash
./venv/bin/python -I benchmarks/bench_lunarlander.py 60           # installed package
./venv/bin/python -I benchmarks/bench_lunarlander.py 10 --local   # local build, quick check
```

It must print `BIT-IDENTICAL`. Timings on this laptop are noisy (background load), so compare
variants in alternating runs.

## Conventions

- C++20, header-only. Doxygen blocks on public methods.
- `Population` checks the order of the steps of a generation (`Population::Step`, `checkOrder`).
  A new operator gets a `checkStep()` call; an evaluation outside the class calls
  `markEvaluated()`; tests that call single operators set `checkOrder = False`.
