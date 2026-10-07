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
| `include/*.hpp` | **The entire core, header-only.** `Population.hpp` (selection, crossover, mutation, evaluation), `Network.hpp` (graph, traversal, clustering), `Node.hpp` (nodes, edges, boundaries), `Fractal.hpp`, `GymnasiumWrapper.hpp` |
| `bindings/bindings.cpp` | pybind11 bindings — the **only** translation unit |
| `fracnetics/` | Python package. `__init__.py` is versioned, `_core*.so` is built |
| `tests/` | `*.cpp` GoogleTest, `test_*.py` pytest |
| `examples/lunarlander.py` | Main experiment; the notebook variant is often older |

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

## Conventions

- C++20, header-only. Doxygen blocks on public methods.
