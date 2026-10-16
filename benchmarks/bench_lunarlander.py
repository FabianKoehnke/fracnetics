#!/usr/bin/env python
"""Time the first N generations of the LunarLander experiment and check bit-identity.

Runs the unchanged experiment script (dissertation repository) with only the number of
generations patched and a non-interactive matplotlib backend, stops before the post-run
validation, and compares all 8 columns of curves.csv EXACTLY with the reference run.

    ./venv/bin/python -I benchmarks/bench_lunarlander.py 60          # acceptance
    ./venv/bin/python -I benchmarks/bench_lunarlander.py 10          # quick check
    ./venv/bin/python -I benchmarks/bench_lunarlander.py 10 --local  # local fracnetics/

Without --local the installed package (site-packages) is measured -- the one the
experiment sees. Run with -I so the source package in the project root does not shadow it.
"""

import argparse
import collections
import contextlib
import os
import re
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LL_DIR = os.path.join(os.path.dirname(ROOT), "dissertation", "paper",
                      "LunarLander - Interval Boundaries")
SCRIPT = os.path.join(LL_DIR, "code", "lunarlander.py")
REFERENCE = os.path.join(LL_DIR, "runs", "run_20260915_025833", "curves.csv")
LOOP_END = "# Everything below runs on single individuals"
CURVE_NAMES = ["generation", "trainingFitness", "validation", "validationBestSoFar",
               "networkSizeMedian", "dormantShare", "crossoverExchanged", "crossoverSkipped"]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("generations", type=int, nargs="?", default=60)
ap.add_argument("--workers", type=int, default=None, help="override nWorkers of the script")
ap.add_argument("--local", action="store_true", help="import fracnetics from the project root")
ap.add_argument("--script", default=SCRIPT, help="experiment script (default: the dissertation one)")
ap.add_argument("--out", default=None, help="working directory (default: a temp dir)")
ap.add_argument("--save-pop", default=None, help="pickle the final population's individuals here")
args = ap.parse_args()

if args.local:
    sys.path.insert(0, ROOT)
# The script imports its problem module (code/lander.py) from its own folder
sys.path.insert(0, os.path.dirname(os.path.abspath(args.script)))

import matplotlib
matplotlib.use("Agg")
import fracnetics as fn
from fracnetics import parallel_eval

timings = collections.defaultdict(float)
calls = collections.defaultdict(int)


def timed(cls, name, label=None):
    """Wrap cls.name so its wall time is booked under label(self, *args)."""
    orig = getattr(cls, name)

    def wrapper(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            key = label(self, *a) if label else name
            timings[key] += time.perf_counter() - t0
            calls[key] += 1
    setattr(cls, name, wrapper)


for method in ["lexicaseSelection", "crossover", "callAddDelNodes", "callEdgeMutation",
               "callBoundaryMutationNormal", "callBoundaryMutationUniform",
               "callAgeUnusedNodes", "capLineageAfterMutation"]:
    timed(fn.Population, method)
timed(fn.Population, "gymnasium",
      lambda pop, *a: f"evaluate sequential (n={len(pop.individuals)})")
timed(parallel_eval.ParallelEvaluator, "evaluate",
      lambda ev, pop, *a: f"evaluate parallel (n={len(pop.individuals)})")

src = open(args.script).read()
src, nSub = re.subn(r"^generations\s*=\s*\d+", f"generations = {args.generations}", src,
                    count=1, flags=re.M)
if nSub != 1 or LOOP_END not in src:
    sys.exit("experiment script changed: 'generations = ...' or the loop-end marker is missing")
if args.workers:
    src, nSub = re.subn(r"^nWorkers\s*=\s*\d+", f"nWorkers = {args.workers}", src,
                        count=1, flags=re.M)
src = src[:src.index(LOOP_END)]
src = src.replace("for g in range(generations):",
                  "__loopStart = time.perf_counter()\nfor g in range(generations):", 1)
src += ("\n__loopTime = time.perf_counter() - __loopStart\n"
        "if evaluator is not None:\n    evaluator.close()\n")

workDir = os.path.abspath(args.out) if args.out else tempfile.mkdtemp(prefix="bench_ll_")
os.makedirs(workDir, exist_ok=True)
os.chdir(workDir)
ns = {"__name__": "__main__", "__file__": args.script}
with open("stdout.log", "w") as log, contextlib.redirect_stdout(log):
    exec(compile(src, args.script, "exec"), ns)

loop = ns["__loopTime"]
if args.save_pop:
    import pickle
    with open(args.save_pop, "wb") as f:
        pickle.dump(list(ns["pop"].individuals), f)
print(f"fracnetics: {os.path.dirname(fn.__file__)}")
print(f"generations={args.generations} workers={ns.get('nWorkers')} "
      f"loop={loop:.2f}s per generation={loop / args.generations:.3f}s")
booked = 0.0
for key, t in sorted(timings.items(), key=lambda kv: -kv[1]):
    print(f"  {key:36s} {t:8.2f}s {100 * t / loop:6.1f}%  calls={calls[key]}")
    booked += t
print(f"  {'rest (python)':36s} {loop - booked:8.2f}s {100 * (loop - booked) / loop:6.1f}%")

if "history" in ns:  # lander.results.History, script since 2026-10-16
    rows = ns["history"].curveRows()
else:                # earlier scripts kept one list per curve
    curves = [range(len(ns["fitnessProgess"])), ns["fitnessProgess"], ns["validationProgess"],
              ns["validationBestSoFarProgess"], ns["networkSizeMedianProgress"],
              ns["unusedShareProgress"], ns["crossoverExchangedProgress"],
              ns["crossoverSkippedProgress"]]
    rows = [",".join(str(c[i]) for c in curves) for i in range(min(len(c) for c in curves))]
with open("curves.csv", "w") as f:
    f.write(",".join(CURVE_NAMES) + "\n" + "\n".join(rows) + "\n")

reference = open(REFERENCE).read().splitlines()[1:]
if len(reference) < len(rows):
    sys.exit(f"reference has only {len(reference)} generations")
diverged = [i for i, (a, b) in enumerate(zip(rows, reference)) if a != b]
print(f"output: {workDir}")
if diverged:
    i = diverged[0]
    print(f"DIVERGES at generation {i}\n  got {rows[i]}\n  ref {reference[i]}")
    sys.exit(1)
print(f"BIT-IDENTICAL to {os.path.basename(os.path.dirname(REFERENCE))} "
      f"over {len(rows)} generations")
