"""Process-parallel fitness evaluation for a GNP population.

Why processes and not threads: the C++ core steps the Gym environment through a
pybind11 callback, so every step re-acquires the GIL. Threads therefore serialise.
Individuals within a generation are independent -- no shared state until selection --
so they can be split across processes, each holding its own environment.

Only the fields that selection and mutation read are sent back (see COLLECTED below);
Network's own pickle state does not carry them.
"""

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import gymnasium as gym

import fracnetics as fn

# Per-worker state, built once by _initWorker and reused for every chunk.
_env = None
_pop = None
_envKwargs = None


def _initWorker(envKwargs, popKwargs):
    """Build one environment and one container population per worker process."""
    global _env, _pop, _envKwargs
    _envKwargs = envKwargs
    _env = gym.make(**envKwargs)
    # The population is only a container for gymnasiumMultiSeed; its own nodes are
    # replaced by the individuals we send in, so ni=1 is enough.
    _pop = fn.Population(**{**popKwargs, "ni": 1})


def _collect(net):
    """Extract everything the parent needs; Network.__getstate__ omits most of it.

    KNOWN GAP: Network::lastStepRewardsII is written by gymnasiumMultiSeed but is not
    exposed by the pybind11 bindings, so it cannot be carried back. The only reader is
    calculateParetoObjectives(), which indexes it up to fitnessValues.size() -- with an
    empty vector in the parent that is an out-of-bounds read. It is unreachable while
    lunarlander.py keeps multiObjective = False (the whole block is skipped), and the
    value it feeds (meanAbsX) is computed and then discarded. Before enabling
    multiObjective with this evaluator, expose the field in bindings.cpp and add it here.
    """
    return {
        "fitness": net.fitness,
        "fitnessValues": list(net.fitnessValues),
        "lexicaseObjectives": list(net.lexicaseObjectives),
        "objectivesPerSeed": [list(o) for o in net.objectivesPerSeed],
        "lastStepRewards": list(net.lastStepRewards),
        "objectives": list(net.objectives),
        # Per-seed active sub-graph; read by the seedSpecialist crossover. Left empty in
        # the parent it silently disables that path instead of erroring.
        "visitedNodesPerSeed": [list(v) for v in net.visitedNodesPerSeed],
        # Traversal position after the last episode. Not in Network.__getstate__, so the
        # parent keeps whatever it held before -- an uninitialised value on first use.
        "currentNodeID": net.currentNodeID,
        "nConsecutiveP": net.nConsecutiveP,
        # Set when a traversal exceeds dMax without reaching a processing node.
        "invalid": net.invalid,
        "decisions": list(net.decisions),
        # Written by updateLineageStats() inside gymnasiumMultiSeed when
        # useLineageFitness is on. Carried unconditionally so the switch stays free.
        "lineageMean": net.lineageMean,
        "lineageM2": net.lineageM2,
        "lineageN": net.lineageN,
        "lineageSuccessN": net.lineageSuccessN,
        # Node-level traversal state. used drives justUsedNodes in every boundary/edge
        # mutation; lastVisitStep and traverseCounter are read by findSuccessorNodes()
        # to pick sub-networks for crossover (Population.hpp), so leaving them at their
        # pre-evaluation value makes the parent's crossover diverge from the sequential
        # run one generation later.
        "used": [n.used for n in net.innerNodes],
        "traverseCounter": [n.traverseCounter for n in net.innerNodes],
        "lastVisitStep": [n.lastVisitStep for n in net.innerNodes],
    }


def _evalChunk(payload):
    """Evaluate one block of individuals on the shared seed batch."""
    nets = pickle.loads(payload["nets"])
    # Node::traverseCounter and lastVisitStep ACCUMULATE across generations --
    # clearUsedNodes() resets only `used`. Neither survives Node.__getstate__, so a
    # freshly unpickled net would start counting from 0 while the sequential run
    # continues from the carried-over value. The mutation operators read both, so the
    # populations would diverge one generation later. Restore them before evaluating.
    for net, counters, visits, lin in zip(nets, payload["traverseCounter"],
                                          payload["lastVisitStep"], payload["lineage"]):
        nodes = net.innerNodes
        for node, counter, visit in zip(nodes, counters, visits):
            node.traverseCounter = counter
            node.lastVisitStep = visit
        net.innerNodes = nodes
        # Lineage statistics accumulate too (updateLineageStats / capLineageStats) and are
        # likewise absent from the pickle -- sending them back without sending them in
        # would reset the parent's running values to zero on every generation.
        net.lineageMean, net.lineageM2, net.lineageN, net.lineageSuccessN = lin
    _pop.individuals = nets
    _pop.gymnasiumMultiSeed(_env, seeds=payload["seeds"], **payload["kwargs"])
    return payload["index"], [_collect(n) for n in _pop.individuals]


def applyResult(net, result):
    """Write a worker result back into the individual held by the parent process."""
    net.fitness = result["fitness"]
    net.fitnessValues = result["fitnessValues"]
    net.lexicaseObjectives = result["lexicaseObjectives"]
    net.objectivesPerSeed = result["objectivesPerSeed"]
    net.lastStepRewards = result["lastStepRewards"]
    net.objectives = result["objectives"]
    net.visitedNodesPerSeed = result["visitedNodesPerSeed"]
    net.currentNodeID = result["currentNodeID"]
    net.nConsecutiveP = result["nConsecutiveP"]
    net.invalid = result["invalid"]
    net.decisions = result["decisions"]
    net.lineageMean = result["lineageMean"]
    net.lineageM2 = result["lineageM2"]
    net.lineageN = result["lineageN"]
    net.lineageSuccessN = result["lineageSuccessN"]
    # innerNodes returns a copy (std::vector<Node> is not opaque), so the whole list
    # has to be written back after patching the traversal flags.
    nodes = net.innerNodes
    for node, used, counter, visit in zip(nodes, result["used"],
                                          result["traverseCounter"],
                                          result["lastVisitStep"]):
        node.used = used
        node.traverseCounter = counter
        node.lastVisitStep = visit
    net.innerNodes = nodes


class ParallelEvaluator:
    """Drop-in replacement for pop.gymnasiumMultiSeed(env, ...) across processes.

    Usage:
        ev = ParallelEvaluator(envKwargs, popKwargs, workers=8)
        ev.evaluate(pop, seeds=seeds, maxSteps=1000, ...)
        ...
        ev.close()
    """

    def __init__(self, envKwargs, popKwargs, workers=None, chunksPerWorker=3,
                 startMethod="fork"):
        self.workers = workers or os.cpu_count() or 1
        self.chunksPerWorker = chunksPerWorker
        # fork, not spawn: spawn re-imports the caller's module in every worker, which
        # would re-run lunarlander.py's 2400 lines of top-level code (it has no
        # __main__ guard) and needs this file on the worker's sys.path. fork inherits
        # both, costs no interpreter start-up, and is safe here because the parent is
        # single-threaded -- Box2D and gymnasium spawn no threads of their own.
        self.pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=get_context(startMethod),
            initializer=_initWorker,
            initargs=(envKwargs, popKwargs),
        )

    def evaluate(self, pop, seeds, **kwargs):
        """Evaluate pop.individuals on seeds, writing results back in place."""
        individuals = list(pop.individuals)
        n = len(individuals)
        if n == 0:
            return

        # Episode cost varies by an order of magnitude (early crash vs. full flight),
        # so hand out more chunks than workers and let the pool balance them.
        nChunks = min(n, max(self.workers, self.workers * self.chunksPerWorker))
        size = (n + nChunks - 1) // nChunks
        payloads = []
        for start in range(0, n, size):
            block = individuals[start:start + size]
            payloads.append({
                "index": start,
                "nets": pickle.dumps(block, protocol=pickle.HIGHEST_PROTOCOL),
                # Carried separately because Node.__getstate__ drops them (see _evalChunk)
                "traverseCounter": [[n.traverseCounter for n in net.innerNodes] for net in block],
                "lastVisitStep": [[n.lastVisitStep for n in net.innerNodes] for net in block],
                "lineage": [(net.lineageMean, net.lineageM2, net.lineageN, net.lineageSuccessN)
                            for net in block],
                "seeds": list(seeds),
                "kwargs": kwargs,
            })

        for start, results in self.pool.map(_evalChunk, payloads):
            for offset, result in enumerate(results):
                applyResult(individuals[start + offset], result)

        # individuals[] yields references into the opaque NetworkVector, so the writes
        # above already landed in the population -- nothing to reassign.

    def close(self):
        self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
