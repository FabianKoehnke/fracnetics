"""Process-parallel fitness evaluation for a GNP population.

Why processes and not threads: the C++ core steps the Gym environment through a
pybind11 callback, so every step re-acquires the GIL. Threads therefore serialise.
Individuals within a generation are independent -- no shared state until selection --
so they can be split across processes, each holding its own environment.

Only the fields that selection and mutation read are sent back (see _collect below);
Network's own pickle state does not carry them.

Small populations (an elite re-scoring, a one-individual validation) are split by seeds
as well, so every worker gets work. Episodes are independent given the network, so the seed
blocks of one individual are stitched back together exactly (see _mergeBlocks).
"""

import gc
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import fracnetics as fn

# Per-worker state, built once by _initWorker and reused for every task.
_env = None
_pop = None
_episodeObjectives = None


def _initWorker(makeEnv, popKwargs, episodeObjectives):
    """Build one environment and one container population per worker process."""
    global _env, _pop, _episodeObjectives
    _env = makeEnv()
    # The population is only a container for gymnasium(); its own nodes are
    # replaced by the individuals we send in, so ni=1 is enough.
    _pop = fn.Population(**{**popKwargs, "ni": 1})
    _episodeObjectives = episodeObjectives
    # A forked worker inherits the parent's whole heap (matplotlib, pandas, ...). Left in
    # the GC's generations it is traversed over and over; freezing it made small tasks
    # 3-4x faster in the LunarLander experiment. Results are unaffected.
    gc.freeze()


def _collect(net):
    """Extract everything the parent needs; Network.__getstate__ omits most of it."""
    return {
        "fitness": net.fitness,
        "caseScores": list(net.caseScores),
        "lexicaseObjectives": list(net.lexicaseObjectives),
        "objectivesPerCase": [list(o) for o in net.objectivesPerCase],
        "objectives": list(net.objectives),
        # Per-case active sub-graph; read by the caseSpecialist crossover. Left empty in
        # the parent it silently disables that path instead of erroring.
        "visitedNodesPerCase": [list(v) for v in net.visitedNodesPerCase],
        # Traversal position after the last episode. Not in Network.__getstate__, so the
        # parent keeps whatever it held before -- an uninitialised value on first use.
        "currentNodeID": net.currentNodeID,
        "nConsecutiveP": net.nConsecutiveP,
        # Set when a traversal exceeds dMax without reaching a processing node.
        "invalid": net.invalid,
        "decisions": list(net.decisions),
        # Written by updateLineageStats() when trackLineage is on. Carried
        # unconditionally so the switch stays free.
        "lineageMean": net.lineageMean,
        "lineageM2": net.lineageM2,
        "lineageN": net.lineageN,
        "lineageSuccessRate": net.lineageSuccessRate,
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


def _evalTask(payload):
    """Evaluate a few individuals on one block of the seed batch."""
    nets = [pickle.loads(b) for b in payload["nets"]]
    # traverseCounter and lastVisitStep need not be sent in: the first seed of every
    # evaluation resets them. Lineage statistics accumulate across generations and are
    # absent from the pickle -- sending them back without sending them in would reset
    # the parent's running values to zero on every generation.
    for net, lin in zip(nets, payload["lineage"]):
        (net.lineageMean, net.lineageM2, net.lineageN,
         net.lineageSuccessRate, net.lineageSuccessN) = lin
    _pop.individuals = nets
    _pop.gymnasium(_env, seeds=payload["seeds"], resetOptions=payload["resetOptions"],
                   episodeObjectives=_episodeObjectives, **payload["kwargs"])
    return payload["task"], [_collect(n) for n in _pop.individuals]


def _mergeBlocks(blocks):
    """Stitch the results of consecutive seed blocks of one individual together.

    Every block started with a fresh traversal (newRun on its first seed), so per node the
    visit counts add up, `used` is an OR, and lastVisitStep -- a timestamp of the
    network-wide step counter -- is shifted by the counter's advance in all earlier blocks.
    Every advance enters exactly one node, so a block's advance is the sum of its node
    counters.
    fitness and lexicaseObjectives are recomputed by Network.aggregateCaseResults().
    """
    merged = dict(blocks[-1])  # state after the last episode: currentNodeID, invalid, ...
    for key in ("caseScores", "objectivesPerCase", "visitedNodesPerCase"):
        merged[key] = [value for block in blocks for value in block[key]]
    nNodes = len(blocks[0]["used"])
    used, counters, visits = [False] * nNodes, [0] * nNodes, [0] * nNodes
    offset = 0
    for block in blocks:
        for i in range(nNodes):
            used[i] = used[i] or block["used"][i]
            if block["traverseCounter"][i] > 0:
                counters[i] += block["traverseCounter"][i]
                visits[i] = offset + block["lastVisitStep"][i]
        offset += sum(block["traverseCounter"])
    merged["used"], merged["traverseCounter"], merged["lastVisitStep"] = used, counters, visits
    return merged


def applyResult(net, result):
    """Write a worker result back into the individual held by the parent process."""
    net.fitness = result["fitness"]
    net.caseScores = result["caseScores"]
    net.lexicaseObjectives = result["lexicaseObjectives"]
    net.objectivesPerCase = result["objectivesPerCase"]
    net.objectives = result["objectives"]
    net.visitedNodesPerCase = result["visitedNodesPerCase"]
    net.currentNodeID = result["currentNodeID"]
    net.nConsecutiveP = result["nConsecutiveP"]
    net.invalid = result["invalid"]
    net.decisions = result["decisions"]
    net.lineageMean = result["lineageMean"]
    net.lineageM2 = result["lineageM2"]
    net.lineageN = result["lineageN"]
    net.lineageSuccessRate = result["lineageSuccessRate"]
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
    """Drop-in replacement for pop.gymnasium(env, ...) across processes.

    Usage:
        ev = ParallelEvaluator(makeEnv, popKwargs, workers=8, episodeObjectives=myObjectives)
        ev.evaluate(pop, seeds=seeds, dMax=10, maxSteps=1000, maxConsecutiveP=5, ...)
        ...
        ev.close()

    Args:
        env: Zero-argument callable that builds the environment (wrappers included), or a
            dict of gymnasium.make() keyword arguments. Called once per worker.
        popKwargs: Population constructor arguments; the workers build a container from
            them (ni is overridden).
        workers: Number of processes (default: all cores).
        chunksPerWorker: Tasks per worker, so the pool can balance episodes of very
            different length.
        startMethod: multiprocessing start method, see __init__.
        episodeObjectives: The episodeObjectives hook of Population.gymnasium(). Set here
            rather than per evaluate() call: the workers inherit it when they are forked, so
            it needs no pickling (a function defined in an exec'd script would not pickle).
    """

    def __init__(self, env, popKwargs, workers=None, chunksPerWorker=3,
                 startMethod="fork", episodeObjectives=None):
        self.workers = workers or os.cpu_count() or 1
        self.chunksPerWorker = chunksPerWorker
        if callable(env):
            makeEnv = env
        else:
            import gymnasium as gym
            makeEnv = lambda: gym.make(**env)
        # fork, not spawn: spawn re-imports the caller's module in every worker, which
        # would re-run an experiment script's top-level code (a script without a __main__
        # guard). fork inherits the parent state, costs no interpreter start-up, and is
        # safe as long as the parent is single-threaded -- Box2D and gymnasium spawn no
        # threads of their own.
        self.pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=get_context(startMethod),
            initializer=_initWorker,
            initargs=(makeEnv, popKwargs, episodeObjectives),
        )

    def _tasks(self, n, nSeeds, splitSeeds):
        """(individual indices, first seed, end seed) per task, about chunksPerWorker per worker.

        Episode cost varies by an order of magnitude (early failure vs. full episode), so
        there are more tasks than workers and the pool balances them. With fewer
        individuals than tasks, each individual's seeds are cut into blocks.
        """
        target = max(1, self.workers * self.chunksPerWorker)
        if n >= target or not splitSeeds or nSeeds <= 1:
            size = (n + min(n, target) - 1) // min(n, target)
            return [(list(range(start, min(start + size, n))), 0, nSeeds)
                    for start in range(0, n, size)]
        nBlocks = min(nSeeds, (target + n - 1) // n)
        bounds = [(b * nSeeds // nBlocks, (b + 1) * nSeeds // nBlocks) for b in range(nBlocks)]
        return [([i], lo, hi) for i in range(n) for lo, hi in bounds]

    def evaluate(self, pop, seeds, resetOptions=None, **kwargs):
        """Evaluate pop.individuals on seeds, writing results back in place.

        Takes the arguments of Population.gymnasium() except env and episodeObjectives.
        """
        if "episodeObjectives" in kwargs:
            raise TypeError("episodeObjectives is set on the ParallelEvaluator itself "
                            "(the workers inherit it when they are forked)")
        individuals = list(pop.individuals)
        n = len(individuals)
        if n == 0:
            pop.markEvaluated()
            return
        seeds = list(seeds)
        if resetOptions is not None:
            resetOptions = list(resetOptions)
            if len(resetOptions) != len(seeds):
                raise ValueError("resetOptions needs one entry per seed")
        # Lineage statistics are a running mean over the seeds in order; a split would
        # have to replay them, so lineage runs keep all seeds of an individual together.
        splitSeeds = not kwargs.get("trackLineage", False)

        netBytes = [pickle.dumps(net, protocol=pickle.HIGHEST_PROTOCOL) for net in individuals]
        # Carried separately because Network.__getstate__ drops them (see _evalTask)
        lineage = [(net.lineageMean, net.lineageM2, net.lineageN,
                    net.lineageSuccessRate, net.lineageSuccessN) for net in individuals]

        payloads = []
        for indices, lo, hi in self._tasks(n, len(seeds), splitSeeds):
            payloads.append({
                "task": (indices, lo),
                "nets": [netBytes[i] for i in indices],
                "lineage": [lineage[i] for i in indices],
                "seeds": seeds[lo:hi],
                "resetOptions": None if resetOptions is None else resetOptions[lo:hi],
                "kwargs": kwargs,
            })

        blocks = [[] for _ in range(n)]
        for (indices, lo), results in self.pool.map(_evalTask, payloads):
            for i, result in zip(indices, results):
                blocks[i].append((lo, result))

        for net, netBlocks in zip(individuals, blocks):
            netBlocks.sort(key=lambda block: block[0])
            if len(netBlocks) == 1:
                applyResult(net, netBlocks[0][1])
            else:
                applyResult(net, _mergeBlocks([result for _, result in netBlocks]))
                net.aggregateCaseResults()

        # individuals[] yields references into the opaque NetworkVector, so the writes
        # above already landed in the population -- nothing to reassign.
        # The workers evaluated copies, so pop's order check has to be told (Population::Step)
        pop.markEvaluated()

    def close(self):
        self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
