"""Observed value range of judgment nodes and the data-guided boundary mutations.

A node spanning a feature's declared range sees only a small part of it in practice, so
most of its boundaries never separate two values that occur. The evaluation records per
node the range of the values it judged (seenMin, seenMax); with dataGuided=True the
uniform mutation redraws a boundary only inside that range, and the normal mutation scales
its step with the range instead of the gap.
"""

import gymnasium as gym
import pytest

import fracnetics as fn

POPKW = dict(seed=11, jn=8, jnf=8, pn=4, pnf=4, fractalJudgment=False, useExperience=False,
             nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
MINF = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAXF = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
EVAL = dict(dMax=10, maxSteps=300, maxConsecutiveP=5)
SEEDS_A, SEEDS_B = [3, 17, 29], [101, 202]
INF = float("inf")


class Record(gym.Wrapper):
    """Keeps every observation, so tests know which values could have been judged."""

    def __init__(self, env):
        super().__init__(env)
        self.observations = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.observations.append(obs)
        return obs, info

    def step(self, action):
        obs, *rest = self.env.step(action)
        self.observations.append(obs)
        return (obs, *rest)


def makeEnv():
    return gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False)


def evaluated(seeds=SEEDS_A, ni=20):
    pop = fn.Population(ni=ni, **POPKW)
    pop.setAllNodeBoundaries(MINF, MAXF)
    env = Record(makeEnv())
    pop.gymnasium(env, seeds=seeds, **EVAL)
    return pop, env.observations


def judgmentNodes(pop):
    for net in pop.individuals:
        for node in net.innerNodes:
            if node.type == "J":
                yield node


def test_evaluation_records_the_range_of_judged_values():
    pop, observations = evaluated()
    seen = 0
    for node in judgmentNodes(pop):
        if node.seenMin == INF:
            assert node.seenMax == -INF       # never judged in this evaluation
            continue
        seen += 1
        values = [float(o[node.f]) for o in observations]
        assert node.used
        assert min(values) <= node.seenMin <= node.seenMax <= max(values)
    assert seen > 0


def test_a_new_evaluation_starts_a_new_range():
    twice, _ = evaluated(SEEDS_A)
    twice.gymnasium(makeEnv(), seeds=SEEDS_B, **EVAL)
    once, _ = evaluated(SEEDS_B)
    ranges = lambda pop: [(n.seenMin, n.seenMax) for n in judgmentNodes(pop)]
    assert ranges(twice) == ranges(once)


def test_data_guided_uniform_redraws_only_inside_the_observed_range():
    pop, _ = evaluated()
    before = [list(n.boundaries) for n in judgmentNodes(pop)]
    pop.checkOrder = False
    pop.callBoundaryMutationUniform(probability=1.0, justUsedNodes=True, dataGuided=True)
    moved = 0
    for node, old in zip(judgmentNodes(pop), before):
        new = list(node.boundaries)
        assert new == sorted(new)
        for i in range(1, len(new) - 1):
            if new[i] != old[i]:
                moved += 1
                assert node.used and node.seenMin <= new[i] <= node.seenMax
    assert moved > 0


def withRange(pop, lo, hi):
    """Set every judgment node's observed range by hand."""
    individuals = list(pop.individuals)
    for net in individuals:
        nodes = net.innerNodes
        for node in nodes:
            node.seenMin, node.seenMax = lo, hi
        net.innerNodes = nodes
    pop.individuals = individuals


@pytest.mark.parametrize("dataGuided", [False, True])
def test_data_guided_normal_steps_scale_with_the_observed_range(dataGuided):
    pop = fn.Population(ni=20, **POPKW)
    pop.setAllNodeBoundaries(MINF, MAXF)
    pop.checkOrder = False
    lo, hi = 0.0, 1e-3                         # a tiny range: steps of sigma * 1e-3
    withRange(pop, lo, hi)
    before = [list(n.boundaries) for n in judgmentNodes(pop)]
    pop.callBoundaryMutationNormal(probability=1.0, sigma=0.2, justUsedNodes=False, dataGuided=dataGuided)
    steps = [abs(b - a) for node, old in zip(judgmentNodes(pop), before)
             for a, b in zip(old[1:-1], list(node.boundaries)[1:-1]) if a != b]
    assert steps
    if dataGuided:
        assert max(steps) < 6 * 0.2 * (hi - lo)
    else:                                       # gap-relative: far larger than the range
        assert sum(s > hi for s in steps) > 0.9 * len(steps)


@pytest.mark.parametrize("call", ["normal", "uniform"])
def test_default_is_the_plain_mutation(call):
    def mutate(**extra):
        pop, _ = evaluated()
        pop.checkOrder = False
        if call == "normal":
            pop.callBoundaryMutationNormal(probability=0.5, sigma=0.2, justUsedNodes=True, **extra)
        else:
            pop.callBoundaryMutationUniform(probability=0.5, justUsedNodes=True, **extra)
        return [list(n.boundaries) for n in judgmentNodes(pop)]
    assert mutate() == mutate(dataGuided=False)


# ── Lifetime range (everSeen) ────────────────────────────────────────────────

def test_lifetime_range_accumulates_over_evaluations():
    pop, _ = evaluated(SEEDS_A)
    first = [(n.everSeenMin, n.everSeenMax, n.seenMin, n.seenMax) for n in judgmentNodes(pop)]
    pop.gymnasium(makeEnv(), seeds=SEEDS_B, **EVAL)
    for (lo0, hi0, slo0, shi0), n in zip(first, judgmentNodes(pop)):
        assert (lo0, hi0) == (slo0, shi0)                 # after one evaluation both agree
        assert n.everSeenMin == min(lo0, n.seenMin)       # then the lifetime range is the union
        assert n.everSeenMax == max(hi0, n.seenMax)


def test_parallel_evaluation_keeps_the_lifetime_range_like_the_sequential_one():
    from fracnetics.parallel_eval import ParallelEvaluator
    seq, _ = evaluated(SEEDS_A, ni=6)
    seq.gymnasium(makeEnv(), seeds=SEEDS_B, **EVAL)
    par = fn.Population(ni=6, **POPKW)
    par.setAllNodeBoundaries(MINF, MAXF)
    with ParallelEvaluator(makeEnv, POPKW, workers=2) as ev:
        ev.evaluate(par, seeds=SEEDS_A, **EVAL)
        ev.evaluate(par, seeds=SEEDS_B, **EVAL)
    ranges = lambda pop: [(n.everSeenMin, n.everSeenMax, n.seenMin, n.seenMax) for n in judgmentNodes(pop)]
    assert ranges(par) == ranges(seq)


def test_lifetime_range_is_used_with_ever_seen():
    pop = fn.Population(ni=20, **POPKW)
    pop.setAllNodeBoundaries(MINF, MAXF)
    pop.checkOrder = False
    individuals = list(pop.individuals)
    for net in individuals:                     # last evaluation: tiny range; lifetime: wide range
        nodes = net.innerNodes
        for node in nodes:
            node.seenMin, node.seenMax = 0.0, 1e-3
            node.everSeenMin, node.everSeenMax = -1.0, 1.0
        net.innerNodes = nodes
    pop.individuals = individuals
    before = [list(n.boundaries) for n in judgmentNodes(pop)]
    pop.callBoundaryMutationUniform(probability=1.0, justUsedNodes=False, dataGuided=True, everSeen=True)
    moved = [b for node, old in zip(judgmentNodes(pop), before)
             for a, b in zip(old[1:-1], list(node.boundaries)[1:-1]) if a != b]
    assert moved and all(-1.0 <= b <= 1.0 for b in moved)
    assert sum(b > 1e-3 or b < 0.0 for b in moved) > 0.5 * len(moved)   # beyond the last evaluation's range
