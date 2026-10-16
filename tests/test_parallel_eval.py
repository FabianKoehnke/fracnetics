"""Process-parallel evaluation must be indistinguishable from the sequential path.

This guards the field list in parallel_eval._collect/applyResult: Network's pickle state
does not carry the case scores, the objective vectors or the traversal flags, so
each has to be copied back explicitly. A new field that selection or mutation reads must
be added there -- this test fails if one is missing.

Compare EXACTLY, never rounded. The bug that once made a full run diverge (stale
accumulated state in the parent) left every fitness value bit-identical and changed only
which nodes callAddDelNodes() grew -- rounded comparisons and fitness-only comparisons
both missed it. test_growth_path_matches therefore runs addDelNodes as well, so the
growth path is covered and not just the evaluation results.
"""

import random

import pytest

gym = pytest.importorskip("gymnasium")
import fracnetics as fn
from fracnetics import parallel_eval
from fracnetics.parallel_eval import ParallelEvaluator

ENV = dict(id="LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False)
POPKW = dict(seed=5, jn=8, jnf=8, pn=4, pnf=4, fractalJudgment=False,
             useExperience=False, nFeatureValues=[3, 3, 3, 3, 3, 3, 2, 2])
MINF = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAXF = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
EVAL = dict(dMax=10, maxSteps=400, maxConsecutiveP=5)
NI, NSEEDS = 16, 4


def objectives(episode):
    """Return, final |x| and step count -- objectives on different scales."""
    return [episode.episodeReturn, -abs(float(episode.observation[0])), float(episode.steps)]


class ScaledReward(gym.Wrapper):
    """Scales the reward by reset(options={"scale": ...}), so per-seed options matter."""

    def reset(self, *, seed=None, options=None):
        self.scale = (options or {}).get("scale", 1.0)
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info


def makeEnv():
    return ScaledReward(gym.make(**ENV))


def _build():
    pop = fn.Population(ni=NI, **POPKW)
    pop.setAllNodeBoundaries(MINF, MAXF)
    return pop


def _snapshot(pop):
    out = []
    for i in range(len(pop.individuals)):
        n = pop.individuals[i]
        out.append((
            pytest.approx(n.fitness),
            [pytest.approx(v) for v in n.caseScores],
            [pytest.approx(v) for v in n.lexicaseObjectives],
            [[pytest.approx(v) for v in o] for o in n.objectivesPerCase],
            [nd.used for nd in n.innerNodes],
        ))
    return out


@pytest.fixture(scope="module")
def seeds():
    return random.Random(7).sample(range(100000), NSEEDS)


def test_parallel_matches_sequential(seeds):
    seqPop = _build()
    seqPop.gymnasium(makeEnv(), seeds=seeds, episodeObjectives=objectives, **EVAL)
    expected = _snapshot(seqPop)

    parPop = _build()
    with ParallelEvaluator(makeEnv, POPKW, workers=2, episodeObjectives=objectives) as ev:
        ev.evaluate(parPop, seeds=seeds, **EVAL)

    assert _snapshot(parPop) == expected


def test_growth_path_matches(seeds):
    """Growth depends on accumulated per-node state, so sizes must match too."""
    def run(parallel):
        pop = _build()
        env = makeEnv()
        ev = ParallelEvaluator(ENV, POPKW, workers=2) if parallel else None
        sizes = []
        for g in range(3):
            if ev:
                ev.evaluate(pop, seeds=seeds, **EVAL)
            else:
                pop.gymnasium(env, seeds=seeds, **EVAL)
            pop.lexicaseSelection(E=2, type="casesStandardized", epsilon=0.1)
            pop.callAddDelNodes(MINF, MAXF, junk=0.1, noElite=True, currentGeneration=g)
            sizes.append([len(pop.individuals[i].innerNodes) for i in range(NI)])
        if ev:
            ev.close()
        return sizes

    assert run(True) == run(False)


def test_parallel_survives_growth_and_selection(seeds):
    """Three generations of select/mutate on top, so grown nodes and used flags count."""
    def run(parallel):
        pop = _build()
        env = makeEnv()
        ev = ParallelEvaluator(ENV, POPKW, workers=2) if parallel else None
        rng = random.Random(11)
        trace = []
        for g in range(3):
            batch = rng.sample(range(100000), NSEEDS)
            if ev:
                ev.evaluate(pop, seeds=batch, **EVAL)
            else:
                pop.gymnasium(env, seeds=batch, **EVAL)
            trace.append((
                [pytest.approx(pop.individuals[i].fitness) for i in range(NI)],
                [len(pop.individuals[i].innerNodes) for i in range(NI)],
            ))
            pop.lexicaseSelection(E=2, type="casesStandardized", epsilon=0.1)
            pop.callAddDelNodes(MINF, MAXF, junk=0.1, noElite=True, currentGeneration=g)
            pop.callEdgeMutation(probInnerNodes=0, probStartNode=0, justUsedNodes=True, k=2.0)
            pop.callBoundaryMutationNormal(probability=0, sigma=0.2, justUsedNodes=True, k=1.0)
        if ev:
            ev.close()
        return trace

    assert run(True) == run(False)


def _exact(pop):
    """Everything the evaluator writes back, compared bit for bit."""
    return [parallel_eval._collect(pop.individuals[i]) for i in range(len(pop.individuals))]


@pytest.mark.parametrize("ni", [1, 3])
@pytest.mark.parametrize("extra", [
    {},
    {"resetOptions": [{"scale": s} for s in (0.5, 1.0, 2.0, 3.0)]},
    {"trackLineage": True, "successThreshold": -50.0},
], ids=["plain", "resetOptions", "lineage"])
def test_seed_split_matches_sequential(seeds, ni, extra):
    """Fewer individuals than tasks: seeds are cut into blocks and stitched back together
    (lineage runs keep the seeds of an individual together instead).

    Runs two evaluations, so the second one starts from carried-over node counters --
    the case where the first seed's visited-node snapshot depends on them.
    """
    kwargs = {**EVAL, **extra}

    def build():
        pop = fn.Population(ni=ni, **POPKW)
        pop.setAllNodeBoundaries(MINF, MAXF)
        return pop

    seqPop, parPop = build(), build()
    env = makeEnv()
    with ParallelEvaluator(makeEnv, POPKW, workers=2, episodeObjectives=objectives) as ev:
        assert all(len(t[0]) == 1 for t in ev._tasks(ni, len(seeds), True))
        for batch in (seeds, seeds[::-1]):
            seqPop.gymnasium(env, seeds=batch, episodeObjectives=objectives, **kwargs)
            ev.evaluate(parPop, seeds=batch, **kwargs)
            assert _exact(parPop) == _exact(seqPop)


def test_objectives_hook_belongs_to_the_evaluator(seeds):
    pop = _build()
    with ParallelEvaluator(makeEnv, POPKW, workers=1) as ev:
        with pytest.raises(TypeError):
            ev.evaluate(pop, seeds=seeds, episodeObjectives=objectives, **EVAL)
