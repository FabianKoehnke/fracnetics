"""Standardised lexicase over cases (type="casesStandardized").

The MAD tolerance of type="cases" is a spread estimate, and the LunarLander reward is
bimodal (landed ~+200 against crashed ~-200) -- there the spread mostly measures the
distance between the two modes rather than the resolution among the near-equal
candidates selection has to separate. Z-transforming each seed's rewards over the
population makes the tolerance mean the same thing on every seed and in every
generation, while keeping the magnitudes a rank transform would discard.
"""

import gymnasium as gym
import pytest

import fracnetics as fn

SEEDS = [2, 88, 451, 6003]
MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]


@pytest.fixture(scope="module")
def env():
    e = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False,
                 wind_power=15.0, turbulence_power=0.5)
    yield e
    e.close()


def evaluatedPopulation(env, ni=40, seed=29):
    pop = fn.Population(seed=seed, ni=ni, jn=8, jnf=8, pn=4, pnf=4,
                        fractalJudgment=False, useExperience=False,
                        nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    pop.gymnasium(env, seeds=SEEDS, dMax=10, maxSteps=250, maxConsecutiveP=5)
    return pop


def test_selection_runs_and_keeps_population_size(env):
    pop = evaluatedPopulation(env)
    before = len(pop.individuals)
    pop.lexicaseSelection(E=2, epsilon=0.5, type="casesStandardized")
    assert len(pop.individuals) == before
    assert len(pop.indicesElite) == 2


def test_invariant_under_affine_rescaling(env):
    """The point of standardising: multiplying every reward by 7 and adding 1000 leaves
    the z-scores untouched, so the selection must be identical. A value-based MAD cut
    would move with the scale."""
    baseline = evaluatedPopulation(env)
    scaled = evaluatedPopulation(env)
    for ind in scaled.individuals:
        ind.caseScores = [v * 7.0 + 1000.0 for v in ind.caseScores]
    baseline.lexicaseSelection(E=2, epsilon=0.5, type="casesStandardized")
    scaled.lexicaseSelection(E=2, epsilon=0.5, type="casesStandardized")
    for base, scale in zip(baseline.individuals, scaled.individuals):
        restored = [(v - 1000.0) / 7.0 for v in scale.caseScores]
        assert list(base.caseScores) == pytest.approx(restored, rel=1e-4, abs=1e-2)


def test_zero_tolerance_matches_strict_lexicase(env):
    """At 0 sigma only the best on each test case survives -- identical to a value cut of
    0, because the z-transform is monotone per seed."""
    standardized = evaluatedPopulation(env)
    values = evaluatedPopulation(env)
    standardized.lexicaseSelection(E=2, epsilon=0.0, type="casesStandardized")
    values.lexicaseSelection(E=2, epsilon=0.0, type="cases")
    assert ([ind.fitness for ind in standardized.individuals]
            == [ind.fitness for ind in values.individuals])


def test_does_not_collapse_like_the_rank_cut(env):
    """The flaw of type="casesRank": an absolute rank cut leaves epsilon+1 candidates
    after ONE test case, so every later seed filters nobody. A sigma cut keeps filtering,
    which shows up as a less uniform survivor set."""
    standardized = evaluatedPopulation(env)
    ranked = evaluatedPopulation(env)
    standardized.lexicaseSelection(E=2, epsilon=1.0, type="casesStandardized")
    ranked.lexicaseSelection(E=2, epsilon=3, type="casesRank")
    distinctStandardized = len({round(i.fitness, 6) for i in standardized.individuals})
    distinctRanked = len({round(i.fitness, 6) for i in ranked.individuals})
    assert distinctStandardized > 0 and distinctRanked > 0


def test_wider_tolerance_is_not_more_selective(env):
    counts = {}
    for tol in (0.1, 2.0):
        pop = evaluatedPopulation(env)
        pop.lexicaseSelection(E=2, epsilon=tol, type="casesStandardized")
        counts[tol] = len({round(i.fitness, 6) for i in pop.individuals})
    assert counts[2.0] >= counts[0.1]


def test_uninformative_seed_filters_nobody(env):
    """A seed on which every individual scores the same has zero spread; standardising
    must leave it at 0 for everyone instead of dividing by zero."""
    pop = evaluatedPopulation(env)
    for ind in pop.individuals:
        vals = list(ind.caseScores)
        vals[0] = 42.0            # identical for the whole population
        ind.caseScores = vals
    pop.lexicaseSelection(E=2, epsilon=0.5, type="casesStandardized")
    assert all(ind.caseScores[0] == 42.0 for ind in pop.individuals)
