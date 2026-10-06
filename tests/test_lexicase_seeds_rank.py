"""Rank-based lexicase over seeds (type="seedsRank").

The value-based MAD tolerance of type="seeds" is computed from the rewards, and the
LunarLander reward is bimodal (landed ~+200 against crashed ~-200). The MAD there
mostly measures the distance between the two modes rather than the resolution among the
near-equal candidates selection actually has to separate. Cutting by rank makes the
selection pressure independent of the shape and scale of the reward.
"""

import gymnasium as gym
import pytest

import fracnetics as fn

SEEDS = [5, 61, 300, 918]
MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]


@pytest.fixture(scope="module")
def env():
    e = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False,
                 wind_power=15.0, turbulence_power=0.5)
    yield e
    e.close()


def evaluatedPopulation(env, ni=40, seed=17):
    pop = fn.Population(seed=seed, ni=ni, jn=8, jnf=8, pn=4, pnf=4,
                        fractalJudgment=False, useExperience=False,
                        nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    pop.gymnasiumMultiSeed(env, dMax=10, maxSteps=250, maxConsecutiveP=5,
                           worstFitness=-500, seeds=SEEDS, useLineageFitness=False,
                           absoluteImpulseCurriculum=False)
    return pop


def test_selection_runs_and_keeps_population_size(env):
    pop = evaluatedPopulation(env)
    before = len(pop.individuals)
    pop.lexicaseSelection(E=2, epsilon=3, type="seedsRank")
    assert len(pop.individuals) == before
    assert len(pop.indicesElite) == 2


def test_zero_tolerance_matches_value_lexicase(env):
    """A rank cut of 0 keeps exactly the best on each test case -- and so does a value
    cut of 0, because ranking is a monotone transform. The two must agree."""
    fromRank = evaluatedPopulation(env)
    fromValue = evaluatedPopulation(env)
    fromRank.lexicaseSelection(E=2, epsilon=0, type="seedsRank")
    fromValue.lexicaseSelection(E=2, epsilon=0.0, type="seeds")
    assert ([ind.fitness for ind in fromRank.individuals]
            == [ind.fitness for ind in fromValue.individuals])


def test_larger_tolerance_is_less_selective(env):
    """More tolerated rank positions must not narrow the survivor set: with a wider cut
    more distinct individuals should make it into the next generation."""
    distinct = {}
    for tol in (0, 10):
        pop = evaluatedPopulation(env)
        pop.lexicaseSelection(E=2, epsilon=tol, type="seedsRank")
        distinct[tol] = len({round(ind.fitness, 6) for ind in pop.individuals})
    assert distinct[10] >= distinct[0]


def test_rank_cut_ignores_the_reward_scale(env):
    """The whole point: an affine rescaling of every reward leaves the ranks untouched,
    so the selection must be identical. A value-based MAD cut would move with it."""
    baseline = evaluatedPopulation(env)
    scaled = evaluatedPopulation(env)
    for ind in scaled.individuals:
        ind.fitnessValues = [v * 7.0 + 1000.0 for v in ind.fitnessValues]
    baseline.lexicaseSelection(E=2, epsilon=3, type="seedsRank")
    scaled.lexicaseSelection(E=2, epsilon=3, type="seedsRank")
    # The elite is picked by aggregated fitness, which the rescaling does not touch here,
    # so compare the selected per-seed patterns instead.
    for base, scale in zip(baseline.individuals, scaled.individuals):
        restored = [(v - 1000.0) / 7.0 for v in scale.fitnessValues]
        assert list(base.fitnessValues) == pytest.approx(restored, rel=1e-4, abs=1e-2)


def test_unknown_type_still_rejected(env):
    pop = evaluatedPopulation(env)
    with pytest.raises(ValueError):
        pop.lexicaseSelection(E=2, type="seedRank")   # typo: singular
