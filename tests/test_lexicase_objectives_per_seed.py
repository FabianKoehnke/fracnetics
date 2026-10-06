"""The (seed x objective) test-case grid for lexicaseSelection.

type="objectivesPerSeed" keeps the five episode objectives separate for every seed
instead of averaging them (lexicaseObjectives) or reducing a seed to one scalar
(fitnessValues). This test pins down the data layout it relies on and that selection
actually runs on it.
"""

import gymnasium as gym
import pytest

import fracnetics as fn

SEEDS = [3, 91, 512]
MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]


@pytest.fixture
def evaluated():
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=0.5)
    pop = fn.Population(seed=3, ni=20, jn=8, jnf=8, pn=4, pnf=4,
                        fractalJudgment=False, useExperience=False,
                        nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    pop.gymnasiumMultiSeed(env, dMax=10, maxSteps=300, maxConsecutiveP=5,
                           worstFitness=-500, seeds=SEEDS, useLineageFitness=False,
                           absoluteImpulseCurriculum=False)
    yield pop
    env.close()


def test_grid_shape(evaluated):
    """One 5-D objective vector per seed, for every individual."""
    nObjectives = len(evaluated.individuals[0].lexicaseObjectives)
    assert nObjectives == 5, "the safe objective set is five criteria"
    for ind in evaluated.individuals:
        assert len(ind.objectivesPerSeed) == len(SEEDS)
        assert all(len(v) == nObjectives for v in ind.objectivesPerSeed)


def test_grid_averages_to_lexicase_objectives(evaluated):
    """The per-seed grid is the decomposition of the averaged vector, not a new quantity."""
    nObjectives = len(evaluated.individuals[0].lexicaseObjectives)
    for ind in evaluated.individuals:
        for objIdx in range(nObjectives):
            mean = sum(v[objIdx] for v in ind.objectivesPerSeed) / len(SEEDS)
            assert mean == pytest.approx(ind.lexicaseObjectives[objIdx], rel=1e-4, abs=1e-3)


def test_grid_separates_seeds(evaluated):
    """The grid must actually differ between seeds -- otherwise it adds no test cases."""
    spread = [
        max(v[objIdx] for v in ind.objectivesPerSeed) - min(v[objIdx] for v in ind.objectivesPerSeed)
        for ind in evaluated.individuals
        for objIdx in range(len(evaluated.individuals[0].lexicaseObjectives))
    ]
    assert max(spread) > 0.0


def test_selection_runs_and_keeps_population_size(evaluated):
    before = len(evaluated.individuals)
    evaluated.lexicaseSelection(E=2, type="objectivesPerSeed")
    assert len(evaluated.individuals) == before
    assert len(evaluated.indicesElite) == 2


def test_unknown_type_still_rejected(evaluated):
    with pytest.raises(ValueError):
        evaluated.lexicaseSelection(E=2, type="nonsense")


def test_objectives_are_the_safe_four(evaluated):
    """The four criteria all have the task as their optimum.

    Deliberately excluded are the former posture and fuel sums: both accumulate
    non-positive per-step terms, so both are maximised by an episode that ends at once
    without firing an engine. Used as lexicase test cases they pulled the population
    towards inertia. The properties checked here are what distinguishes the new set.
    """
    for ind in evaluated.individuals:
        for seedIdx, obj in enumerate(ind.objectivesPerSeed):
            gymReward, descentVertical, lateral, precision, landed = obj
            # [0] total Gymnasium reward -- unbounded in sign, just has to be finite.
            assert gymReward == gymReward  # not NaN
            # [1] vertical speed, [2] lateral speed and [3] horizontal precision are
            # negated magnitudes when landed, a fixed penalty otherwise -- both <= 0.
            assert descentVertical <= 0.0
            assert lateral <= 0.0
            assert precision <= 0.0
            # Without a landing they must sit at the penalty, so hovering cannot win them.
            if landed < 0.5:
                assert descentVertical == -4.0
                assert lateral == -4.0
                assert precision == -2.0
            # [4] landed is a flag.
            assert landed in (0.0, 1.0)


def test_hovering_cannot_win_the_endstate_objectives(evaluated):
    """The failure mode this fix addresses: an episode that never lands must not score
    well on the end-state objectives just because it came to rest above the pad.

    Every non-landing episode sits at the penalty, so its value can never beat a landing
    -- whatever its final position and speed happened to be.
    """
    landings, failures = [], []
    for ind in evaluated.individuals:
        for obj in ind.objectivesPerSeed:
            (landings if obj[4] > 0.5 else failures).append(obj)
    assert failures, "no failed episodes in this population -- test says nothing"
    for obj in failures:
        assert obj[1] == -4.0 and obj[2] == -4.0 and obj[3] == -2.0
    for landed in landings:
        for failed in failures:
            # A landing is never worse than a failure on any of the three.
            assert landed[1] >= failed[1]
            assert landed[2] >= failed[2]
            assert landed[3] >= failed[3]


def test_lateral_and_vertical_speed_are_separate(evaluated):
    """The whole point of splitting objective 1: a network can be good vertically and
    bad laterally. If the two were still combined they would move together."""
    vertical = [obj[1] for ind in evaluated.individuals for obj in ind.objectivesPerSeed]
    lateral = [obj[2] for ind in evaluated.individuals for obj in ind.objectivesPerSeed]
    assert len(vertical) == len(lateral)
    # They must not be the same series -- that would mean the split did not happen.
    assert vertical != lateral
    # And at least one episode must differ markedly in the two, otherwise the split
    # carries no information on this population.
    assert max(abs(v - l) for v, l in zip(vertical, lateral)) > 0.01
