"""The (case x objective) test-case grid for lexicaseSelection.

type="objectivesPerCase" keeps the objectives of every case separate instead of averaging
them (lexicaseObjectives) or reducing a case to one scalar (caseScores). The objectives
come from the episodeObjectives hook of Population.gymnasium(). This test pins down the
data layout the grid relies on and that selection actually runs on it.
"""

import gymnasium as gym
import pytest

import fracnetics as fn

SEEDS = [3, 91, 512]
MIN_FEATURES = [-4.8, -5, -0.418, -10]
MAX_FEATURES = [4.8, 5, 0.418, 10]


def objectives(episode):
    """Three criteria with different scales: return, final position, final pole angle."""
    obs = episode.observation
    return [episode.episodeReturn, -abs(float(obs[0])), -abs(float(obs[2]))]


@pytest.fixture
def evaluated():
    env = gym.make("CartPole-v1")
    pop = fn.Population(seed=3, ni=20, jn=4, jnf=4, pn=2, pnf=2,
                        fractalJudgment=False, nFeatureValues=[0, 0, 0, 0])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    pop.gymnasium(env, seeds=SEEDS, dMax=10, maxSteps=200, maxConsecutiveP=5,
                  episodeObjectives=objectives)
    yield pop
    env.close()


def test_grid_shape(evaluated):
    """One objective vector per case, for every individual."""
    for ind in evaluated.individuals:
        assert len(ind.lexicaseObjectives) == 3
        assert len(ind.objectivesPerCase) == len(SEEDS)
        assert all(len(v) == 3 for v in ind.objectivesPerCase)


def test_grid_averages_to_lexicase_objectives(evaluated):
    """The per-case grid is the decomposition of the averaged vector, not a new quantity."""
    for ind in evaluated.individuals:
        for objIdx in range(3):
            mean = sum(v[objIdx] for v in ind.objectivesPerCase) / len(SEEDS)
            assert mean == pytest.approx(ind.lexicaseObjectives[objIdx], rel=1e-4, abs=1e-3)


def test_first_objective_is_the_case_score(evaluated):
    """The hook saw the same return that went into caseScores."""
    for ind in evaluated.individuals:
        assert [v[0] for v in ind.objectivesPerCase] == list(ind.caseScores)


def test_grid_separates_cases(evaluated):
    """The grid must actually differ between cases -- otherwise it adds no test cases."""
    spread = [
        max(v[objIdx] for v in ind.objectivesPerCase) - min(v[objIdx] for v in ind.objectivesPerCase)
        for ind in evaluated.individuals
        for objIdx in range(3)
    ]
    assert max(spread) > 0.0


@pytest.mark.parametrize("selection", ["objectivesPerCase", "objectives"])
def test_selection_runs_and_keeps_population_size(evaluated, selection):
    before = len(evaluated.individuals)
    evaluated.lexicaseSelection(E=2, type=selection)
    assert len(evaluated.individuals) == before
    assert len(evaluated.indicesElite) == 2


def test_unknown_type_still_rejected(evaluated):
    with pytest.raises(ValueError):
        evaluated.lexicaseSelection(E=2, type="nonsense")


def test_objective_types_need_objectives():
    """Without a hook there are no objectives; selecting on them must fail loudly instead
    of degrading to a random draw."""
    env = gym.make("CartPole-v1")
    pop = fn.Population(seed=3, ni=6, jn=4, jnf=4, pn=2, pnf=2,
                        fractalJudgment=False, nFeatureValues=[0, 0, 0, 0])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    pop.gymnasium(env, seeds=SEEDS, dMax=10, maxSteps=50, maxConsecutiveP=5)
    env.close()
    assert all(len(ind.objectivesPerCase) == 0 for ind in pop.individuals)
    for selection in ("objectives", "objectivesPerCase"):
        with pytest.raises(ValueError):
            pop.lexicaseSelection(E=1, type=selection)
