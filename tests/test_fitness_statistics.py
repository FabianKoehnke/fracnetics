"""bestFit, meanFitness and minFitness are computed from the current individuals.

They used to be stored by the selections only, so after a validation run without a selection
they still showed the training values. Now every read reflects the fitness the individuals
carry at that moment.
"""

import pickle

import gymnasium as gym
import numpy as np
import pytest

import fracnetics as fn

MIN_F = [-4.8, -5, -0.418, -10]
MAX_F = [4.8, 5, 0.418, 10]


def buildPopulation():
    pop = fn.Population(seed=3, ni=12, jn=4, jnf=4, pn=2, pnf=2,
                        fractalJudgment=False, nFeatureValues=[0, 0, 0, 0])
    pop.setAllNodeBoundaries(MIN_F, MAX_F)
    return pop


def evaluate(pop, seeds):
    env = gym.make("CartPole-v1")
    pop.gymnasium(env, seeds=seeds, dMax=10, maxSteps=50, maxConsecutiveP=5)
    env.close()


def assertStatistics(pop):
    fitness = np.array([ind.fitness for ind in pop.individuals], dtype=np.float64)
    assert pop.bestFit == pytest.approx(fitness.max())
    assert pop.meanFitness == pytest.approx(fitness.mean())
    assert pop.minFitness == pytest.approx(fitness.min())


def test_statistics_follow_evaluation_and_selection():
    pop = buildPopulation()
    evaluate(pop, [1, 2])
    assertStatistics(pop)
    best = pop.bestFit
    pop.tournamentSelection(N=2, E=1)
    assertStatistics(pop)  # the mean now includes the elite
    assert pop.bestFit == best  # the elite keeps the best individual


def test_statistics_follow_a_validation_without_selection():
    pop = buildPopulation()
    evaluate(pop, [1, 2])
    pop.tournamentSelection(N=2, E=1)
    pop.callEdgeMutation(0.1, 0.1)
    evaluate(pop, [100, 101, 102])  # validation: evaluation only, no selection
    assertStatistics(pop)
    for i, ind in enumerate(pop.individuals):  # fitness set in Python, as in the IRIS example
        ind.fitness = float(i)
    assert (pop.bestFit, pop.meanFitness, pop.minFitness) == (11.0, 5.5, 0.0)


def test_statistics_are_read_only():
    pop = buildPopulation()
    for name in ("bestFit", "meanFitness", "minFitness"):
        with pytest.raises(AttributeError):
            setattr(pop, name, 1.0)


def test_pickle_round_trip_and_legacy_state():
    pop = buildPopulation()
    evaluate(pop, [1, 2])
    pop.tournamentSelection(N=2, E=1)
    back = pickle.loads(pickle.dumps(pop))
    assertStatistics(back)
    assert back.bestFit == pop.bestFit
    assert list(back.indicesElite) == list(pop.indicesElite)

    # Older pickles carried the three statistics between fractalJudgment and the individuals
    state = pop.__getstate__()
    legacyState = state[:6] + (-1.0, state[6], -2.0, -3.0, state[7])
    old = fn.Population.__new__(fn.Population)
    old.__setstate__(legacyState)
    assertStatistics(old)  # the stored values are ignored
    assert list(old.indicesElite) == list(pop.indicesElite)
