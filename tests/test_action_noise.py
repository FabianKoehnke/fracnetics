"""actionNoise of Population.gymnasium(): exploration by random actions during an evaluation.

Checked on CartPole (2 actions). The random stream depends on the seed only, so every
network meets the same disturbances on the same seed.
"""

import gymnasium as gym
import pytest

import fracnetics as fn
from fracnetics.parallel_eval import ParallelEvaluator

MIN_FEATURES = [-4.8, -5, -0.418, -10]
MAX_FEATURES = [4.8, 5, 0.418, 10]
KW = dict(seed=5, jn=4, jnf=4, pn=3, pnf=2, fractalJudgment=False, nFeatureValues=[0, 0, 0, 0])
SEEDS = [1, 2, 3, 4, 5, 6]


def population(ni=8):
    pop = fn.Population(ni=ni, **KW)
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    return pop


def scores(pop, **kwargs):
    pop.gymnasium(gym.make("CartPole-v1"), seeds=SEEDS, dMax=10, maxSteps=200, maxConsecutiveP=5, **kwargs)
    return [list(ind.caseScores) for ind in pop.individuals]


def test_zero_noise_changes_nothing():
    pop = population()
    assert scores(pop) == scores(pop, actionNoise=0.0)


def test_noise_is_reproducible_and_changes_episodes():
    pop = population()
    noisy = scores(pop, actionNoise=0.3)
    assert noisy == scores(pop, actionNoise=0.3)
    assert noisy != scores(pop)


def test_full_noise_gives_every_network_the_same_episode():
    """actionNoise 1 replaces every action, and the stream depends on the seed only: all
    networks play the same random episode per seed, whatever their structure."""
    pop = population()
    allScores = scores(pop, actionNoise=1.0)
    assert all(s == allScores[0] for s in allScores)
    assert len(set(allScores[0])) > 1          # but the seeds differ from each other


def test_noise_outside_zero_one_is_rejected():
    pop = population()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            scores(pop, actionNoise=bad)


def test_parallel_evaluator_passes_the_noise_on():
    """Split over workers (seed blocks), the noisy cases equal the serial ones."""
    serial = population()
    expected = scores(serial, actionNoise=0.3)
    pop = population()
    ev = ParallelEvaluator(lambda: gym.make("CartPole-v1"), KW, workers=2)
    try:
        ev.evaluate(pop, SEEDS, dMax=10, maxSteps=200, maxConsecutiveP=5, actionNoise=0.3)
    finally:
        ev.close()
    assert [list(ind.caseScores) for ind in pop.individuals] == expected
