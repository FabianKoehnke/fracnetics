"""Potential-based reward shaping must not change any ranking.

With phi(terminal) = 0 and gamma = 1 the shaping telescopes over an episode to
-phi(s_0). That offset depends on the initial state, i.e. on the seed, and not on the
individual -- so on any given seed every individual's return shifts by exactly the same
amount and the ordering is untouched. This test pins that property down.
"""

import gymnasium as gym
import numpy as np
import pytest

import fracnetics as fn

SEEDS = [11, 234, 777, 4096]
MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]


def rewardMatrix(pop, env, potential):
    pop.gymnasiumMultiSeed(
        env, dMax=10, maxSteps=300, maxConsecutiveP=5, worstFitness=-500,
        seeds=SEEDS, useLineageFitness=False, absoluteImpulseCurriculum=False,
        potential=potential,
    )
    return np.array([list(ind.fitnessValues) for ind in pop.individuals], dtype=float)


@pytest.fixture(scope="module")
def matrices():
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=0.5)
    pop = fn.Population(seed=7, ni=25, jn=8, jnf=8, pn=4, pnf=4,
                        fractalJudgment=False, useExperience=False,
                        nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    plain = rewardMatrix(pop, env, potential=False)
    shaped = rewardMatrix(pop, env, potential=True)
    env.close()
    return plain, shaped


def test_shaping_offset_is_constant_per_seed(matrices):
    """On one seed, every individual's return shifts by the same amount."""
    plain, shaped = matrices
    delta = shaped - plain
    for seedIdx in range(delta.shape[1]):
        column = delta[:, seedIdx]
        assert np.ptp(column) < 1e-2, (
            f"seed {SEEDS[seedIdx]}: shaping offset varies between individuals "
            f"({column.min():.4f} .. {column.max():.4f}) -- the shaping is not "
            f"potential-based and can bias selection"
        )


def test_shaping_preserves_per_seed_ranking(matrices):
    """The ordering of individuals on each seed survives the shaping."""
    plain, shaped = matrices
    for seedIdx in range(plain.shape[1]):
        assert (np.argsort(plain[:, seedIdx], kind="stable").tolist()
                == np.argsort(shaped[:, seedIdx], kind="stable").tolist())


def test_offset_differs_between_seeds(matrices):
    """The offset is -phi(s_0), so it must actually vary from seed to seed."""
    plain, shaped = matrices
    offsets = (shaped - plain).mean(axis=0)
    assert np.ptp(offsets) > 1e-3, "offsets identical across seeds -- phi(s_0) not in play"
