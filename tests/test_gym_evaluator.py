"""The problem-independent Gymnasium evaluator, Population.gymnasium().

Checked on CartPole (4 observation values, 2 actions) rather than on LunarLander: the
evaluator must not assume anything about the observation layout or the action set.
"""

import functools
import glob
import os
import pickle

import gymnasium as gym
import numpy as np
import pytest

import fracnetics as fn

MIN_FEATURES = [-4.8, -5, -0.418, -10]
MAX_FEATURES = [4.8, 5, 0.418, 10]
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class Recorder(gym.Wrapper):
    """Records reset options and executed actions."""

    def __init__(self, env):
        super().__init__(env)
        self.options, self.actions = [], []

    def reset(self, *, seed=None, options=None):
        self.options.append(options)
        self.actions.append([])
        return self.env.reset(seed=seed)

    def step(self, action):
        self.actions[-1].append(int(action))
        return self.env.step(action)


def cartpolePopulation(ni=10, pnf=2, seed=5):
    pop = fn.Population(seed=seed, ni=ni, jn=4, jnf=4, pn=3, pnf=pnf,
                        fractalJudgment=False, nFeatureValues=[0, 0, 0, 0])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    return pop


def test_cases_and_fitness():
    pop = cartpolePopulation()
    pop.gymnasium(gym.make("CartPole-v1"), seeds=[1, 2, 3], dMax=10, maxSteps=100,
                  maxConsecutiveP=5)
    for ind in pop.individuals:
        assert len(ind.caseScores) == 3
        assert len(ind.visitedNodesPerCase) == 3
        assert ind.fitness == pytest.approx(sum(ind.caseScores) / 3)
        assert all(0 < s <= 100 for s in ind.caseScores)  # +1 per step, capped by maxSteps
        assert len(ind.objectivesPerCase) == 0 and len(ind.lexicaseObjectives) == 0


def test_visited_nodes_do_not_carry_over_from_the_previous_evaluation():
    """The first case resets the visit counters, so its node list must not contain nodes
    that only the previous evaluation entered. A pickled copy starts from zero counters
    and gives the reference."""
    env = gym.make("CartPole-v1")
    kwargs = dict(dMax=10, maxSteps=100, maxConsecutiveP=5)
    pop = cartpolePopulation()
    pop.checkOrder = False  # two evaluations in a row
    pop.gymnasium(env, seeds=[11, 12], **kwargs)
    pop.gymnasium(env, seeds=[7, 8], **kwargs)

    fresh = cartpolePopulation()
    fresh.individuals = [pickle.loads(pickle.dumps(ind)) for ind in pop.individuals]
    fresh.gymnasium(env, seeds=[7, 8], **kwargs)
    for ind, ref in zip(pop.individuals, fresh.individuals):
        assert [sorted(v) for v in ind.visitedNodesPerCase] == \
               [sorted(v) for v in ref.visitedNodesPerCase]


def test_observation_must_cover_all_judgment_functions():
    """jnf=8 on a 4-value observation would read past its end; refuse instead."""
    pop = fn.Population(seed=5, ni=2, jn=4, jnf=8, pn=2, pnf=2, fractalJudgment=False,
                        nFeatureValues=[0] * 8)
    pop.setAllNodeBoundaries(MIN_FEATURES * 2, MAX_FEATURES * 2)
    with pytest.raises(ValueError, match="jnf=8"):
        pop.gymnasium(gym.make("CartPole-v1"), seeds=[1], dMax=10, maxSteps=10,
                      maxConsecutiveP=5)


def test_seeds_must_not_be_empty():
    with pytest.raises(ValueError):
        cartpolePopulation().gymnasium(gym.make("CartPole-v1"), seeds=[], dMax=10,
                                       maxSteps=10, maxConsecutiveP=5)


def test_reset_options_reach_the_environment():
    env = Recorder(gym.make("CartPole-v1"))
    pop = cartpolePopulation(ni=2)
    options = [{"level": 0.1}, {"level": 0.2}]
    pop.gymnasium(env, seeds=[1, 2], dMax=10, maxSteps=20, maxConsecutiveP=5,
                  resetOptions=options)
    assert env.options == options * 2  # two individuals, two seeds each
    with pytest.raises(ValueError):
        pop.gymnasium(env, seeds=[1, 2], dMax=10, maxSteps=20, maxConsecutiveP=5,
                      resetOptions=[{}])


def test_hold_action_is_never_sent_to_the_environment():
    """With pnf=3 on CartPole, processing function 2 is not an action. As holdAction it
    repeats the previous decision; without it the environment rejects the action."""
    pop = cartpolePopulation(ni=20, pnf=3)
    env = Recorder(gym.make("CartPole-v1"))
    pop.gymnasium(env, seeds=[1, 2], dMax=10, maxSteps=200, maxConsecutiveP=5, holdAction=2)
    actions = [a for episode in env.actions for a in episode]
    assert set(actions) <= {0, 1}
    assert any(n.f == 2 and n.type == "P" for ind in pop.individuals for n in ind.innerNodes)
    with pytest.raises(Exception):
        pop.gymnasium(gym.make("CartPole-v1"), seeds=[1], dMax=10, maxSteps=200,
                      maxConsecutiveP=5)


def test_fallback_action_once_invalid():
    """dMax=1 makes every network that reaches a judgment node invalid at once; from then
    on the episode runs on fallbackAction."""
    pop = cartpolePopulation(ni=10)
    env = Recorder(gym.make("CartPole-v1"))
    flags = []
    pop.gymnasium(env, seeds=[1], dMax=1, maxSteps=50, maxConsecutiveP=5, fallbackAction=1,
                  episodeObjectives=lambda ep: flags.append(ep.invalid) or [0.0])
    for invalid, actions in zip(flags, env.actions):
        if invalid:
            assert actions[-1] == 1
    assert any(flags)


def test_episode_result():
    seen = []

    def hook(ep):
        seen.append(ep)
        return [ep.episodeReturn, float(ep.steps)]

    pop = cartpolePopulation(ni=3)
    pop.gymnasium(gym.make("CartPole-v1"), seeds=[7, 8], dMax=10, maxSteps=30,
                  maxConsecutiveP=5, episodeObjectives=hook)
    assert [ep.seed for ep in seen] == [7, 8] * 3
    for ep in seen:
        assert ep.steps == len(ep.rewards) <= 30
        assert ep.rewards.dtype == np.float32
        # Summed in float, step by step, like Network::fitness
        assert ep.episodeReturn == functools.reduce(lambda a, b: np.float32(a + b), ep.rewards,
                                                    np.float32(0.0))
        assert ep.truncated is False  # CartPole's own limit is 500; maxSteps ended it
        assert ep.terminated or ep.steps == 30
        assert np.asarray(ep.initialObservation).shape == (4,)
        assert np.asarray(ep.observation).shape == (4,)
        assert isinstance(ep.info, dict)
    for ind in pop.individuals:
        assert [v[0] for v in ind.objectivesPerCase] == list(ind.caseScores)


def test_lineage_tracking_uses_success_threshold():
    pop = cartpolePopulation(ni=4)
    pop.gymnasium(gym.make("CartPole-v1"), seeds=[1, 2, 3, 4], dMax=10, maxSteps=60,
                  maxConsecutiveP=5, trackLineage=True, successThreshold=20.0)
    for ind in pop.individuals:
        assert ind.lineageN == 4
        expected = sum(1 for s in ind.caseScores if s > 20.0) / 4
        assert ind.lineageSuccessRate == pytest.approx(expected)
        assert ind.lineageMean == pytest.approx(ind.fitness, rel=1e-5)


def test_accuracy_fills_case_scores_for_lexicase():
    """Classification: one case per sample, so lexicase can select on single samples."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(40, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int32)
    pop = fn.Population(seed=2, ni=12, jn=3, jnf=4, pn=2, pnf=2, fractalJudgment=False,
                        nFeatureValues=[0, 0, 0, 0])
    pop.setAllNodeBoundaries([-1] * 4, [1] * 4)
    pop.accuracy(X, y, dMax=10, penalty=1)
    for ind in pop.individuals:
        assert len(ind.caseScores) == len(y)
        assert set(ind.caseScores) <= {0.0, 1.0}
        assert ind.fitness == pytest.approx(sum(ind.caseScores) / len(y))
    pop.lexicaseSelection(E=1, type="cases")
    assert len(pop.individuals) == 12


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(FIXTURES, "*.pkl"))))
def test_pre_3_0_networks_still_load(path):
    """Archived networks were pickled with a trailing lastStepRewards field."""
    net = pickle.load(open(path, "rb"))
    assert len(net.innerNodes) > 0
    again = pickle.loads(pickle.dumps(net))
    assert [n.edges for n in again.innerNodes] == [n.edges for n in net.innerNodes]
    assert list(again.caseScores) == list(net.caseScores)
