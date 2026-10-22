"""The order check of the steps of a generation (Population::Step).

Evaluation writes the usage flags, fitness and per-case data every later step reads; each
step that changes the networks makes some of them stale. The population therefore only
accepts the steps of a generation in the documented order and rejects every wrong order
with a RuntimeError -- until the next evaluation starts a new generation.
"""

import gymnasium as gym
import numpy as np
import pytest

import fracnetics as fn

MIN_F = [-4.8, -5, -0.418, -10]
MAX_F = [4.8, 5, 0.418, 10]
SEEDS = [1, 2, 3]


def buildPopulation():
    pop = fn.Population(seed=7, ni=10, jn=4, jnf=4, pn=2, pnf=2,
                        fractalJudgment=False, nFeatureValues=[0, 0, 0, 0])
    pop.setAllNodeBoundaries(MIN_F, MAX_F)
    return pop


def evaluate(pop):
    env = gym.make("CartPole-v1")
    pop.gymnasium(env, seeds=SEEDS, dMax=10, maxSteps=50, maxConsecutiveP=5)
    env.close()


def edges(pop):
    return [[list(n.edges) for n in ind.innerNodes] for ind in pop.individuals]


@pytest.fixture
def pop():
    p = buildPopulation()
    evaluate(p)
    return p


# One call per step, in the documented order
STEPS = {
    "age": lambda p: p.callAgeUnusedNodes(),
    "selection": lambda p: p.tournamentSelection(N=2, E=1),
    "adaptiveK": lambda p: p.updateAdaptiveK(),
    "caseSpecialist": lambda p: p.crossover(probability=1.0, type="caseSpecialistAppend",
                                            crossoverProtection=1),
    "addDel": lambda p: p.callAddDelNodes(MIN_F, MAX_F, junk=0.1, noElite=True),
    "crossover": lambda p: p.crossover(probability=1.0, type="semantic"),
    "edgeMutation": lambda p: p.callEdgeMutation(0.1, 0.1),
    "boundaryMutation": lambda p: p.callBoundaryMutationNormal(0.1, 0.2),
    "capLineage": lambda p: p.capLineageAfterMutation(),
}

# The last step of each sequence is the one that must be rejected
WRONG_ORDERS = {
    "selection after mutation": ["edgeMutation", "selection"],
    "selection after crossover": ["crossover", "selection"],
    "addDelNodes after edge mutation": ["edgeMutation", "addDel"],
    "addDelNodes after boundary mutation": ["boundaryMutation", "addDel"],
    "addDelNodes after crossover": ["crossover", "addDel"],
    "caseSpecialist after addDelNodes": ["addDel", "caseSpecialist"],
    "crossover after mutation": ["edgeMutation", "crossover"],
    "ageUnusedNodes after addDelNodes": ["addDel", "age"],
    "ageUnusedNodes after crossover": ["crossover", "age"],
    "ageUnusedNodes twice": ["age", "age"],
    "selection twice": ["selection", "selection"],
    "caseSpecialist twice": ["caseSpecialist", "caseSpecialist"],
    "updateAdaptiveK after mutation": ["edgeMutation", "adaptiveK"],
    "mutation after capLineage": ["capLineage", "edgeMutation"],
}


def test_documented_order_passes(pop):
    for _ in range(2):
        for step in STEPS.values():
            step(pop)
        evaluate(pop)


@pytest.mark.parametrize("sequence", WRONG_ORDERS.values(), ids=WRONG_ORDERS.keys())
def test_wrong_order_raises(pop, sequence):
    *before, last = sequence
    for name in before:
        STEPS[name](pop)
    with pytest.raises(RuntimeError, match="Population: "):
        STEPS[last](pop)


def test_rejected_step_changes_nothing(pop):
    pop.callEdgeMutation(0.5, 0.5)
    before = edges(pop)
    with pytest.raises(RuntimeError):
        pop.callAddDelNodes(MIN_F, MAX_F, junk=0.0)
    with pytest.raises(RuntimeError):
        pop.crossover(probability=1.0, type="uniform")
    assert edges(pop) == before


def test_every_step_is_optional(pop):
    pop.callAddDelNodes(MIN_F, MAX_F, junk=0.1)
    pop.callBoundaryMutationUniform(0.1)


def test_mutations_and_crossovers_repeat_in_any_order(pop):
    pop.crossover(probability=1.0, type="semantic")
    pop.crossover(probability=1.0, type="uniform")
    pop.callBoundaryMutationNormal(0.1, 0.2)
    pop.callEdgeMutation(0.1, 0.1)
    pop.callBoundaryMutationUniform(0.1)


def test_nothing_is_checked_before_the_first_evaluation():
    """Fresh (or unpickled) populations accept any step."""
    p = buildPopulation()
    p.callEdgeMutation(0.1, 0.1)
    p.callAddDelNodes(MIN_F, MAX_F, junk=0.1)
    p.crossover(probability=0.5, type="uniform")
    p.callAgeUnusedNodes()


def test_check_order_false_disables_the_check(pop):
    pop.checkOrder = False
    pop.callEdgeMutation(0.1, 0.1)
    pop.callAddDelNodes(MIN_F, MAX_F, junk=0.1)
    # The steps are still recorded, so switching back on mid-generation stays consistent
    pop.checkOrder = True
    with pytest.raises(RuntimeError):
        pop.crossover(probability=1.0, type="semantic")


def test_every_evaluation_starts_a_new_generation(pop):
    X = np.random.default_rng(0).uniform(-1, 1, size=(5, 4)).astype(np.float32)
    y = np.zeros(5, dtype=np.int32)
    evaluations = [
        lambda p: evaluate(p),
        lambda p: p.callTraversePath(X, dMax=10),
        lambda p: p.accuracy(X, y, dMax=10, penalty=1),
        lambda p: p.markEvaluated(),
    ]
    for evaluation in evaluations:
        pop.callEdgeMutation(0.1, 0.1)
        evaluation(pop)
        pop.callAgeUnusedNodes()
        pop.tournamentSelection(N=2, E=1)


def test_pareto_objectives_belong_to_the_evaluation(pop):
    pop.calculateParetoObjectives()
    pop.paretoTournamentSelection(N=2, E=1)
    with pytest.raises(RuntimeError):
        pop.calculateParetoObjectives()


def test_rejected_selection_call_does_not_count(pop):
    with pytest.raises(ValueError):
        pop.lexicaseSelection(E=1, type="nonsense")
    pop.lexicaseSelection(E=1, type="cases")


def test_error_names_both_steps(pop):
    pop.callBoundaryMutationUniform(0.1)
    with pytest.raises(RuntimeError, match=r"callAddDelNodes\(\) after callBoundaryMutationUniform\(\)"):
        pop.callAddDelNodes(MIN_F, MAX_F)
