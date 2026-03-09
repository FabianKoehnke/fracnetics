"""Tests for parallel fitness evaluation methods (accuracy, callTraversePath)."""
import math

import fracnetics as fn


def make_population(ni=20, seed=42):
    pop = fn.Population(
        seed=seed,
        ni=ni,
        jn=5,
        jnf=4,
        pn=3,
        pnf=2,
        fractalJudgment=False,
    )
    minF = [-4.8, -5.0, -0.418, -10.0]
    maxF = [4.8, 5.0, 0.418, 10.0]
    pop.setAllNodeBoundaries(minF, maxF)
    return pop


def test_accuracy_parallel():
    """accuracy() should set fitness for all individuals using OpenMP parallel loop."""
    pop = make_population(ni=20)
    initial_fitness = [ind.fitness for ind in pop.individuals]

    # Build a simple 2-class dataset
    X = [[float(i % 4), float(i % 3), float(i % 2), float(i % 5)] for i in range(40)]
    y = [i % 2 for i in range(40)]

    pop.accuracy(X, y, dMax=20, penalty=10)

    updated_fitness = [ind.fitness for ind in pop.individuals]
    changed = sum(1 for a, b in zip(initial_fitness, updated_fitness) if a != b)
    assert changed > 0, "accuracy() did not update any individual fitness"
    assert all(0.0 <= f <= 1.0 for f in updated_fitness), "fitness values must be in [0.0, 1.0]"


def test_call_traverse_path_parallel():
    """callTraversePath() should populate decisions for all individuals via OpenMP."""
    pop = make_population(ni=20)
    X = [[float(i % 4), float(i % 3), float(i % 2), float(i % 5)] for i in range(10)]

    pop.callTraversePath(X, dMax=20)

    for ind in pop.individuals:
        assert len(ind.decisions) == len(X), (
            f"Expected {len(X)} decisions, got {len(ind.decisions)}"
        )


def test_accuracy_consistent_results():
    """accuracy() results should be deterministic: two populations with the same seed give the same fitness."""
    X = [[float(i % 4), float(i % 3), float(i % 2), float(i % 5)] for i in range(40)]
    y = [i % 2 for i in range(40)]

    pop1 = make_population(ni=10, seed=7)
    pop1.accuracy(X, y, dMax=20, penalty=10)

    pop2 = make_population(ni=10, seed=7)
    pop2.accuracy(X, y, dMax=20, penalty=10)

    for f1, f2 in zip(
        [ind.fitness for ind in pop1.individuals],
        [ind.fitness for ind in pop2.individuals],
    ):
        assert f1 == f2, "accuracy() is not deterministic for same seed"


def test_accuracy_all_individuals_evaluated():
    """accuracy() must evaluate every individual in the population."""
    pop = make_population(ni=30)
    X = [[float(i % 4), float(i % 3), float(i % 2), float(i % 5)] for i in range(50)]
    y = [i % 2 for i in range(50)]

    pop.accuracy(X, y, dMax=20, penalty=10)

    for idx, ind in enumerate(pop.individuals):
        assert not math.isinf(ind.fitness), (
            f"Individual {idx} was not evaluated by accuracy()"
        )
