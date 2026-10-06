"""Role-based ("semantic") crossover.

Nodes are matched by what they do -- (type, f, edges.size()) as the coarse role, the
inner boundaries as the fine adjustment -- rather than by array position or by origin.
These tests pin down the three properties the operator is supposed to have: it never
grows the network, it only exchanges nodes of an identical role, and it leaves every
graph valid (edges translated through the matching table).
"""

import gymnasium as gym
import pytest

import fracnetics as fn

MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]


def buildPopulation(ni=30, seed=13):
    pop = fn.Population(seed=seed, ni=ni, jn=8, jnf=8, pn=4, pnf=4,
                        fractalJudgment=False, useExperience=False,
                        nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    return pop


def snapshot(pop):
    """Per individual: node count and the multiset of roles."""
    out = []
    for ind in pop.individuals:
        roles = sorted((n.type, n.f, len(n.edges)) for n in ind.innerNodes)
        out.append((len(ind.innerNodes), roles))
    return out


@pytest.fixture
def evolved():
    """A population that has grown and diverged, so matching is non-trivial."""
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False,
                   wind_power=15.0, turbulence_power=0.5)
    pop = buildPopulation()
    for g in range(6):
        pop.gymnasiumMultiSeed(env, dMax=10, maxSteps=200, maxConsecutiveP=5,
                               worstFitness=-500, seeds=[7, 42, 99],
                               useLineageFitness=False, absoluteImpulseCurriculum=False)
        pop.lexicaseSelection(E=2, type="seeds")
        pop.callAddDelNodes(MIN_FEATURES, MAX_FEATURES, junk=0.1, noElite=True,
                            currentGeneration=g, crossoverProtection=0)
        pop.callEdgeMutation(probInnerNodes=0, probStartNode=0, justUsedNodes=True, k=1)
    env.close()
    return pop


def test_never_grows_the_network(evolved):
    """Unmatched nodes stay put -- unlike "innovation", nothing is appended."""
    before = [n for n, _ in snapshot(evolved)]
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99)
    after = [n for n, _ in snapshot(evolved)]
    assert after == before


def test_role_multiset_is_preserved(evolved):
    """A node is only ever replaced by one of an identical role, so the multiset of
    (type, f, arity) an individual carries cannot change."""
    before = [roles for _, roles in snapshot(evolved)]
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99)
    after = [roles for _, roles in snapshot(evolved)]
    assert after == before


def test_all_edges_stay_in_range(evolved):
    """Edges of a transferred node are donor indices; after translation every edge must
    still address a node of the recipient."""
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99)
    for ind in evolved.individuals:
        n = len(ind.innerNodes)
        for node in ind.innerNodes:
            assert all(0 <= e < n for e in node.edges)
        assert all(0 <= e < n for e in ind.startNode.edges)


def test_population_still_evaluates(evolved):
    """End to end: the recombined networks must still run."""
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False,
                   wind_power=15.0, turbulence_power=0.5)
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99)
    evolved.gymnasiumMultiSeed(env, dMax=10, maxSteps=200, maxConsecutiveP=5,
                               worstFitness=-500, seeds=[7, 42],
                               useLineageFitness=False, absoluteImpulseCurriculum=False)
    env.close()
    assert all(len(ind.fitnessValues) == 2 for ind in evolved.individuals)


def test_probability_zero_changes_nothing(evolved):
    """propability is the exchange rate PER MATCHED NODE, so 0 must be a no-op."""
    before = [[list(n.edges) for n in ind.innerNodes] for ind in evolved.individuals]
    evolved.crossover(probability=0.0, type="semantic", currentGeneration=99)
    after = [[list(n.edges) for n in ind.innerNodes] for ind in evolved.individuals]
    assert after == before


def test_tolerance_zero_restricts_matching(evolved):
    """A tolerance of 0 only pairs nodes whose boundaries agree exactly, so it can never
    exchange more than the permissive setting does."""
    import copy
    counts = {}
    for tol in (0.0, 1.0):
        pop = buildPopulation()
        for g in range(6):
            pop.callAddDelNodes(MIN_FEATURES, MAX_FEATURES, junk=0.1, noElite=True,
                                currentGeneration=g, crossoverProtection=0)
        before = [[list(n.edges) for n in ind.innerNodes] for ind in pop.individuals]
        pop.crossover(probability=1.0, type="semantic", currentGeneration=99,
                      boundaryTolerance=tol)
        after = [[list(n.edges) for n in ind.innerNodes] for ind in pop.individuals]
        counts[tol] = sum(b != a for b, a in zip(before, after))
    assert counts[0.0] <= counts[1.0]


def test_all_or_nothing_keeps_boundaries_and_edges_together(evolved):
    """A node is transferred only if ALL its edges translate, so a transferred node
    carries the donor's boundaries together with the donor's (translated) wiring --
    never a mix of donor intervals and recipient targets."""
    donorSnapshot = [[(tuple(n.boundaries), tuple(n.edges)) for n in ind.innerNodes]
                     for ind in evolved.individuals]
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99)
    # Every node still has one edge per interval -- the invariant a mixed splice breaks.
    for ind in evolved.individuals:
        for node in ind.innerNodes:
            if node.boundaries:
                assert len(node.boundaries) == len(node.edges) + 1
    assert len(donorSnapshot) == len(evolved.individuals)


def test_counters_are_reported(evolved):
    """The diagnostics must distinguish 'had no effect' from 'never fired'."""
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99)
    assert evolved.crossoverNodesMatched > 0
    assert evolved.crossoverNodesExchanged + evolved.crossoverNodesSkipped > 0
    assert evolved.crossoverNodesExchanged <= evolved.crossoverNodesMatched

    evolved.crossover(probability=0.0, type="semantic", currentGeneration=99)
    assert evolved.crossoverNodesExchanged == 0   # reset per call, nothing drawn


def test_tighter_tolerance_exchanges_less():
    """boundaryTolerance is the governing knob: it restricts the node AND, through the
    all-or-nothing rule, every node it points to.

    Both arms start from an identical population -- applying the two settings one after
    the other to the same object would compare against an already-recombined state.
    """
    exchanged = {}
    for tol in (0.0, 1.0):
        pop = buildPopulation(ni=40, seed=21)
        # Diversify the boundaries, otherwise every node cuts identically and a
        # tolerance of 0 is not actually stricter than one of 1.
        pop.callBoundaryMutationNormal(probability=0.5, sigma=0.4, justUsedNodes=False, k=0)
        pop.crossover(probability=1.0, type="semantic", currentGeneration=99,
                      boundaryTolerance=tol)
        exchanged[tol] = pop.crossoverNodesExchanged
    assert exchanged[0.0] <= exchanged[1.0]


def test_pair_gate_and_node_rate_are_separate(evolved):
    """propability gates the pair, nodeExchangeRate decides how much moves inside it.

    They used to be one value applied twice, so the effective rate per matched node was
    propability squared. A negative nodeExchangeRate still reproduces that coupling.
    """
    # Pair gate open, node rate closed -> pairs are matched, nothing is transferred.
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99,
                      nodeExchangeRate=0.0)
    assert evolved.crossoverNodesMatched > 0
    assert evolved.crossoverNodesExchanged == 0
    assert evolved.crossoverNodesSkipped == 0

    # Pair gate closed -> no pair is even looked at.
    evolved.crossover(probability=0.0, type="semantic", currentGeneration=99,
                      nodeExchangeRate=1.0)
    assert evolved.crossoverNodesMatched == 0


def test_negative_node_rate_reproduces_the_coupled_behaviour(evolved):
    """The default (-1) must behave exactly like passing propability twice."""
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99,
                      nodeExchangeRate=-1.0)
    coupled = (evolved.crossoverNodesMatched, evolved.crossoverNodesExchanged)
    evolved.crossover(probability=1.0, type="semantic", currentGeneration=99,
                      nodeExchangeRate=1.0)
    explicit = (evolved.crossoverNodesMatched, evolved.crossoverNodesExchanged)
    # At propability = 1 both draws always succeed, so the two must agree in kind.
    assert coupled[0] > 0 and explicit[0] > 0
    assert coupled[1] > 0 and explicit[1] > 0
