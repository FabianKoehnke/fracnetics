"""k/N rate of the fractal boundary mutation.

The fractal mutation redraws production-rule ratios, not boundaries, so its k/N rate is
normalised by the number of mutable ratios of the eligible nodes: about k redraws per network.
"""

import fracnetics as fn

POPKW = dict(seed=5, jn=8, jnf=8, pn=4, pnf=4, fractalJudgment=True, useExperience=False,
             nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
MINF = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAXF = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]


def population(ni=300):
    pop = fn.Population(ni=ni, **POPKW)
    pop.setAllNodeBoundaries(MINF, MAXF)
    pop.checkOrder = False
    return pop


def ratios(pop):
    return [[list(n.productionRuleParameter)[1:-1] for n in net.innerNodes if n.type == "J"]
            for net in pop.individuals]


def redrawsPerNetwork(pop, **kwargs):
    before = ratios(pop)
    pop.callBoundaryMutationFractal(minF=MINF, maxF=MAXF, justUsedNodes=False, **kwargs)
    after = ratios(pop)
    return [sum(a != b for nodeA, nodeB in zip(netA, netB) for a, b in zip(nodeA, nodeB))
            for netA, netB in zip(before, after)]


def test_k_gives_about_k_ratio_redraws_per_network():
    for k in (1.0, 2.0):
        counts = redrawsPerNetwork(population(), probability=0.0, k=k)
        mean = sum(counts) / len(counts)
        assert abs(mean - k) < 0.3, (k, mean)


def test_k_zero_keeps_the_probability_per_ratio():
    pop = population()
    total = sum(len(node) for net in ratios(pop) for node in net)
    counts = redrawsPerNetwork(pop, probability=0.5)
    assert abs(sum(counts) / total - 0.5) < 0.05


def test_default_k_is_the_plain_probability():
    def mutate(**extra):
        pop = population(ni=20)
        pop.callBoundaryMutationFractal(0.5, MINF, MAXF, **extra)
        return ratios(pop)
    assert mutate() == mutate(k=0.0)


def test_k_on_a_non_fractal_population_changes_nothing():
    pop = fn.Population(ni=10, **{**POPKW, "fractalJudgment": False})
    pop.setAllNodeBoundaries(MINF, MAXF)
    pop.checkOrder = False
    before = [[list(n.boundaries) for n in net.innerNodes] for net in pop.individuals]
    pop.callBoundaryMutationFractal(probability=0.0, minF=MINF, maxF=MAXF, k=2.0)
    assert [[list(n.boundaries) for n in net.innerNodes] for net in pop.individuals] == before


# ── Sorted cuts ──────────────────────────────────────────────────────────────

def bigPopulation():
    # 20 nodes: k_d up to (4, 2), i.e. three ratios per node
    pop = fn.Population(ni=20, **{**POPKW, "jn": 16, "jnf": 8,
                                  "nFeatureValues": [0, 0, 0, 0, 0, 0, 2, 2]})
    pop.setAllNodeBoundaries(MINF, MAXF)
    pop.checkOrder = False
    return pop


def cuts(pop):
    return [list(n.productionRuleParameter) for net in pop.individuals for n in net.innerNodes
            if n.type == "J"]


def test_cuts_are_sorted_from_the_start():
    allCuts = cuts(bigPopulation())
    assert any(len(c) >= 5 for c in allCuts)            # nodes with three or more ratios exist
    assert all(c == sorted(c) and c[0] == 0 and c[-1] == 1 for c in allCuts)


def test_a_redraw_stays_between_the_neighbouring_cuts():
    pop = bigPopulation()
    before = cuts(pop)
    pop.callBoundaryMutationFractal(probability=1.0, minF=MINF, maxF=MAXF)
    after = cuts(pop)
    assert after != before
    for c in after:
        assert c == sorted(c) and c[0] == 0 and c[-1] == 1
        assert all(0 < v < 1 for v in c[1:-1])
