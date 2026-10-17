"""Growth and shrinkage of networks (callAddDelNodes) before any evaluation.

Without an evaluation no node is marked used, so the junk quota alone would delete every
network down to one node -- and redirecting that node's edges then looped forever in
Node::changeEdge(). Whether it happened depended on the random numbers, which differ between
C++ standard libraries: on Linux tests/test_semantic_crossover.py hung for 6 hours.
"""

import fracnetics as fn

MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]


def test_unevaluated_networks_keep_two_valid_nodes():
    for seed in range(1, 6):  # seed 1 with ni=3 hung on macOS as well
        pop = fn.Population(seed=seed, ni=3 if seed == 1 else 20, jn=8, jnf=8, pn=4, pnf=4,
                            fractalJudgment=False, nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
        pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
        for g in range(6):
            pop.callAddDelNodes(MIN_FEATURES, MAX_FEATURES, junk=0.1, noElite=True,
                                currentGeneration=g, crossoverProtection=0)
        for ind in pop.individuals:
            nodes = ind.innerNodes
            assert len(nodes) >= 2
            assert 0 <= ind.startNode.edges[0] < len(nodes)
            for i, node in enumerate(nodes):
                assert node.id == i
                assert all(0 <= e < len(nodes) and e != i for e in node.edges)
