"""Grace-period protection for recently active nodes (addDelNodes).

Replaced an indirect route via the coactivation matrix (recordTransition ->
findTransitionClusters -> clusterLabels), which has been removed. junk alone only caps
HOW MANY unused nodes survive; this counter decides WHICH ones.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fracnetics as fn

MINF = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAXF = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
POPKW = dict(seed=5, jn=6, jnf=8, pn=3, pnf=4, fractalJudgment=False,
             useExperience=False, nFeatureValues=[3, 3, 3, 3, 3, 3, 2, 2])


def _build(ni=10):
    pop = fn.Population(ni=ni, **POPKW)
    pop.setAllNodeBoundaries(MINF, MAXF)
    return pop


def _markUsed(net, idx, flag):
    """used is set by the traversal and cleared by the next evaluation
    (fitGymnasium(newRun=True) -> clearUsedNodes)."""
    nodes = net.innerNodes
    nodes[idx].used = flag
    net.innerNodes = nodes


def test_counter_starts_clean():
    pop = _build()
    for i in range(len(pop.individuals)):
        for nd in pop.individuals[i].innerNodes:
            assert nd.everUsed is False
            assert nd.unusedSince == 0


def test_never_used_nodes_do_not_age():
    """Nodes that were never traversed stay immediately deletable (decided design)."""
    pop = _build()
    for _ in range(4):
        pop.callAgeUnusedNodes()
    for i in range(len(pop.individuals)):
        for nd in pop.individuals[i].innerNodes:
            assert nd.everUsed is False
            assert nd.unusedSince == 0, "a never-used node must not accumulate a grace period"


def test_used_node_resets_then_ages():
    pop = _build(ni=1)
    net = pop.individuals[0]
    _markUsed(net, 0, True)

    pop.callAgeUnusedNodes()
    assert net.innerNodes[0].everUsed is True
    assert net.innerNodes[0].unusedSince == 0, "a node in use has no idle generations"

    _markUsed(net, 0, False)          # next generation did not traverse it
    for expected in (1, 2, 3):
        pop.callAgeUnusedNodes()
        assert net.innerNodes[0].unusedSince == expected

    _markUsed(net, 0, True)           # used again -> counter resets
    pop.callAgeUnusedNodes()
    assert net.innerNodes[0].unusedSince == 0


def test_grace_period_semantics_are_off_by_one():
    """unusedSince >= nodeGracePeriod deletes, so n grants n-1 idle generations.

    n = 1 therefore protects nothing -- worth pinning down, it is easy to misread.
    """
    pop = _build(ni=1)
    net = pop.individuals[0]
    _markUsed(net, 0, True)
    pop.callAgeUnusedNodes()          # everUsed=True, unusedSince=0
    _markUsed(net, 0, False)
    pop.callAgeUnusedNodes()          # unusedSince=1

    nd = net.innerNodes[0]
    assert nd.unusedSince == 1
    assert not (nd.everUsed and nd.unusedSince < 1), "grace period 1 must protect nothing"
    assert nd.everUsed and nd.unusedSince < 2, "grace period 2 must still protect it"


def test_pickle_carries_grace_fields():
    import pickle
    pop = _build(ni=1)
    net = pop.individuals[0]
    _markUsed(net, 0, True)
    pop.callAgeUnusedNodes()
    _markUsed(net, 0, False)
    pop.callAgeUnusedNodes()

    back = pickle.loads(pickle.dumps(net))
    assert back.innerNodes[0].everUsed is True
    assert back.innerNodes[0].unusedSince == net.innerNodes[0].unusedSince


@pytest.mark.parametrize("run", ["run_20260912_134234", "run_20260914_025926"])
def test_archived_pickles_still_load(run):
    """Node.__setstate__ must accept the old 11-field tuples."""
    import pickle
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runs", run, "final.pkl")
    if not os.path.exists(path):
        pytest.skip(f"{run} not present")
    net = pickle.load(open(path, "rb"))
    assert len(net.innerNodes) > 0
    assert net.innerNodes[0].unusedSince == 0     # default for pre-grace-period pickles
