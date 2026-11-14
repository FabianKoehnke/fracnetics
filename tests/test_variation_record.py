"""Variation record: which operators produced a network (Network.uid, lastVariedBy, lastVariedRound).

The first change of a network in a new evaluation round gives it a new uid and starts a new
record; copies keep both. Used to trace which operator produced a new champion.
"""
import pickle

import fracnetics as fn

POPKW = dict(seed=3, jn=8, jnf=8, pn=4, pnf=4, fractalJudgment=False, useExperience=False,
             nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
MINF = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAXF = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
V = fn.Variation


def population(ni=30, **kw):
    pop = fn.Population(ni=ni, **{**POPKW, **kw})
    pop.setAllNodeBoundaries(MINF, MAXF)
    pop.checkOrder = False
    return pop


def record(pop):
    return [(n.uid, n.lastVariedBy, n.lastVariedRound) for n in pop.individuals]


def test_new_networks_have_distinct_uids_and_no_record():
    pop = population()
    uids = [n.uid for n in pop.individuals]
    assert len(set(uids)) == len(uids) and min(uids) >= 1
    assert all(by == 0 and r == -1 for _, by, r in record(pop))


def test_changes_are_recorded_per_round():
    pop = population()
    pop.markEvaluated()                                   # round 1
    before = record(pop)
    pop.callEdgeMutation(probInnerNodes=0, probStartNode=0, k=2.0)
    afterEdge = record(pop)
    changed = [i for i, (a, b) in enumerate(zip(before, afterEdge)) if a != b]
    assert changed
    for i, (uid, by, r) in enumerate(afterEdge):
        if i in changed:
            assert by == int(V.EdgeMutation) and r == pop.evaluationRound and uid > 30
        else:
            assert (uid, by, r) == before[i]
    pop.callBoundaryMutationUniform(probability=1.0)      # same round: bits add up, uid stays
    for (uid0, by0, _), (uid1, by1, r1) in zip(afterEdge, record(pop)):
        if by0:
            assert uid1 == uid0 and by1 == by0 | int(V.BoundaryUniform)
        else:
            assert by1 == int(V.BoundaryUniform) and r1 == pop.evaluationRound
    pop.markEvaluated()                                   # round 2: a new change starts over
    pop.callBoundaryMutationNormal(probability=1.0, sigma=0.2, justUsedNodes=False)
    assert all(by == int(V.BoundaryNormal) and r == pop.evaluationRound for _, by, r in record(pop))


def test_each_operator_sets_its_bit():
    pop = population()
    pop.markEvaluated()
    pop.callAddDelNodes(MINF, MAXF, junk=1.0, noElite=False)
    pop.markEvaluated()
    pop.crossover(probability=1.0, type="semantic", nodeExchangeRate=1.0)
    bits = {by for _, by, r in record(pop) if r == pop.evaluationRound}
    assert int(V.Crossover) in bits
    fractal = population(fractalJudgment=True)
    fractal.markEvaluated()
    fractal.callBoundaryMutationFractal(probability=1.0, minF=MINF, maxF=MAXF)
    assert all(by == int(V.BoundaryFractal) for _, by, _r in record(fractal))


def test_copies_and_pickles_keep_the_record():
    pop = population()
    pop.markEvaluated()
    pop.callBoundaryMutationUniform(probability=1.0)
    other = fn.Population(ni=1, **POPKW)
    other.individuals = [pop.individuals[0]]
    assert record(other)[0] == record(pop)[0]
    net = pickle.loads(pickle.dumps(pop.individuals[0]))
    assert (net.uid, net.lastVariedBy, net.lastVariedRound) == record(pop)[0]
    again = pickle.loads(pickle.dumps(pop))             # new uids continue after the highest
    again.checkOrder = False
    top = max(n.uid for n in again.individuals)
    again.markEvaluated()
    again.callBoundaryMutationUniform(probability=1.0)
    assert min(n.uid for n in again.individuals) > top
