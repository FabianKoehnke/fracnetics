"""crossover(type="randomWidth"): the elite only donates, the recipient is the one marked.

Two bugs this pins down: when the elite was parent2, the donor instead of the recipient got
Node::generationReceived, so crossoverProtection guarded the wrong network; and when the
elite's sub-network had a single node, neither elite branch applied and the pair fell
through to the bidirectional swap, which rewrote the elite.
"""

import gymnasium as gym

import fracnetics as fn

MIN_F = [-4.8, -5, -0.418, -10]
MAX_F = [4.8, 5, 0.418, 10]


def snapshot(net):
    nodes = [(n.type, n.f, list(n.edges), list(n.boundaries), n.generationReceived)
             for n in net.innerNodes]
    return nodes, list(net.startNode.edges)


def test_elite_only_donates():
    env = gym.make("CartPole-v1")
    for seed in range(3):
        pop = fn.Population(seed=seed, ni=20, jn=6, jnf=4, pn=3, pnf=2,
                            fractalJudgment=False, nFeatureValues=[0, 0, 0, 0])
        pop.setAllNodeBoundaries(MIN_F, MAX_F)
        for g in range(15):
            pop.gymnasium(env, seeds=[g], dMax=10, maxSteps=60, maxConsecutiveP=5)
            pop.tournamentSelection(N=3, E=4)
            elites = {i: snapshot(pop.individuals[i]) for i in pop.indicesElite}
            crossoversBefore = [ind.nCrossovers for ind in pop.individuals]

            pop.crossover(probability=1.0, type="randomWidth", currentGeneration=100 + g)

            for i, before in elites.items():
                assert snapshot(pop.individuals[i]) == before, f"seed {seed}, generation {g}: elite {i} changed"
            # Every network that received genes carries the mark of this generation
            for i, ind in enumerate(pop.individuals):
                if i not in elites and ind.nCrossovers > crossoversBefore[i]:
                    assert any(n.generationReceived == 100 + g for n in ind.innerNodes)
    env.close()
