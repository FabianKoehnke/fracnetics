import fracnetics as fn
import gymnasium as gym
import statistics
import time

start = time.perf_counter()
seed = 42
env = gym.make("CartPole-v1")
# initializing population
pop = fn.Population(
    seed=seed,
    ni=10,
    jn=1,
    jnf=4,
    pn=2,
    pnf=2,
    fractalJudgment=False,
    nFeatureValues=[0, 0, 0, 0],
)

minFeatures = [-4.8, -5, -0.418, -10]
maxFeatures = [4.8, 5, 0.418, 10]
pop.setAllNodeBoundaries(minFeatures, maxFeatures)

fitnessProgess = []
for g in range(10):
    pop.gymnasium(
        env,
        dMax=10,
        maxSteps=500,
        maxConsecutiveP=10,
        worstFitness=0,
        seed=seed + g,
    )
    pop.tournamentSelection(2, 1)
    pop.crossover(0.05, "uniform")
    pop.callAddDelNodes(minFeatures, maxFeatures, 0.3)
    pop.callEdgeMutation(0.03, 0.03, True, True)
    maxFitness = pop.bestFit
    print(maxFitness)
    fitnessProgess.append(maxFitness)

pop.individuals[-1].fitness
print(f"Start Node: {pop.individuals[-1].startNode.edges}")
for node in pop.individuals[-1].innerNodes:
    print(
        f"Type: {node.type} | Function: {node.f} Edges: {node.edges} | Boundaries: {node.boundaries}"
    )

print(f"Done in: {round(time.perf_counter()-start,2)}")
