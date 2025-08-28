import fracnetics as fn 
import gymnasium as gym
import statistics
import time 

start = time.perf_counter()
env = gym.make("CartPole-v1")
# initializing population
pop = fn.Population(
    seed=42,
    ni=500,
    jn=1,
    jnf=4,
    pn=2,
    pnf=2,
    fractalJudgment=False
)

minFeatures = [-4.8,-5,-0.418,-10] 
maxFeatures = [4.8,5,0.418,10] 
pop.setAllNodeBoundaries(minFeatures,maxFeatures)

fitnessProgess = []
for g in range(200):
  pop.callFitness(
          X=[[1,2,3],[1,2,3]],
          y=[1,2,3],
          dMax=10,
          penalty=1,
          type="gymnasium",
          maxConsecutiveP=5,
          env=env,
          steps=500)
  pop.tournamentSelection(2,1)
  pop.callEdgeMutation(0.03, 0.03)
  pop.crossover(0.05)
  pop.callAddDelNodes(minFeatures,maxFeatures)
  maxFitness = pop.bestFit
  print(maxFitness)
  fitnessProgess.append(maxFitness)
  #env.render()

pop.individuals[-1].fitness
print(f"Start Node: {pop.individuals[-1].startNode.edges}")
for node in pop.individuals[-1].innerNodes:
  print(f"Type: {node.type} | Function: {node.f} Edges: {node.edges} | Boundaries: {node.boundaries}")


print("Validation")
env = gym.make("CartPole-v1", render_mode="human")
validationResults = []
for v in range(10):
    pop.callFitness(
          X=[[1,2,3],[1,2,3]],
          y=[1,2,3],
          dMax=10,
          penalty=1,
          type="gymnasium",
          maxConsecutiveP=5,
          env=env,
          steps=500)
    validationResults.append(pop.bestFit)
print(statistics.mean(validationResults))
print(f"Done in: {round(time.perf_counter()-start,2)}")
