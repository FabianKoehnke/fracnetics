import fracnetics as fn 
import gymnasium as gym
import statistics
import time 
from gymnasium.wrappers import RecordVideo

start = time.perf_counter()
env = gym.make("LunarLander-v3")
# initializing population
pop = fn.Population(
    seed=42,
    ni=10_000,
    jn=1,
    jnf=8,
    pn=4,
    pnf=4,
    fractalJudgment=False
)

minFeatures = [ -2.5, -2.5, -10, -10, -6.2831855, -10, -0, -0 ] 
maxFeatures = [ 2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1 ] 
pop.setAllNodeBoundaries(minFeatures,maxFeatures)

fitnessProgess = []
for g in range(100):
  pop.gymnasium(
          env,
          dMax=10,
          penalty=2,
          maxSteps=500,
          maxConsecutiveP=10,
          worstFitness=-200
          )
  pop.tournamentSelection(2,1)
  pop.callEdgeMutation(0.03, 0.03)
  pop.crossover(0.05)
  pop.callBoundaryMutationUniform(0.05)
  #pop.callBoundaryMutationFractal(0.05, minFeatures, maxFeatures)
  pop.callAddDelNodes(minFeatures,maxFeatures)
  maxFitness = pop.bestFit
  print(f"Generation: {g} | maxFitness: {maxFitness} | meanFitness: {pop.meanFitness}")
  fitnessProgess.append(maxFitness)
  #env.render()

pop.individuals[-1].fitness
print(f"Start Node: {pop.individuals[-1].startNode.edges}")
for node in pop.individuals[-1].innerNodes:
  print(f"Type: {node.type} | Function: {node.f} Edges: {node.edges} | Boundaries: {node.boundaries}")


print("Validation")
env = gym.make("LunarLander-v3")
validationResults = []
for v in range(10):
    pop.gymnasium(
          env,
          dMax=10,
          penalty=2,
          maxSteps=500,
          maxConsecutiveP=10,
          worstFitness=-200
          )
    validationResults.append(pop.bestFit)
print(statistics.mean(validationResults))

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos")
pop.individuals = [pop.individuals[-1]]
pop.gymnasium(
      env,
      dMax=10,
      penalty=1,
      maxSteps=500,
      maxConsecutiveP=5,
      worstFitness=-200)

env.close()
