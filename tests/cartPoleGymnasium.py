import fracnetics as fn 
import gymnasium as gym
import statistics
import time 
from gymnasium.wrappers import RecordVideo

start = time.perf_counter()
seed=42
env = gym.make("CartPole-v1")
# initializing population
pop = fn.Population(
    seed=seed,
    ni=100,
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
for g in range(250):
  pop.gymnasium(
          env,
          dMax=10,
          penalty=2,
          maxSteps=500,
          maxConsecutiveP=10,
          worstFitness=0,
          seed=seed+g)
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
env = gym.make("CartPole-v1")
validationResults = []
for v in range(10):
    pop.gymnasium(
          env,
          dMax=10,
          penalty=2,
          maxSteps=500,
          maxConsecutiveP=10,
          worstFitness=0)
    validationResults.append(pop.bestFit)
print(statistics.mean(validationResults))

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix="cartpole")

pop.individuals = [pop.individuals[-1]]
pop.gymnasium(
      env,
      dMax=10,
      penalty=1,
      maxSteps=500,
      maxConsecutiveP=5,
      worstFitness=0)
env.close()
print(f"Done in: {round(time.perf_counter()-start,2)}")
