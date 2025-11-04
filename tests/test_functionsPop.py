import fracnetics as fn
import pandas as pd 
import sys
import gymnasium as gym

def test_population_function():


    # initializing population
    pop = fn.Population(
        seed=42,
        ni=10,
        jn=5,
        jnf=3,
        pn=3,
        pnf=2,
        fractalJudgment=True
    )
    
    try:
        minF= [-4.8, -5, -0.418, -10]
        maxF= [4.8, 5, 0.418, 10]
        pop.setAllNodeBoundaries(minF, maxF)

    except Exception as e:
        print("❌ error in pop.setAllNodeBoundaries()")
        print(e)
        sys.exit(1)

    print("✅ pop.setAllNodeBoundaries()") 

    try:
        env = gym.make("CartPole-v1")
        fit1 = pop.individuals[0].fitness
        pop.gymnasium(
            env,
            dMax=10,
            maxSteps=500,
            maxConsecutiveP=10,
            worstFitness=0,
            seed=123
        )
        fit2 = pop.individuals[0].fitness
        if fit1 == fit2:
            print("no difference of fitnes after applying callFitness()")
            sys.exit(1)
    except Exception as e:
        print("❌ error in pop.callFitness()")
        print(e)
        sys.exit(1)

    print("✅ pop.callFitness()") 

    try:
        pop.tournamentSelection(2,1)
    except Exception as e:
        print("❌ error in pop.tournamentSelection()")
        print(e)
        sys.exit(1)

    print("✅ pop.tournamentSelection()") 
    
    try:
        edges = []
        for i in range(len(pop.individuals[0].innerNodes)):
            edges.append(pop.individuals[0].innerNodes[i].edges) 
        pop.callEdgeMutation(0.5,0)
        counter = 0
        for i in range(len(pop.individuals[0].innerNodes)):
            if edges[i] != pop.individuals[0].innerNodes[i].edges:
                counter += 1
        if counter == 0:
            print("❌ no changes by callEdgeMutation()!")
            sys.exit(1)
 
    except Exception as e:
        print("❌ error in pop.callEdgeMutation()")
        print(e)
        sys.exit(1)

    print("✅ pop.callEdgeMutation()") 

    def testBoundaryMutation(testFunc, name):
        try:
            boundaries = []
            for i in range(len(pop.individuals[0].innerNodes)):
                boundaries.append(pop.individuals[0].innerNodes[i].boundaries) 
            testFunc()
            counter = 0
            for i in range(len(pop.individuals[0].innerNodes)):
                if boundaries[i] != pop.individuals[0].innerNodes[i].boundaries:
                    counter += 1
            if counter == 0:
                print(f"❌ no changes by {name}!")
                sys.exit(1)

        except Exception as e:
            print(f"❌ error in pop.{name}")
            print(e)
            sys.exit(1)

        print(f"✅ pop.{name}") 

    testBoundaryMutation(lambda: pop.callBoundaryMutationUniform(1), "callBoundaryMutationUniform()")
    testBoundaryMutation(lambda: pop.callBoundaryMutationNormal(0.5,0.2), "callBoundaryMutationNormal()")
    testBoundaryMutation(lambda: pop.callBoundaryMutationNetworkSizeDependingSigma(0.5,0.2), "callBoundaryMutationNetworkSizeDependingSigma()")
    testBoundaryMutation(lambda: pop.callBoundaryMutationEdgeSizeDependingSigma(0.5,0.2), "callBoundaryMutationEdgeSizeDependingSigma()")
    testBoundaryMutation(lambda: pop.callBoundaryMutationFractal(0.8,minF,maxF), "callBoundaryMutationFractal()")
    
    try:
        edges = []
        for i in range(len(pop.individuals[0].innerNodes)):
            edges.append(pop.individuals[0].innerNodes[i].edges) 
        pop.crossover(0.5)
        counter = 0
        for i in range(len(pop.individuals[0].innerNodes)):
            if edges[i] != pop.individuals[0].innerNodes[i].edges:
                counter += 1
        if counter == 0:
            print("❌ no changes by crossover()!")
            sys.exit(1)

    except Exception as e:
        print(f"❌ error in pop.crossover()")
        print(e)
        sys.exit(1)

    try:
        pop.callAddDelNodes(minF,maxF)
    except Exception as e:
        print(f"❌ error in pop.callAddDelNodes()")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    test_population_function()

