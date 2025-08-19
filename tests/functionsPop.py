import fracnetics as fn
import pandas as pd 
import sys

def test_population_function():

    data = pd.read_csv("../fracnetics/data/IRIS.csv")
    X = data.iloc[:,1:5].values
    y = data.iloc[:,5].values

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
        pop.setAllNodeBoundaries([1,2,3],[0,0,0])
    except Exception as e:
        print("❌ error in pop.setAllNodeBoundaries()")
        print(e)
        sys.exit(1)

    print("✅ pop.setAllNodeBoundaries()") 

    try:
        pop.callFitness(X,y,10,2,"accuracy",1)
    except Exception as e:
        print("❌ error in pop.callFitness()")
        print(e)
        sys.exit(1)

    print("✅ pop.callFitness()") 

if __name__ == "__main__":
    test_population_function()

