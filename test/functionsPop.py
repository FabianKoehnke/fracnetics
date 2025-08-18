import fracnetics 
import sys

def test_population_function():

    pop = fracnetics.Population(
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

if __name__ == "__main__":
    test_population_function()

