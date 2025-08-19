import fracnetics 
import sys

def test_population_init():
    try:
        seed = 42
        ni = 10
        jn = 5
        jnf = 3
        pn = 2
        pnf = 1
        fractal_judgment = True

        pop = fracnetics.Population(
            seed=seed,
            ni=ni,
            jn=jn,
            jnf=jnf,
            pn=pn,
            pnf=pnf,
            fractalJudgment=fractal_judgment
        )

        pop.ni
        pop.meanFitness
        pop.individuals[0]

        print("✅ initializing population")

    except Exception as e:
        print("❌ error initializing the population:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    test_population_init()

