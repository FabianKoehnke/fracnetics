"""Hall of fame (fracnetics.hall_of_fame): elite re-evaluation with a carried-over champion.

A fake evaluation gives every network a fixed quality plus a per-case offset, so the
tests control which network wins a round and which one a re-test confirms.
"""

import fracnetics as fn
from fracnetics.hall_of_fame import ChampionMemory, HallOfFame

POPKW = dict(seed=3, jn=8, jnf=8, pn=4, pnf=4, fractalJudgment=False, useExperience=False,
             nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
MINF = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAXF = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
ROUND = [101, 102, 103, 104]
RETEST = [201, 202, 203, 204, 205, 206]
CONFIRM = [301, 302, 303, 304, 305, 306]


def signature(net):
    return tuple((n.f, tuple(n.edges)) for n in net.innerNodes)


class FakeEvaluation:
    """Score = quality of the network + offset of the case (default 0)."""

    def __init__(self):
        self.quality, self.offset, self.calls = {}, {}, []

    def __call__(self, pop, cases):
        individuals = list(pop.individuals)
        for ind in individuals:
            q = self.quality.get(signature(ind), 0.0)
            ind.caseScores = [q + self.offset.get((signature(ind), c), 0.0) for c in cases]
            ind.fitness = sum(ind.caseScores) / len(cases)
        pop.individuals = individuals
        self.calls.append((len(individuals), list(cases)))


def population(ni=20):
    pop = fn.Population(ni=ni, **POPKW)
    pop.setAllNodeBoundaries(MINF, MAXF)
    individuals = list(pop.individuals)
    for i, ind in enumerate(individuals):      # training fitness: individual i ranks i-th
        ind.fitness = float(-i)
    pop.individuals = individuals
    return pop


class Draws:
    def __init__(self):
        self.calls = 0

    def __call__(self, n):
        self.calls += 1
        return RETEST[:n]


def setup(memory=False, retestCases=6, marginSE=1.0, confirmCases=0):
    pop, ev = population(), FakeEvaluation()
    sig = [signature(ind) for ind in pop.individuals]
    hof = HallOfFame(POPKW, ev, candidates=10, memory=memory, retestCases=retestCases,
                     marginSE=marginSE, confirmCases=confirmCases)
    return pop, ev, sig, hof


# ── Without memory ───────────────────────────────────────────────────────────

def test_round_field_is_the_best_by_training_fitness_and_later_the_champion():
    pop, ev, sig, hof = setup()
    hof.update(pop, ROUND)
    hof.update(pop, ROUND)
    assert ev.calls[0] == (10, ROUND)        # first round: the 10 best, no champion yet
    assert ev.calls[1] == (11, ROUND)        # then the champion joins as 11th


def test_first_round_crowns_the_best_on_the_fresh_cases():
    pop, ev, sig, hof = setup()
    ev.quality[sig[4]] = 50.0                # training rank 5, best on the fresh cases
    r = hof.update(pop, ROUND)
    assert (r.winner, r.held, r.takeover, r.retested) == (4, False, True, False)
    assert signature(hof.champion) == sig[4]
    assert (hof.score, hof.scoreCases) == (50.0, len(ROUND))


def drop(pop, i):
    """Give individual i the worst training fitness, so it leaves the round's field."""
    individuals = list(pop.individuals)
    individuals[i].fitness = -1000.0
    pop.individuals = individuals


def test_a_copy_of_the_champion_among_the_candidates_wins_the_tie():
    # Kept on purpose (bit-identical runs): the earlier field index wins a tie, so the
    # title formally changes hands although the network stays the same
    pop, ev, sig, hof = setup()
    ev.quality[sig[4]] = 50.0
    hof.update(pop, ROUND)
    r = hof.update(pop, ROUND)
    assert (r.winner, r.held) == (4, False)
    assert signature(hof.champion) == sig[4]


def test_champion_defends_and_a_better_round_winner_dethrones_it():
    pop, ev, sig, hof = setup()
    ev.quality[sig[4]] = 50.0
    hof.update(pop, ROUND)
    drop(pop, 4)                             # only the hall of fame still holds it
    r = hof.update(pop, ROUND)
    assert (r.winner, r.held) == (10, True)  # the champion sits at the end of the field
    ev.quality[sig[2]] = 60.0
    r = hof.update(pop, ROUND)
    assert (r.winner, r.held, r.takeover) == (2, False, True)
    assert signature(hof.champion) == sig[2]


def test_place_elites_fills_the_elite_slots_in_round_order():
    pop, ev, sig, hof = setup()
    ev.quality[sig[7]], ev.quality[sig[3]] = 50.0, 40.0
    hof.update(pop, ROUND)
    pop.indicesElite = [15, 16]
    hof.placeElites(pop)
    assert signature(pop.individuals[15]) == sig[7]
    assert signature(pop.individuals[16]) == sig[3]


# ── With memory ──────────────────────────────────────────────────────────────

def test_memory_starts_with_the_retest_of_the_first_champion():
    pop, ev, sig, hof = setup(memory=True)
    draws = Draws()
    ev.quality[sig[4]] = 50.0
    r = hof.update(pop, ROUND, draws)
    assert (r.winner, r.retested, r.takeover) == (4, True, True)
    assert draws.calls == 1 and ev.calls[1] == (1, RETEST)
    assert (hof.scoreCases, hof.score) == (len(RETEST), 50.0)


def test_memory_draws_no_retest_while_the_champion_tops_the_round():
    pop, ev, sig, hof = setup(memory=True)
    draws = Draws()
    ev.quality[sig[4]] = 50.0
    hof.update(pop, ROUND, draws)
    r = hof.update(pop, ROUND, draws)
    assert (r.winner, r.held, r.retested) == (10, True, False)
    assert draws.calls == 1                  # only the first champion's re-test
    assert hof.scoreCases == len(RETEST) + len(ROUND)
    assert hof.log[-1] == (0, 0, len(RETEST) + len(ROUND), 50.0)


def test_memory_lucky_round_winner_is_not_crowned():
    pop, ev, sig, hof = setup(memory=True)
    draws = Draws()
    ev.quality[sig[4]] = 50.0
    hof.update(pop, ROUND, draws)
    for c in ROUND:                          # sig[2] is lucky on the round, average elsewhere
        ev.offset[(sig[2], c)] = 100.0
    r = hof.update(pop, ROUND, draws)
    assert (r.winner, r.held, r.retested, r.takeover) == (10, True, True, False)
    assert signature(hof.champion) == sig[4]
    assert hof.scoreCases == len(RETEST) + len(ROUND)


def test_memory_clearly_better_challenger_takes_over_and_memory_restarts():
    pop, ev, sig, hof = setup(memory=True)
    draws = Draws()
    ev.quality[sig[4]] = 50.0
    hof.update(pop, ROUND, draws)
    ev.quality[sig[2]] = 90.0
    r = hof.update(pop, ROUND, draws)
    assert (r.winner, r.held, r.retested, r.takeover) == (2, False, True, True)
    assert (hof.scoreCases, hof.score) == (len(RETEST), 90.0)
    assert signature(hof.champion) == sig[2]


def test_confirmation_starts_the_memory_of_a_new_champion_without_the_lucky_retest():
    pop, ev, sig, hof = setup(memory=True, confirmCases=6)
    ev.quality[sig[4]] = 50.0
    hof.update(pop, ROUND, lambda n: RETEST[:n])     # first champion: no confirmation needed
    assert hof.scoreCases == len(RETEST)
    ev.quality[sig[2]] = 60.0
    for c in RETEST:                                 # sig[2] is lucky on the deciding re-test
        ev.offset[(sig[2], c)] = 40.0
    draws = []
    r = hof.update(pop, ROUND, lambda n: draws.append(n) or (RETEST if len(draws) == 1 else CONFIRM)[:n])
    assert (r.winner, r.takeover) == (2, True) and draws == [6, 6]
    assert (hof.scoreCases, hof.score) == (len(CONFIRM), 60.0)   # 100 with the re-test


def test_without_confirmation_the_memory_keeps_the_retest():
    pop, ev, sig, hof = setup(memory=True)
    ev.quality[sig[4]] = 50.0
    hof.update(pop, ROUND, lambda n: RETEST[:n])
    ev.quality[sig[2]] = 60.0
    for c in RETEST:
        ev.offset[(sig[2], c)] = 40.0
    hof.update(pop, ROUND, lambda n: RETEST[:n])
    assert hof.score == 100.0


# ── ChampionMemory ───────────────────────────────────────────────────────────

def test_without_champion_every_challenger_wins():
    assert ChampionMemory().challengerWins([-50.0, -60.0])


def test_reset_replaces_and_record_extends_the_memory():
    memory = ChampionMemory()
    memory.reset([1.0, 2.0])
    memory.record([3.0])
    assert memory.scores == [1.0, 2.0, 3.0] and memory.n == 3 and memory.mean == 2.0
    memory.reset([10.0])
    assert memory.scores == [10.0]


def test_equal_challenger_does_not_take_over():
    memory = ChampionMemory(marginSE=1.0)
    memory.reset([190.0, 210.0] * 100)
    assert not memory.challengerWins([memory.mean] * 100)


def test_the_margin_decides_a_small_lead():
    champion = [200.0, 210.0, 190.0, 205.0, 195.0] * 40       # mean 200, 200 scores
    challenger = [208.0, 218.0, 198.0, 213.0, 203.0] * 20     # mean 208, ~8 SE ahead
    lenient, strict = ChampionMemory(marginSE=0.0), ChampionMemory(marginSE=10.0)
    lenient.reset(champion)
    strict.reset(champion)
    assert lenient.challengerWins(challenger)
    assert not strict.challengerWins(challenger)
