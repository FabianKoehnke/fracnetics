"""Hall of fame: elite re-evaluation on fresh cases with a carried-over champion.

When single evaluations are noisy (stochastic episodes), the training fitness on a few
cases ranks the population poorly. Each generation the best `candidates` individuals by
training fitness and the current champion are therefore re-scored on the same fresh
cases. The round winner becomes the champion, and the round order fills the elite slots
after the selection (see HallOfFame.placeElites()).

With `memory=True` the champion keeps all its round scores. A round winner is re-tested
on further fresh cases and takes over only if its re-test mean beats the champion's
accumulated mean by `marginSE` combined standard errors. The round it won never enters its
memory: it was picked as the best of several on exactly those cases (winner's curse). The
deciding re-test is biased as well -- a takeover happens only when it came out high; in
LunarLander a new champion's memory then fell by 18 points on average over its reign, so
better challengers were held off and a late champion was overrated. With `confirmCases`
the memory of a new champion starts with an independent confirmation on further fresh
cases instead; the re-test only decides.

Problem-independent: the caller passes the evaluation and draws the fresh cases, so the
order of its random draws stays under its control.

    hof = HallOfFame(popKwargs, evaluate, candidates=10, memory=True)
    for g in range(generations):
        evaluate(pop, trainingCases)
        result = hof.update(pop, freshCases, drawRetestCases=lambda n: drawCases(n))
        pop.lexicaseSelection(E=2, ...)
        hof.placeElites(pop)
"""

import math
import statistics
from dataclasses import dataclass

from ._core import Population


class ChampionMemory:
    """Accumulated fresh-case scores of the current champion and the takeover rule.

    Args:
        marginSE: How many combined standard errors the challenger's re-test mean must lie
            above the champion's mean (0 = any lead suffices).
    """

    def __init__(self, marginSE=1.0):
        self.marginSE = marginSE
        self.scores = []

    def reset(self, scores):
        """Start the memory of a new champion with its (unbiased) re-test scores."""
        self.scores = list(scores)

    def record(self, scores):
        """Add the champion's scores of another round."""
        self.scores.extend(scores)

    @property
    def n(self):
        return len(self.scores)

    @property
    def mean(self):
        return statistics.fmean(self.scores) if self.scores else float("-inf")

    def challengerWins(self, retestScores):
        """True if the challenger's re-test mean beats the champion's mean by the margin.

        Without a champion (empty memory) every challenger wins.
        """
        if not self.scores:
            return True
        challengerMean = statistics.fmean(retestScores)
        se = math.sqrt(_variance(retestScores) / len(retestScores)
                       + _variance(self.scores) / len(self.scores))
        return challengerMean - self.mean > self.marginSE * se


def _variance(values):
    # A single score carries no spread information
    return statistics.variance(values) if len(values) > 1 else 0.0


@dataclass
class Round:
    """Outcome of one HallOfFame.update().

    Attributes:
        winner: Index of the new champion in the round's field (candidates in training
            order, then the previous champion, if any).
        winnerScore: Its score in this round.
        held: True if the previous champion kept its title. Without memory a tie goes to
            the earlier field index, so a copy of the champion among the candidates takes
            the title (held False) although the network stays the same; with memory a
            challenger must score strictly higher to be re-tested at all.
        retested: True if a round winner was re-tested (memory only).
        takeover: True if the title changed hands (always True in the first round).
    """
    winner: int
    winnerScore: float
    held: bool
    retested: bool
    takeover: bool


class HallOfFame:
    """Champion and elite order from a re-evaluation on fresh cases.

    Args:
        popKwargs: Population constructor arguments; they build the one-individual
            containers the networks are copied into for evaluation.
        evaluate: Callable evaluate(population, cases) that scores population.individuals
            on the cases (fitness and caseScores), e.g. ParallelEvaluator.evaluate.
        candidates: How many individuals, best training fitness first, enter each round.
        memory: Keep the champion's scores across rounds and re-test round winners.
        retestCases: Number of fresh cases a round winner is re-tested on (memory only).
        marginSE: Lead a re-tested challenger needs, in combined standard errors.
        confirmCases: With memory: after a takeover over an existing champion, evaluate the
            new champion on this many further fresh cases and start its memory with them
            instead of with the deciding re-test. 0 keeps the re-test (earlier behaviour).

    Attributes:
        score: The champion's score: its round score, or its accumulated mean with memory.
        scoreCases: Number of cases behind `score`.
        log: With memory, one (retested, takeover, memoryCases, memoryMean) per update().
    """

    def __init__(self, popKwargs, evaluate, candidates=10, memory=False, retestCases=100,
                 marginSE=1.0, confirmCases=0):
        self.evaluate = evaluate
        self.candidates = candidates
        self.retestCases = retestCases
        self.confirmCases = confirmCases
        self.memory = ChampionMemory(marginSE) if memory else None
        container = lambda: Population(**{**popKwargs, "ni": 1})
        self._score = container()       # the round's field, re-scored on the fresh cases
        self._keep = container()        # the same field with its training scores
        self._champion = container()
        self._retest = container() if memory else None
        self.hasChampion = False
        self.score = None
        self.scoreCases = 0
        self.order = []
        self.log = []

    @property
    def champion(self):
        """The current champion (a copy held by the hall of fame)."""
        return self._champion.individuals[0]

    def update(self, pop, cases, drawRetestCases=None):
        """Play one round on `cases` and crown its champion.

        Args:
            pop: The population, already evaluated on its training cases.
            cases: Fresh cases, the same for every member of the round.
            drawRetestCases: Callable n -> n further fresh cases; with memory it is called
                only when a challenger wins the round (and once more for the confirmation of
                a takeover), so unneeded draws never happen.

        Returns:
            Round
        """
        fitness = [ind.fitness for ind in pop.individuals]
        best = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:self.candidates]
        field = [pop.individuals[i] for i in best]
        championIndex = None
        if self.hasChampion:
            championIndex = len(field)
            field.append(self._champion.individuals[0])
        self._keep.individuals = field
        self._score.individuals = field
        self.evaluate(self._score, cases)
        scores = [ind.fitness for ind in self._score.individuals]
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        winner, retested = order[0], False

        if self.memory is not None:
            if championIndex is not None:
                self.memory.record(self._score.individuals[championIndex].caseScores)
            challenger = next(j for j in order if j != championIndex)
            winner = championIndex
            if championIndex is None or scores[challenger] > scores[championIndex]:
                self._retest.individuals = [self._keep.individuals[challenger]]
                self.evaluate(self._retest, drawRetestCases(self.retestCases))
                retestScores = list(self._retest.individuals[0].caseScores)
                retested = True
                if self.memory.challengerWins(retestScores):
                    winner = challenger
                    if self.confirmCases > 0 and championIndex is not None:
                        # The re-test was selected for coming out high; start unbiased
                        self.evaluate(self._retest, drawRetestCases(self.confirmCases))
                        retestScores = list(self._retest.individuals[0].caseScores)
                    self.memory.reset(retestScores)
            order = [winner] + [j for j in order if j != winner]

        held = championIndex is not None and winner == championIndex
        self._champion.individuals = [self._keep.individuals[winner]]
        self.hasChampion = True
        self.order = order
        if self.memory is not None:
            self.score, self.scoreCases = self.memory.mean, self.memory.n
            self.log.append((int(retested), int(not held), self.memory.n, self.memory.mean))
        else:
            self.score, self.scoreCases = scores[winner], len(cases)
        return Round(winner=winner, winnerScore=scores[winner], held=held, retested=retested,
                     takeover=not held)

    def placeElites(self, pop):
        """Put the round's order into pop's elite slots (call after the selection set them)."""
        for slot, j in zip(pop.indicesElite, self.order[:len(pop.indicesElite)]):
            pop.individuals[slot] = self._keep.individuals[j]
