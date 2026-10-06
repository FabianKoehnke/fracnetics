"""Analysis primitives for the seed-noise study.

Everything here operates on one reward matrix R[i, s] = reward of individual i on
seed s. The matrix is measured once per checkpoint (see seed_noise_study.py); all
statistics below are then derived from it by resampling, so no extra episodes are
needed to answer "what would have happened with a batch of size S?".
"""

import numpy as np


# Gymnasium awards +100 on a safe landing and -100 on a crash; Network.hpp uses the
# same constant to decide whether an episode counts as a successful landing.
LANDING_SUCCESS_THRESHOLD = 100.0


def rankdata(x):
    """Average ranks, ties shared -- avoids a scipy dependency."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average the ranks inside each group of equal values
    sortedX = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sortedX[j + 1] == sortedX[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def varianceComponents(R):
    """Split the reward matrix into individual, seed and interaction variance.

    Model: R[i,s] = mu + a_i + b_s + e_is, estimated by a two-way ANOVA without
    replication. The three components answer three different questions:

    * varIndividual -- real, reproducible skill differences. This is the signal.
    * varSeed       -- "this batch was easy/hard", identical for every individual.
                       Because the whole population shares one seed batch, this term
                       cancels out of every within-generation comparison. It only
                       moves the reported fitness curve up and down between
                       generations.
    * varInteraction -- "this individual happens to like these seeds". This is the
                       term that actually corrupts selection.
    """
    R = np.asarray(R, dtype=float)
    n, m = R.shape
    if n < 2 or m < 2:
        raise ValueError("variance decomposition needs at least 2 individuals and 2 seeds")

    grand = R.mean()
    rowMeans = R.mean(axis=1)
    colMeans = R.mean(axis=0)
    a = rowMeans - grand
    b = colMeans - grand
    resid = R - rowMeans[:, None] - colMeans[None, :] + grand

    msIndividual = m * np.sum(a ** 2) / (n - 1)
    msSeed = n * np.sum(b ** 2) / (m - 1)
    msResidual = np.sum(resid ** 2) / ((n - 1) * (m - 1))

    varInteraction = float(msResidual)
    varIndividual = float(max(0.0, (msIndividual - msResidual) / m))
    varSeed = float(max(0.0, (msSeed - msResidual) / n))
    total = varIndividual + varSeed + varInteraction

    return {
        "varIndividual": varIndividual,
        "varSeed": varSeed,
        "varInteraction": varInteraction,
        "sdIndividual": float(np.sqrt(varIndividual)),
        "sdSeed": float(np.sqrt(varSeed)),
        "sdInteraction": float(np.sqrt(varInteraction)),
        "shareIndividual": float(varIndividual / total) if total > 0 else float("nan"),
        "shareSeed": float(varSeed / total) if total > 0 else float("nan"),
        "shareInteraction": float(varInteraction / total) if total > 0 else float("nan"),
        "nIndividuals": int(n),
        "nSeeds": int(m),
    }


def predictedReliability(components, batchSize):
    """Spearman-Brown style prediction of the test-retest correlation.

    The seed component drops out: every individual sees the same batch, so a hard
    batch shifts all scores by the same amount and leaves the ranking untouched.
    """
    varA = components["varIndividual"]
    varE = components["varInteraction"]
    if varA <= 0:
        return 0.0
    return float(batchSize * varA / (batchSize * varA + varE))


def requiredBatchSize(components, targetReliability):
    """Number of seeds needed to reach a given test-retest reliability."""
    varA = components["varIndividual"]
    varE = components["varInteraction"]
    if varA <= 0:
        return float("inf")
    r = targetReliability
    return float(r * varE / ((1.0 - r) * varA))


def predictedCurveNoise(components, batchSize):
    """Standard deviation of the *reported population mean* across generations.

    This is pure measurement noise: the same population re-measured on a fresh seed
    batch produces a mean that scatters by this much, without anything having
    changed. Dominated by the seed component, which is exactly the term that never
    affects selection -- it only makes the fitness curve wobble.
    """
    n = components["nIndividuals"]
    varB = components["varSeed"]
    varE = components["varInteraction"]
    return float(np.sqrt(varB / batchSize + varE / (n * batchSize)))


def splitStatistics(R, batchSize, repeats, eliteSize, rng):
    """Test-retest and selection quality for one batch size.

    Per repetition the seeds are shuffled and split into batch A, batch B and (for
    the ground truth) the remaining held-out seeds. Ground truth is always measured
    on seeds that batch A never saw, so the winner's curse is not hidden by reusing
    the same episodes.

    Batch size must stay well below half the measured pool. Two disjoint batches
    drawn from a finite pool are negatively dependent -- at S = M/2 they are exact
    complements, so a lucky seed in A is necessarily missing from B and the two
    scores become perfectly anti-correlated once real skill differences are small.
    That artefact is invisible in the correlations (real skill dominates them) but it
    wrecks the elite-overlap statistic, which lives exactly in the region where skill
    differences are negligible. Callers should keep 3 * batchSize <= M.
    """
    R = np.asarray(R, dtype=float)
    n, m = R.shape
    if 2 * batchSize > m:
        return None

    out = {k: [] for k in (
        "pearson", "spearman", "eliteOverlapAB", "eliteOverlapTruth",
        "bestRegret", "eliteRegret", "winnersCurse", "seRankShift")}

    for _ in range(repeats):
        perm = rng.permutation(m)
        idxA = perm[:batchSize]
        idxB = perm[batchSize:2 * batchSize]
        idxHeldOut = perm[batchSize:]  # everything batch A did not see

        scoreA = R[:, idxA].mean(axis=1)
        scoreB = R[:, idxB].mean(axis=1)
        # Ground truth for this repetition: every seed batch A did not see. Using the
        # full matrix instead would include A's own lucky seeds and quietly flatter
        # the individual that A crowned.
        trueScore = R[:, idxHeldOut].mean(axis=1)
        trueOrder = np.argsort(-trueScore)
        trueRank = rankdata(-trueScore)

        out["pearson"].append(pearson(scoreA, scoreB))
        out["spearman"].append(spearman(scoreA, scoreB))

        eliteA = set(np.argsort(-scoreA)[:eliteSize].tolist())
        eliteB = set(np.argsort(-scoreB)[:eliteSize].tolist())
        eliteTrue = set(trueOrder[:eliteSize].tolist())
        out["eliteOverlapAB"].append(len(eliteA & eliteB) / eliteSize)
        out["eliteOverlapTruth"].append(len(eliteA & eliteTrue) / eliteSize)

        bestA = int(np.argmax(scoreA))
        out["bestRegret"].append(float(trueScore[trueOrder[0]] - trueScore[bestA]))
        out["eliteRegret"].append(float(
            trueScore[trueOrder[:eliteSize]].mean() - trueScore[list(eliteA)].mean()))
        # Optimism of the reported champion score: what the batch says minus what
        # unseen seeds say about the very same individual.
        out["winnersCurse"].append(float(scoreA[bestA] - trueScore[bestA]))
        # How far a batch moves an individual in the ranking, in rank positions.
        out["seRankShift"].append(float(np.abs(rankdata(-scoreA) - trueRank).mean()))

    summary = {"batchSize": int(batchSize), "repeats": int(repeats)}
    for k, v in out.items():
        v = np.asarray(v, dtype=float)
        summary[k] = float(np.nanmean(v))
        summary[k + "Sd"] = float(np.nanstd(v))
    return summary


def batchMeanScatter(R, batchSize, repeats, rng):
    """Empirical scatter of the reported population mean and best fitness."""
    R = np.asarray(R, dtype=float)
    n, m = R.shape
    means, bests = [], []
    for _ in range(repeats):
        idx = rng.choice(m, size=batchSize, replace=False)
        score = R[:, idx].mean(axis=1)
        means.append(score.mean())
        bests.append(score.max())
    return {
        "batchSize": int(batchSize),
        "populationMeanSd": float(np.std(means)),
        "bestFitnessSd": float(np.std(bests)),
        "bestFitnessMean": float(np.mean(bests)),
    }


def rewardShape(R):
    """Landing rate, bimodality and how crowded the top of the population is."""
    R = np.asarray(R, dtype=float)
    landed = R > LANDING_SUCCESS_THRESHOLD
    perIndividualSd = R.std(axis=1, ddof=1) if R.shape[1] > 1 else np.zeros(R.shape[0])
    trueScore = R.mean(axis=1)
    order = np.argsort(-trueScore)
    top10 = trueScore[order[:min(10, len(order))]]
    return {
        # Exact duplicates are common after selection; they make the population
        # smaller than it looks and crowd the top with indistinguishable candidates.
        "uniqueIndividuals": int(len(np.unique(np.round(R, 6), axis=0))),
        "topSpreadSd": float(top10.std(ddof=1)) if len(top10) > 1 else 0.0,
        "gapBest": float(trueScore[order[0]] - trueScore[order[min(4, len(order) - 1)]]),
        "episodes": int(R.size),
        "landingRate": float(landed.mean()),
        "rewardMean": float(R.mean()),
        "rewardSd": float(R.std(ddof=1)),
        "perIndividualSdMean": float(perIndividualSd.mean()),
        "perIndividualSdMedian": float(np.median(perIndividualSd)),
        "perSeedMeanSd": float(R.mean(axis=0).std(ddof=1)),
        "perSeedMeanMin": float(R.mean(axis=0).min()),
        "perSeedMeanMax": float(R.mean(axis=0).max()),
    }


def requiredBatchSizeFromMeasured(reliabilityAtS, batchSize, targetReliability):
    """Extrapolate the seed budget from an OBSERVED reliability.

    Uses the same Spearman-Brown form as requiredBatchSize(), but calibrated on a
    measured correlation instead of the variance model. Applied to the measured rank
    correlation this is the pessimistic (and for selection, the honest) estimate,
    because the additive model silently assumes every individual carries the same
    amount of noise.
    """
    r = reliabilityAtS
    if not (0.0 < r < 1.0):
        return float("inf")
    noiseToSignal = batchSize * (1.0 - r) / r          # varE / varA
    t = targetReliability
    return float(t * noiseToSignal / (1.0 - t))
