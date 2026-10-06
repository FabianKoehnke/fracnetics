#!/usr/bin/env python
"""Seed-noise study for the LunarLander GNP experiment.

Answers the three questions from the noise discussion, without changing anything
in the evolutionary algorithm itself:

  1. Test-retest reliability -- if the same population is scored on two independent
     seed batches, how much of the ranking survives? This says whether selection is
     picking skill or luck.
  2. How many seeds would be needed -- the reliability curve over batch size plus an
     extrapolation to a target reliability.
  3. Variance decomposition -- how much of the scatter is "this batch was hard"
     (shared by everyone, harms only the reported curve) versus "this individual
     likes these seeds" (harms selection).

Method: run the normal evolution loop for a while, and at configured checkpoint
generations score the *entire* population on a large seed set (default 60 seeds).
That one reward matrix is written to CSV; every statistic is then derived from it by
resampling, so the expensive part happens exactly once per checkpoint.

Note on determinism: with enable_wind=False a LunarLander episode is fully
determined by its seed. Re-running the same seed yields the identical reward, so
there is no measurement noise to average away -- the entire fluctuation studied here
comes from *which* seeds were drawn.

Usage:
    ./venv/bin/python research/seed_noise_study.py --quick          # smoke test
    ./venv/bin/python research/seed_noise_study.py                  # full study
    ./venv/bin/python research/seed_noise_study.py --analyze-only research/results/<run>
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import noise_analysis as na  # noqa: E402


# ── Experiment configuration, mirrored from examples/lunarlander.py ────────────
# Kept identical on purpose: the study is only informative if it measures the same
# population and the same reward regime the real run uses.
POP_SEED = 5
N_INDIVIDUALS = 100
MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
SEED_POOL = 10_000          # drawSeeds() samples from range(0, 10_000)
D_MAX = 10
MAX_CONSECUTIVE_P = 5
WORST_FITNESS = -500
CURRICULUM_LEVEL = 1.0
ABSOLUTE_IMPULSE_CURRICULUM = False
ELITE_SIZE = 4              # lexicaseSelection(E=4)
LEXICASE_TYPE = "seeds"     # overridden by --lexicase-type


def buildPopulation(fn):
    pop = fn.Population(
        seed=POP_SEED,
        ni=N_INDIVIDUALS,
        jn=8,
        jnf=8,
        pn=4,
        pnf=4,
        fractalJudgment=False,
        useExperience=False,
        nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2],
    )
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    return pop


def evaluate(pop, env, seeds, maxSteps, potential=False):
    """Score every individual on every seed; returns the reward matrix."""
    pop.gymnasiumMultiSeed(
        env,
        dMax=D_MAX,
        maxSteps=maxSteps,
        maxConsecutiveP=MAX_CONSECUTIVE_P,
        worstFitness=WORST_FITNESS,
        seeds=list(seeds),
        curriculumLevel=CURRICULUM_LEVEL,
        absoluteImpulseCurriculum=ABSOLUTE_IMPULSE_CURRICULUM,
        useLineageFitness=False,
        uniformDirectionCurriculum=False,
        survivalMode=False,
        potential=potential,
        landingQuote=False,
    )
    return np.array([list(ind.fitnessValues) for ind in pop.individuals], dtype=float)


def varyPopulation(pop, generation):
    """Selection is done by the caller; this applies the variation operators.

    Same order and parameters as the non-seedSpecialist branch of the training loop
    in examples/lunarlander.py.
    """
    pop.callAddDelNodes(
        MIN_FEATURES, MAX_FEATURES,
        junk=0.1, noElite=True, currentGeneration=generation, crossoverProtection=0,
    )
    pop.callFindTransitionClusters(2)
    pop.crossover(
        probability=0.1, type="innovation", crossoverProtection=0,
        lineagePriorCap=75, currentGeneration=generation,
    )
    pop.callEdgeMutation(probInnerNodes=0, probStartNode=0, justUsedNodes=True, k=1)


# ── Measurement ───────────────────────────────────────────────────────────────

def analyzeMatrix(R, batchSizes, repeats, trainingBatchSize, rng):
    components = na.varianceComponents(R)
    curve, scatter = [], []
    for s in batchSizes:
        stats = na.splitStatistics(R, s, repeats, ELITE_SIZE, rng)
        if stats is not None:
            stats["predictedReliability"] = na.predictedReliability(components, s)
            curve.append(stats)
        sc = na.batchMeanScatter(R, s, repeats, rng)
        sc["predictedPopulationMeanSd"] = na.predictedCurveNoise(components, s)
        scatter.append(sc)

    return {
        "components": components,
        "reliabilityCurve": curve,
        "batchMeanScatter": scatter,
        "rewardShape": na.rewardShape(R),
        "requiredSeeds": {
            f"{t:.2f}": na.requiredBatchSize(components, t) for t in (0.7, 0.8, 0.9, 0.95)
        },
        # Same extrapolation, but anchored on the largest measured rank correlation --
        # this is what selection actually experiences.
        "requiredSeedsRank": ({
            f"{t:.2f}": na.requiredBatchSizeFromMeasured(
                curve[-1]["spearman"], curve[-1]["batchSize"], t)
            for t in (0.7, 0.8, 0.9, 0.95)
        } if curve else {}),
        "trainingBatchSize": trainingBatchSize,
    }


def referenceBatchSize(cp, preferred):
    """Batch size the report zooms in on: the one used in training if measured."""
    sizes = [r["batchSize"] for r in cp["reliabilityCurve"]]
    if not sizes:
        return None
    return preferred if preferred in sizes else max(sizes)


def formatReport(runInfo, checkpoints, trainingCurve):
    """Human-readable summary; the numbers that decide what to fix first."""
    S = runInfo["trainingBatchSize"]
    L = []
    add = L.append
    add("=" * 78)
    add("SEED NOISE STUDY -- LunarLander / GNP")
    add("=" * 78)
    add(f"population            : {runInfo['nIndividuals']} individuals")
    add(f"training batch        : {S} seeds per generation, redrawn every generation")
    add(f"measurement batch     : {runInfo['measureSeeds']} seeds per checkpoint")
    add(f"generations evolved   : {runInfo['generations']}")
    add(f"lexicase test cases   : {runInfo.get('lexicaseType', 'seeds')}")
    add(f"reward mode           : "
        f"{'potential-based shaping (PBRS)' if runInfo.get('potential') else 'raw Gymnasium reward'}")
    add(f"checkpoints           : {runInfo['checkpoints']}")
    add(f"episodes spent        : {runInfo['episodes']:,}")
    add(f"wall time             : {runInfo['wallTimeSeconds'] / 60:.1f} min")
    add("")
    add("A LunarLander episode is deterministic given its seed (enable_wind=False),")
    add("so all scatter below comes from WHICH seeds were drawn, not from replaying.")
    add("")

    for cp in checkpoints:
        g = cp["generation"]
        c = cp["components"]
        shape = cp["rewardShape"]
        ref = referenceBatchSize(cp, S)
        atS = next((r for r in cp["reliabilityCurve"] if r["batchSize"] == ref), None)
        scS = next((r for r in cp["batchMeanScatter"] if r["batchSize"] == ref), None)

        add("-" * 78)
        add(f"CHECKPOINT  generation {g}")
        add("-" * 78)
        add(f"reward over all measured episodes: mean {shape['rewardMean']:8.1f} "
            f"sd {shape['rewardSd']:7.1f}   landing rate {shape['landingRate'] * 100:5.1f}%")
        add(f"scatter within one individual across seeds: sd {shape['perIndividualSdMean']:.1f} "
            f"(median {shape['perIndividualSdMedian']:.1f})")
        add(f"distinct individuals (exact duplicates removed): "
            f"{shape['uniqueIndividuals']} of {c['nIndividuals']}")
        add(f"crowding at the top: spread of the best 10 sd {shape['topSpreadSd']:.1f}, "
            f"gap rank1 -> rank5 {shape['gapBest']:.1f}")
        add("")
        add("[3] WHERE THE SCATTER COMES FROM (two-way variance decomposition)")
        add(f"  real skill differences   sd {c['sdIndividual']:7.1f}   "
            f"{c['shareIndividual'] * 100:5.1f}%   <- the signal")
        add(f"  seed difficulty (shared) sd {c['sdSeed']:7.1f}   "
            f"{c['shareSeed'] * 100:5.1f}%   <- moves the CURVE, not the ranking")
        add(f"  individual x seed        sd {c['sdInteraction']:7.1f}   "
            f"{c['shareInteraction'] * 100:5.1f}%   <- corrupts SELECTION")
        add("")
        signalToNoise = (c["sdIndividual"] / (c["sdInteraction"] / math.sqrt(ref))
                         if c["sdInteraction"] > 0 and ref else float("inf"))
        add(f"  With {ref} seeds the noise on one individual's score is "
            f"{c['sdInteraction'] / math.sqrt(ref):.1f} reward,")
        add(f"  against a true spread between individuals of {c['sdIndividual']:.1f} "
            f"-> signal/noise = {signalToNoise:.2f}")
        add("")

        if atS is not None:
            note = "" if ref == S else f"  [!] {S} not measurable, 2*{S} > {runInfo['measureSeeds']} seeds"
            add(f"[1] TEST-RETEST AT THE BATCH SIZE ACTUALLY USED ({ref} seeds){note}")
            add(f"  score correlation between two batches (Pearson)  : "
                f"{atS['pearson']:.3f}  (+/- {atS['pearsonSd']:.3f})")
            add(f"  rank correlation between two batches (Spearman)  : "
                f"{atS['spearman']:.3f}  (+/- {atS['spearmanSd']:.3f})")
            add(f"  predicted from the variance components           : "
                f"{atS['predictedReliability']:.3f}")
            if atS["pearson"] - atS["spearman"] > 0.1:
                add("  -> Spearman far below Pearson: the scores are reproducible, but many")
                add("     individuals sit so close together that their ORDER is decided by noise.")
            add(f"  elite of {ELITE_SIZE}: same individuals under both batches   : "
                f"{atS['eliteOverlapAB'] * 100:.0f}%  "
                f"({atS['eliteOverlapAB'] * ELITE_SIZE:.1f} of {ELITE_SIZE})")
            add(f"  elite of {ELITE_SIZE}: overlap with the true best {ELITE_SIZE}       : "
                f"{atS['eliteOverlapTruth'] * 100:.0f}%")
            add(f"  average rank displacement per individual         : "
                f"{atS['seRankShift']:.1f} positions")
            add(f"  regret of the picked champion (true reward lost) : "
                f"{atS['bestRegret']:.1f}")
            add(f"  winner's curse (reported minus held-out reward)  : "
                f"{atS['winnersCurse']:.1f}  <- the champion is overrated by this much")
            add("")

        add("[2] HOW MANY SEEDS WOULD BE NEEDED")
        add("      seeds   Pearson   Spearman   model   elite overlap   champion regret")
        for r in cp["reliabilityCurve"]:
            add(f"      {r['batchSize']:5d}   {r['pearson']:7.3f}   {r['spearman']:8.3f}   "
                f"{r['predictedReliability']:5.3f}   "
                f"{r['eliteOverlapAB'] * 100:12.0f}%   {r['bestRegret']:15.1f}")
        add("")
        add("      target      seeds needed (score level)   seeds needed (rank level)")
        for target in sorted(cp["requiredSeeds"]):
            a = cp["requiredSeeds"][target]
            b = cp.get("requiredSeedsRank", {}).get(target, float("nan"))
            fa = f"{a:,.0f}" if math.isfinite(a) else "unreachable"
            fb = f"{b:,.0f}" if math.isfinite(b) else "unreachable"
            add(f"      {target}        {fa:>20}   {fb:>24}")
        add("      (score level = the variance model, matches Pearson. rank level =")
        add("       extrapolated from the measured Spearman, i.e. what selection sees.)")
        add("")

        if scS is not None:
            add("[bonus] HOW MUCH THE REPORTED CURVE WOBBLES WITHOUT ANY CHANGE")
            add(f"  re-scoring the SAME population on a fresh {ref}-seed batch moves")
            add(f"    the population mean by sd {scS['populationMeanSd']:.1f} "
                f"(predicted {scS['predictedPopulationMeanSd']:.1f})")
            add(f"    the best fitness by       sd {scS['bestFitnessSd']:.1f}")
            if trainingCurve and len(trainingCurve.get("bestFitness", [])) > 2:
                deltas = np.abs(np.diff(np.array(trainingCurve["bestFitness"])))
                add(f"  observed generation-to-generation change of the best fitness: "
                    f"mean |delta| {deltas.mean():.1f}")
                add("  -> compare the two: whatever the wobble explains is not progress.")
            add("")

    add("=" * 78)
    return "\n".join(L)


# ── Plots ─────────────────────────────────────────────────────────────────────

def makePlots(checkpoints, R_last, outDir, trainingBatchSize, trainingCurve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = trainingBatchSize

    # 1) reliability over batch size
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cp in checkpoints:
        sizes = [r["batchSize"] for r in cp["reliabilityCurve"]]
        line, = ax.plot(sizes, [r["spearman"] for r in cp["reliabilityCurve"]], "o-",
                        label=f"gen {cp['generation']} rank (Spearman)")
        ax.plot(sizes, [r["pearson"] for r in cp["reliabilityCurve"]], "s-.",
                color=line.get_color(), alpha=0.7,
                label=f"gen {cp['generation']} score (Pearson)")
        ax.plot(sizes, [r["predictedReliability"] for r in cp["reliabilityCurve"]], "--",
                color=line.get_color(), alpha=0.4, label=f"gen {cp['generation']} model")
    ax.axvline(S, color="red", ls=":", label=f"currently used ({S} seeds)")
    ax.axhline(0.9, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("seeds per generation")
    ax.set_ylabel("test-retest rank correlation")
    ax.set_title("How reproducible is the ranking?")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "fig1_reliability.png"), dpi=150)
    plt.close(fig)

    # 2) variance decomposition
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"gen {cp['generation']}" for cp in checkpoints]
    bottom = np.zeros(len(checkpoints))
    parts = [("shareIndividual", "real skill (signal)"),
             ("shareSeed", "seed difficulty (shifts the curve)"),
             ("shareInteraction", "individual x seed (corrupts selection)")]
    for key, label in parts:
        vals = np.array([cp["components"][key] * 100 for cp in checkpoints])
        ax.bar(labels, vals, bottom=bottom, label=label)
        for x, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(x, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8)
        bottom += vals
    ax.set_ylabel("share of total variance [%]")
    ax.set_title("Where the fitness scatter comes from")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "fig2_variance_components.png"), dpi=150)
    plt.close(fig)

    # 3) selection quality
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cp in checkpoints:
        sizes = [r["batchSize"] for r in cp["reliabilityCurve"]]
        axA.plot(sizes, [r["eliteOverlapAB"] * 100 for r in cp["reliabilityCurve"]],
                 "o-", label=f"gen {cp['generation']}")
        axB.plot(sizes, [r["bestRegret"] for r in cp["reliabilityCurve"]],
                 "o-", label=f"gen {cp['generation']}")
    for a in (axA, axB):
        a.axvline(S, color="red", ls=":")
        a.set_xlabel("seeds per generation")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    axA.set_ylabel(f"identical elite members [%] (E={ELITE_SIZE})")
    axA.set_title("Would selection pick the same individuals again?")
    axB.set_ylabel("true reward lost [reward]")
    axB.set_title("Regret of the picked champion")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "fig3_selection_quality.png"), dpi=150)
    plt.close(fig)

    # 4) curve wobble
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cp in checkpoints:
        sizes = [r["batchSize"] for r in cp["batchMeanScatter"]]
        ax.plot(sizes, [r["populationMeanSd"] for r in cp["batchMeanScatter"]],
                "o-", label=f"gen {cp['generation']} population mean")
        ax.plot(sizes, [r["bestFitnessSd"] for r in cp["batchMeanScatter"]],
                "s--", alpha=0.6, label=f"gen {cp['generation']} best fitness")
    if trainingCurve and len(trainingCurve.get("bestFitness", [])) > 2:
        deltas = np.abs(np.diff(np.array(trainingCurve["bestFitness"])))
        ax.axhline(deltas.mean(), color="black", ls="-.", lw=1,
                   label="observed |change| per generation")
    ax.axvline(S, color="red", ls=":")
    ax.set_xlabel("seeds per generation")
    ax.set_ylabel("scatter of the reported value [reward]")
    ax.set_title("How much the fitness curve moves without anything changing")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "fig4_curve_noise.png"), dpi=150)
    plt.close(fig)

    # 5) reward shape at the last checkpoint
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))
    axA.hist(R_last.ravel(), bins=60, color="steelblue")
    axA.axvline(na.LANDING_SUCCESS_THRESHOLD, color="red", ls=":", label="landing threshold")
    axA.set_xlabel("reward of a single episode")
    axA.set_ylabel("count")
    axA.set_title("Reward distribution per episode")
    axA.legend(fontsize=8)
    seedMeans = R_last.mean(axis=0)
    axB.hist(seedMeans, bins=25, color="darkorange")
    axB.set_xlabel("population mean reward on one seed")
    axB.set_ylabel("count")
    axB.set_title("Seed difficulty spread")
    fig.tight_layout()
    fig.savefig(os.path.join(outDir, "fig5_reward_shape.png"), dpi=150)
    plt.close(fig)


# ── Runner ────────────────────────────────────────────────────────────────────

def runStudy(args):
    import fracnetics as fn
    import gymnasium as gym

    outDir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(outDir, exist_ok=True)

    random.seed(args.studySeed)
    rng = np.random.default_rng(args.studySeed)

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=0.5)
    pop = buildPopulation(fn)

    checkpointGens = sorted(set(args.checkpoints))
    lastGen = max(checkpointGens)
    trainingCurve = {"generation": [], "bestFitness": [], "meanFitness": []}
    checkpoints, matrices = [], {}
    episodes = 0
    t0 = time.time()

    for g in range(lastGen + 1):
        if g in checkpointGens:
            measureSeeds = random.sample(range(0, SEED_POOL), args.measureSeeds)
            print(f"[gen {g}] measuring population on {len(measureSeeds)} seeds ...", flush=True)
            R = evaluate(pop, env, measureSeeds, args.maxSteps, args.potential)
            episodes += R.size
            matrices[g] = (R, measureSeeds)
            np.savetxt(os.path.join(outDir, f"matrix_gen{g}.csv"), R, delimiter=",",
                       header=",".join(str(s) for s in measureSeeds), comments="")
            cp = analyzeMatrix(R, args.batchSizes, args.repeats, args.trainingSeeds, rng)
            cp["generation"] = g
            cp["seeds"] = measureSeeds
            checkpoints.append(cp)
            ref = referenceBatchSize(cp, args.trainingSeeds)
            relRef = next((r["spearman"] for r in cp["reliabilityCurve"]
                           if r["batchSize"] == ref), float("nan"))
            print(f"[gen {g}] reliability@{ref} seeds = {relRef:.3f}", flush=True)

        if g == lastGen:
            break  # nothing left to evolve into

        # one ordinary generation, identical to examples/lunarlander.py
        seeds = random.sample(range(0, SEED_POOL), args.trainingSeeds)
        R_train = evaluate(pop, env, seeds, args.maxSteps, args.potential)
        episodes += R_train.size
        pop.lexicaseSelection(E=ELITE_SIZE, type=args.lexicaseType)
        best = pop.individuals[pop.indicesElite[0]]
        trainingCurve["generation"].append(g)
        trainingCurve["bestFitness"].append(float(best.fitness))
        trainingCurve["meanFitness"].append(float(pop.meanFitness))
        varyPopulation(pop, g)
        if g % 10 == 0:
            print(f"[gen {g}] best {best.fitness:8.1f}  mean {pop.meanFitness:8.1f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    runInfo = {
        "nIndividuals": N_INDIVIDUALS,
        "trainingBatchSize": args.trainingSeeds,
        "measureSeeds": args.measureSeeds,
        "generations": lastGen,
        "checkpoints": checkpointGens,
        "maxSteps": args.maxSteps,
        "studySeed": args.studySeed,
        "potential": bool(args.potential),
        "lexicaseType": args.lexicaseType,
        "episodes": episodes,
        "wallTimeSeconds": time.time() - t0,
    }

    writeOutputs(outDir, runInfo, checkpoints, trainingCurve,
                 matrices[max(matrices)][0])
    return outDir


def writeOutputs(outDir, runInfo, checkpoints, trainingCurve, R_last):
    report = formatReport(runInfo, checkpoints, trainingCurve)
    print("\n" + report)
    with open(os.path.join(outDir, "report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(outDir, "summary.json"), "w") as f:
        json.dump({"run": runInfo, "checkpoints": checkpoints,
                   "trainingCurve": trainingCurve}, f, indent=2)
    if trainingCurve["generation"]:
        np.savetxt(os.path.join(outDir, "training_curve.csv"),
                   np.column_stack([trainingCurve["generation"],
                                    trainingCurve["bestFitness"],
                                    trainingCurve["meanFitness"]]),
                   delimiter=",", header="generation,bestFitness,meanFitness", comments="")
    makePlots(checkpoints, R_last, outDir, runInfo["trainingBatchSize"], trainingCurve)
    print(f"\nresults written to {outDir}")


def analyzeOnly(args):
    """Recompute everything from stored matrices -- no episodes are run."""
    outDir = args.analyzeOnly
    rng = np.random.default_rng(args.studySeed)
    files = sorted(
        (f for f in os.listdir(outDir) if f.startswith("matrix_gen") and f.endswith(".csv")),
        key=lambda f: int(f[len("matrix_gen"):-len(".csv")]))
    if not files:
        raise SystemExit(f"no matrix_gen*.csv found in {outDir}")

    checkpoints, R_last = [], None
    for f in files:
        g = int(f[len("matrix_gen"):-len(".csv")])
        R = np.loadtxt(os.path.join(outDir, f), delimiter=",", skiprows=1)
        R_last = R
        cp = analyzeMatrix(R, args.batchSizes, args.repeats, args.trainingSeeds, rng)
        cp["generation"] = g
        checkpoints.append(cp)

    trainingCurve = {"generation": [], "bestFitness": [], "meanFitness": []}
    curvePath = os.path.join(outDir, "training_curve.csv")
    if os.path.exists(curvePath):
        c = np.loadtxt(curvePath, delimiter=",", skiprows=1)
        if c.ndim == 2:
            trainingCurve = {"generation": c[:, 0].tolist(),
                             "bestFitness": c[:, 1].tolist(),
                             "meanFitness": c[:, 2].tolist()}

    runInfo = {"nIndividuals": int(R_last.shape[0]),
               "trainingBatchSize": args.trainingSeeds,
               "measureSeeds": int(R_last.shape[1]),
               "generations": max(cp["generation"] for cp in checkpoints),
               "checkpoints": [cp["generation"] for cp in checkpoints],
               "maxSteps": args.maxSteps, "studySeed": args.studySeed,
               "potential": bool(args.potential),
               "lexicaseType": args.lexicaseType,
               "episodes": 0, "wallTimeSeconds": 0.0}
    writeOutputs(outDir, runInfo, checkpoints, trainingCurve, R_last)
    return outDir


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generations", type=int, default=60,
                   help="ignored if --checkpoints is given explicitly")
    p.add_argument("--checkpoints", type=int, nargs="+", default=None,
                   help="generations at which the full measurement is taken")
    p.add_argument("--measure-seeds", dest="measureSeeds", type=int, default=60,
                   help="seeds per checkpoint measurement (cost: nIndividuals x this)")
    p.add_argument("--training-seeds", dest="trainingSeeds", type=int, default=10,
                   help="seeds per generation during evolution (maxSeeds in lunarlander.py)")
    p.add_argument("--batch-sizes", dest="batchSizes", type=int, nargs="+",
                   default=[1, 2, 3, 5, 10, 15, 20, 30],
                   help="batch sizes for the reliability curve (need 3*S <= measure-seeds)")
    p.add_argument("--repeats", type=int, default=400,
                   help="resampling repetitions per batch size")
    p.add_argument("--max-steps", dest="maxSteps", type=int, default=1000)
    p.add_argument("--lexicase-type", dest="lexicaseType", default=LEXICASE_TYPE,
                   choices=["seeds", "objectives", "objectivesPerSeed"],
                   help="test-case decomposition used by lexicaseSelection during evolution")
    p.add_argument("--potential", action="store_true",
                   help="evaluate with potential-based reward shaping (Network::fitGymnasium)")
    p.add_argument("--study-seed", dest="studySeed", type=int, default=42)
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--analyze-only", dest="analyzeOnly", default=None,
                   help="re-run the analysis on stored matrices in this directory")
    p.add_argument("--quick", action="store_true",
                   help="small smoke-test configuration")
    args = p.parse_args(argv)

    if args.quick:
        args.measureSeeds = 36
        args.batchSizes = [1, 2, 4, 8, 12]
        args.repeats = 100
        args.maxSteps = 300
        if args.checkpoints is None:
            args.checkpoints = [0, 3]
    if args.checkpoints is None:
        args.checkpoints = sorted({0, args.generations // 3, args.generations})
    # 3 * S <= M: batch A, batch B and an untouched held-out set. Two disjoint
    # batches from a finite pool are negatively dependent, and at S = M/2 they are
    # exact complements -- see noise_analysis.splitStatistics().
    args.batchSizes = [s for s in args.batchSizes if 3 * s <= args.measureSeeds]
    if args.trainingSeeds not in args.batchSizes and 3 * args.trainingSeeds <= args.measureSeeds:
        args.batchSizes = sorted(args.batchSizes + [args.trainingSeeds])
    return args


if __name__ == "__main__":
    a = parseArgs()
    if a.analyzeOnly:
        analyzeOnly(a)
    else:
        runStudy(a)
