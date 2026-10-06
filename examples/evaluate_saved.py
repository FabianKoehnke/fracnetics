#!/usr/bin/env python
"""Re-evaluate a network saved by lunarlander.py, without retraining.

A full run takes many hours, so its result must survive the process. lunarlander.py
pickles the final and the best-ever individual into runs/<timestamp>/; this script loads
one of them and measures it again -- on more seeds, on a different curriculum level, or
simply to reproduce a number after the terminal output is gone.

Usage:
    ./venv/bin/python examples/evaluate_saved.py runs/run_20260907_121004
    ./venv/bin/python examples/evaluate_saved.py runs/run_.../bestEver.pkl --seeds 2000
    ./venv/bin/python examples/evaluate_saved.py runs/run_... --render 3
"""

import argparse
import json
import os
import pickle
import random
import statistics

import gymnasium as gym

import fracnetics as fn

MIN_FEATURES = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
MAX_FEATURES = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]
LANDING_THRESHOLD = 100.0


def loadIndividual(path):
    """Accepts either a .pkl file or a run directory (then prefers bestEver.pkl)."""
    if os.path.isdir(path):
        for name in ("bestEver.pkl", "final.pkl"):
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise SystemExit(f"no final.pkl or bestEver.pkl in {path}")
    with open(path, "rb") as f:
        return pickle.load(f), path


def buildHost():
    """A one-individual population to run the loaded network in.

    The constructor arguments only size the initial random network, which is replaced by
    the loaded one -- but nFeatureValues must match what the network was trained with.
    """
    pop = fn.Population(seed=5, ni=1, jn=8, jnf=8, pn=4, pnf=4,
                        fractalJudgment=False, useExperience=False,
                        nFeatureValues=[0, 0, 0, 0, 0, 0, 2, 2])
    pop.setAllNodeBoundaries(MIN_FEATURES, MAX_FEATURES)
    return pop


def evaluate(individual, seeds, maxSteps, curriculumLevel, absoluteImpulseCurriculum):
    """Returns one record per seed: reward plus the four episode objectives.

    The objectives are what turns "it failed" into "it failed because": survived steps
    (equal to maxSteps means the time ran out), final speed, final horizontal distance
    from the pad, and whether Gymnasium counted it as a landing.
    """
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False,
                   wind_power=15.0, turbulence_power=0.5)
    pop = buildHost()
    records = []
    for s in seeds:
        pop.individuals = [individual]
        pop.gymnasium(env, dMax=10, maxSteps=maxSteps, maxConsecutiveP=5, worstFitness=0,
                      seed=s, validation=True, curriculumLevel=curriculumLevel,
                      absoluteImpulseCurriculum=absoluteImpulseCurriculum)
        ind = pop.individuals[0]
        # Layout of lexicaseObjectives (Network::lastEpisodeObjectives):
        #   [0] total Gym reward
        #   [1] -|vy_end|, [2] -|vx_end|, [3] -|x_end| -- ONLY on a landing,
        #       otherwise a fixed penalty (-4/-4/-2)
        #   [4] landed
        # The penalties make 1-3 useless as a measure for failures, so the diagnosis reads
        # Network::lastFitnessII (|x| in the last frame, always set) and lastFitness
        # (speed at first ground contact) instead.
        objectives = list(ind.lexicaseObjectives) or [0.0] * 5
        landed = len(objectives) > 4 and objectives[4] > 0.5
        records.append({
            "seed": s,
            "reward": ind.fitness,
            "landed": landed,
            "endDistance": ind.lastFitnessII,     # |x| in the last frame, always available
            "touchdownSpeed": ind.lastFitness,    # speed at ground contact
            # Only meaningful on a landing, otherwise the penalty:
            "endSpeedVertical": (-objectives[1]) if landed else None,
            "endSpeedLateral": (-objectives[2]) if landed else None,
        })
    env.close()
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="run directory or .pkl file")
    p.add_argument("--seeds", type=int, default=1000, help="number of validation seeds")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for drawing them")
    p.add_argument("--max-steps", dest="maxSteps", type=int, default=1500)
    p.add_argument("--curriculum-level", dest="curriculumLevel", type=float, default=1.0)
    p.add_argument("--absolute-impulse", dest="absoluteImpulse", action="store_true")
    p.add_argument("--render", type=int, default=0, metavar="N",
                   help="record the first N episodes as video into videos/")
    p.add_argument("--render-worst", dest="renderWorst", type=int, default=0, metavar="N",
                   help="record the N WORST episodes -- the interesting ones")
    p.add_argument("--video-dir", dest="videoDir", default="videos")
    p.add_argument("--save-stats", dest="saveStats", default=None, metavar="PATH",
                   help="write the breakdown as CSV (default: extra_auswertung.csv "
                        "next to the loaded model)")
    args = p.parse_args()

    individual, path = loadIndividual(args.path)
    print(f"loaded {path}")
    print(f"  {len(individual.innerNodes)} nodes, "
          f"{sum(1 for n in individual.innerNodes if n.used)} of them marked used")

    summaryPath = os.path.join(os.path.dirname(path), "summary.json")
    if os.path.exists(summaryPath):
        with open(summaryPath) as f:
            summary = json.load(f)
        print(f"  run reported: final {summary['finalValidationMean']:.2f}"
              + (f", best ever {summary['bestEver']['validationMean']:.2f}"
                 if summary.get("bestEver") else ""))

    random.seed(args.seed)
    seeds = [random.randint(0, 10_000) for _ in range(args.seeds)]
    records = evaluate(individual, seeds, args.maxSteps, args.curriculumLevel,
                       args.absoluteImpulse)
    rewards = [r["reward"] for r in records]

    landed = [r for r in rewards if r > LANDING_THRESHOLD]
    print(f"\n{len(rewards)} seeds | mean {statistics.mean(rewards):.2f} "
          f"| median {statistics.median(rewards):.2f}")
    print(f"  landing rate {100 * len(landed) / len(rewards):.1f}% "
          f"({len(landed)} of {len(rewards)})")
    if landed:
        print(f"  mean reward when landed: {statistics.mean(landed):.2f}")
    crashed = [r for r in rewards if r <= LANDING_THRESHOLD]
    if crashed:
        print(f"  mean reward when not landed: {statistics.mean(crashed):.2f}")

    # ── Why do the failures fail? ────────────────────────────────────────────
    failures = [r for r in records if r["reward"] <= LANDING_THRESHOLD]
    offPad, hard = [], []
    if failures:
        # Without the step count (objectives[0] now holds the Gym reward) a timeout can no
        # longer be detected directly, so the split uses the end position.
        offPad = [r for r in failures if r["endDistance"] > 0.2]
        hard = [r for r in failures if r["endDistance"] <= 0.2]
        print(f"\n{len(failures)} failures, broken down:")
        print(f"  came down away from the pad (|x| > 0.2):  {len(offPad):>4} "
              f"({100*len(offPad)/len(failures):.0f}%)")
        print(f"  came down near the pad but not counted:   {len(hard):>4} "
              f"({100*len(hard)/len(failures):.0f}%)")
        for label, group in (("off pad", offPad), ("near pad", hard)):
            if group:
                print(f"    {label:>9}: mean reward {statistics.mean(x['reward'] for x in group):7.1f} "
                      f"| mean |x| {statistics.mean(x['endDistance'] for x in group):.2f} "
                      f"| mean touchdown speed {statistics.mean(x['touchdownSpeed'] for x in group):.2f}")
        successes = [r for r in records if r["reward"] > LANDING_THRESHOLD]
        if successes:
            print(f"  for comparison, the successes: mean |x| "
                  f"{statistics.mean(x['endDistance'] for x in successes):.2f} "
                  f"| mean touchdown speed "
                  f"{statistics.mean(x['touchdownSpeed'] for x in successes):.2f}")
            lateral = [x["endSpeedLateral"] for x in successes if x["endSpeedLateral"] is not None]
            if lateral:
                print(f"    of those, mean lateral speed at the end: {statistics.mean(lateral):.3f}")

    # ── Breakdown as CSV ─────────────────────────────────────────────────────
    statsPath = args.saveStats
    if statsPath is None and os.path.isdir(os.path.dirname(path) or "."):
        statsPath = os.path.join(os.path.dirname(path), "extra_auswertung.csv")
    if statsPath:
        successes = [r for r in records if r["reward"] > LANDING_THRESHOLD]
        groups = [
            ("alle", records),
            ("gelandet", successes),
            ("nicht gelandet", failures),
            ("  davon neben dem Pad", offPad if failures else []),
            ("  davon nahe am Pad (nicht gewertet)", hard if failures else []),
        ]
        with open(statsPath, "w") as f:
            f.write("gruppe,n,anteil_prozent,mittlerer_reward,median_reward,"
                    "mittleres_x,mittlere_aufsetzgeschwindigkeit\n")
            for label, group in groups:
                if not group:
                    f.write(f"{label},0,0,,,,\n")
                    continue
                f.write(
                    f"{label},{len(group)},{100*len(group)/len(records):.1f},"
                    f"{statistics.mean(r['reward'] for r in group):.2f},"
                    f"{statistics.median(r['reward'] for r in group):.2f},"
                    f"{statistics.mean(r['endDistance'] for r in group):.3f},"
                    f"{statistics.mean(r['touchdownSpeed'] for r in group):.3f}\n")
        print(f"\nAufschluesselung geschrieben: {statsPath}")

        # Raw per-seed data alongside, so the groups can be traced back.
        perSeedPath = statsPath.replace(".csv", "_je_seed.csv")
        with open(perSeedPath, "w") as f:
            f.write("seed,reward,gelandet,x_ende,geschwindigkeit_aufsetzen,"
                    "vy_ende,vx_ende\n")
            for r in sorted(records, key=lambda r: r["reward"]):
                vy = f"{r['endSpeedVertical']:.3f}" if r['endSpeedVertical'] is not None else ""
                vx = f"{r['endSpeedLateral']:.3f}" if r['endSpeedLateral'] is not None else ""
                f.write(f"{r['seed']},{r['reward']:.2f},{int(r['landed'])},"
                        f"{r['endDistance']:.3f},{r['touchdownSpeed']:.3f},{vy},{vx}\n")
        print(f"Rohdaten je Seed geschrieben: {perSeedPath}")

    toRender = []
    if args.render > 0:
        toRender += [(r, "first") for r in records[:args.render]]
    if args.renderWorst > 0:
        worst = sorted(records, key=lambda r: r["reward"])[:args.renderWorst]
        toRender += [(r, "worst") for r in worst]

    if toRender:
        from gymnasium.wrappers import RecordVideo
        os.makedirs(args.videoDir, exist_ok=True)
        pop = buildHost()
        print(f"\nrecording {len(toRender)} episodes into {args.videoDir}/")
        for record, tag in toRender:
            s = record["seed"]
            env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                           enable_wind=False, wind_power=15.0, turbulence_power=0.5,
                           render_mode="rgb_array")
            # The reward goes into the file name, so a directory listing already tells
            # which episode is which.
            env = RecordVideo(env, video_folder=args.videoDir,
                              name_prefix=f"{tag}_seed{s}_reward{record['reward']:.0f}")
            pop.individuals = [individual]
            pop.gymnasium(env, dMax=10, maxSteps=args.maxSteps, maxConsecutiveP=5,
                          worstFitness=0, seed=s, validation=True,
                          curriculumLevel=args.curriculumLevel,
                          absoluteImpulseCurriculum=args.absoluteImpulse)
            env.close()
            print(f"  {tag:>5} seed {s:>5}: reward {record['reward']:7.1f} | "
                  f"|x| {record['endDistance']:.2f} | "
                  f"touchdown speed {record['touchdownSpeed']:.2f} | "
                  f"{'landed' if record['landed'] else 'not landed'}")


if __name__ == "__main__":
    main()
