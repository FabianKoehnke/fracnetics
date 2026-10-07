#!/usr/bin/env python

import fracnetics as fn
import gymnasium as gym
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors  # feste Farbskala der Seed-Heatmap (TwoSlopeNorm)
import statistics
from gymnasium.wrappers import RecordVideo
from pyvis.network import Network # for network visualization 
import math
import time 
import numpy as np
import math 
import pandas as pd
import os
import json
import pickle
import sys
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_eval import ParallelEvaluator

# ## Initializing the Population

seed=10
# Initialize the population
# Number of outgoing edges per judgment node, one entry per feature. A 0 means "draw a
# random number between 2 and the network size" (Node::setEdges), so the resolution of a
# node is left to chance -- some cut their feature once, others thirty times.
#
# Measured over two runs that differed only in this setting (1000 generations each):
#   [5,5,5,5,5,5,2,2] -> 140.8 on 1000 unseen seeds, 64.1% landing rate, 17 nodes median
#   [0,0,0,0,0,0,2,2] -> 126.8,                      55.6%,                35 nodes median
# The fixed count wins with half the network. With 5 edges the cut points span the whole
# range: on vy over [-2.27, 0.40] they land at -1.74, -1.20, -0.67, -0.13 -- one separating
# fast from moderate descent, the others inside the landing regime. A 2-edge node has a
# single cut and cannot do both; the random variant produced nodes with up to 34 edges,
# which resolve 34 states where five suffice and cost 34 edges to evolve.
# Leg contacts stay at 2: they are binary, one cut is exactly right.
#
# Set to 3 after measuring the 235.7 run (runs/run_20260912_134234): of the 70 inner
# boundaries in its 19 active nodes, 58 (83%) sit outside the range the feature ever
# takes -- vy stays in [-0.87, 0.39] while the node cuts at -9.28 or +7.08. Both
# boundary mutations are confined to (b[i-1], b[i+1]), so a dead boundary can only
# return by walking its neighbours inward first; 21 of the 58 cannot reach the live
# range at all. That run used 0, i.e. "up to network size" edges, which is how nodes
# with 15 and 18 edges appeared.
# CAUTION: the two runs above are the only direct comparison and they favour 5, not 3.
# 3 is chosen to shrink the search space further; if it underperforms, 5 is the
# measured fallback, not 0.
nFeatureValues = [0, 0, 0, 0, 0, 0, 2, 2]

# Share of dormant nodes Network::addDelNodes tolerates before it stops adding: growth is
# allowed only while (size - used) <= size * junkShare. At 0 the network freezes at its
# initial size, which measurably capped a run at ~100.
junkShare = 0.1

# ── How unused nodes are protected from deletion ─────────────────────────────
# junk only caps HOW MANY unused nodes a network may keep; it never protects a
# particular node. Without a second criterion addDelNodes() simply deletes from the
# lowest index upwards, which in 20.5% of cases hit a node that ran one generation
# earlier and was merely missing from this generation's seed panel.
#
# A node is deleted once unusedSince >= nodeGracePeriod, so n grants n-1 idle
# generations; n = 1 grants none and disables the protection entirely.
#
# This replaced an indirect route via the coactivation matrix (recordTransition ->
# findTransitionClusters -> clusterLabels), which has since been removed. Both were
# compared on the same state (30 generations, 1357 deletion candidates): the cluster
# route protected 34.0%, a grace period of 3 protects 45.5%, and not one node was
# protected by the cluster route alone -- it was a strict subset. The A/B run
# (seed 10, 2000 generations) gave 203.6 against 201.1, i.e. no measurable difference
# at a seed-to-seed spread of 34.6 points.
# Nodes that were NEVER traversed stay immediately deletable -- 50.8% of the deletable
# nodes, and genuine junk.
nodeGracePeriod = 3

# Kept as a dict so the worker processes can rebuild an identical container population
# (see parallel_eval.ParallelEvaluator).
populationKwargs = dict(
    seed=seed,
    jn=8,         # judgment nodes
    jnf=8,        # judgment node functions
    pn=4,         # processing nodes
    pnf=4,        # processing node functions (0-3: actions, 4: "last decision" -> enables judgment-node hysteresis)
    fractalJudgment=False,
    useExperience=False,
    nFeatureValues=nFeatureValues
)

pop = fn.Population(ni=100, **populationKwargs)   # ni = number of individuals

# ── Input feature boundaries ─────────────────────────────────────────────────
# A judgment node splits [minFeatures[f], maxFeatures[f]] into EQUAL intervals
# (Node::setEdgesBoundaries), and Node::judge() CLIPS values outside the outermost
# boundaries onto the first or last edge.
#
# These are the values of the run that reached 195 on 1000 unseen seeds -- carried over
# from CartPole and measurably too wide (a 3-edge angularVel node cuts at -3.33/+3.33
# while the values stay within [-0.94, 0.92], so every observation lands in the same
# interval). Narrower bounds were tried twice and are NOT established as better:
#   [-1.16 ... 1.17] etc.  -> 140.8 and 126.8, but over 1000 generations instead of 2000
#   [-1.30 ... 1.30] etc.  -> clipped 13-34% of vy on networks that do not yet fly well,
#                             which brought back the fast, aggressive approach
# Until 195 is reproduced on this baseline, the bounds stay as they were.
minFeatures = [-2.5, -2.5, -10, -10, -6.2831855, -10, 0, 0]
maxFeatures = [2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1]

pop.setAllNodeBoundaries(minFeatures, maxFeatures)

# ## Helperfunctions

def renderVideos(pop, generation = None, seeds = [], N=5, curriculumLevel=1.0, validation=True,
                  uniformDirectionCurriculum=False, directionAngles=None,
                  maxSteps=None, absoluteImpulseCurriculum=None, survivalMode=None,
                  potential=False):
    """Render episodes of the current elite.

    The reward-mode arguments default to the values the training loop is using, because
    a video is only worth watching if it shows the SAME task the individual was scored
    on. Passing absoluteImpulseCurriculum=True while training runs with False changes
    the initial impulse and turns the episode into a different problem -- measured on a
    trained individual, that one flag moved the reward from +10.7 to -260.7.
    """
    if maxSteps is None:
        maxSteps = globals().get("maxSteps", 1000)
    if absoluteImpulseCurriculum is None:
        absoluteImpulseCurriculum = globals().get("absoluteImpulseCurriculum", False)
    if survivalMode is None:
        survivalMode = globals().get("survivalMode", False)
    
    popRender = fn.Population(
        seed=seed,
        ni=1,       # number of individuals
        jn=8,         # judgment nodes
        jnf=8,        # judgment node functions
        pn=4,         # processing nodes
        pnf=5,        # processing node functions (0-3: actions, 4: "last decision" -> enables judgment-node hysteresis)
        fractalJudgment=False,
        #nFeatureValues=[0,0,0,0,0,0,2,2,0,0,0,0,0,0,2,2]
        nFeatureValues=[5,5,5,5,5,5,2,2]
    )
    
    popRender.individuals = [pop.individuals[pop.indicesElite[0]]]
    if len(seeds) == 0:
        seeds = [random.randint(0, 10_000) for _ in range(N)]

    # Falls keine passenden Winkel mitgegeben wurden (z.B. Aufruf mit anderer
    # Seed-Anzahl als das letzte Trainings-Batch), neue gleichverteilte Winkel
    # ziehen -- damit das Video IMMER die aktuell aktiven Curriculum-
    # Einstellungen widerspiegelt, auch wenn validation=True gerendert wird.
    if uniformDirectionCurriculum and (directionAngles is None or len(directionAngles) != len(seeds)):
        directionAngles = drawDirectionAngles(
            len(seeds),
            excludedBands=[
                # (math.radians(60), math.radians(120)),
                # (math.radians(240), math.radians(300)),
            ],
        )
    elif not uniformDirectionCurriculum:
        directionAngles = [0.0] * len(seeds)

    print(seeds)
        
    for s, angle in zip(seeds, directionAngles):
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=0.5, render_mode="rgb_array")
        
        # Named before the episode runs, so the fitness in the name would be the previous
        # seed's. Renamed after the run below instead.
        videoPrefix = f"lunarlander_generation{generation if generation is not None else g}_seed{s}"
        env = RecordVideo(env, video_folder="videos", name_prefix=videoPrefix)
        
        # Record a run
        popRender.gymnasium(
            env,
            dMax=10,
            maxSteps=maxSteps,
            maxConsecutiveP=5,
            worstFitness=worstFitness,
            seed=s,
            validation=validation,
            curriculumLevel=curriculumLevel,
            absoluteImpulseCurriculum=absoluteImpulseCurriculum,
            uniformDirectionCurriculum=uniformDirectionCurriculum,
            directionAngle=angle,
            survivalMode=survivalMode,
            potential=potential
        )
        env.close()
        print(f"Fitness: {popRender.individuals[-1].fitness}")

def analyzeSeedDifficulty(individual, seeds, plot=True, outputDir="."):
    """
    Diagnose-Mechanismus: prueft, ob Abstuerze/niedrige Fitness mit der Staerke
    des initialen Zufallsimpulses korrelieren (die einzige Zufallsquelle bei
    LunarLander, die die Ausgangssituation eines Seeds schwer oder leicht macht
    -- Spawn-Position und -Winkel sind immer fix).

    Fuer jeden Seed wird:
    1. der initiale Impuls gemessen (vx, vy direkt aus der ersten Beobachtung
       nach env.reset(seed=s) -- Gymnasium fuehrt intern bereits einen No-Op-Step
       aus, wodurch der Impuls schon in eine Geschwindigkeit umgesetzt ist),
    2. eine volle Episode mit `individual` gespielt, um Fitness und Landeerfolg
       zu bestimmen. "Landed" wird ueber die Gesamt-Fitness (`ind.fitness > 100`)
       bestimmt: Gymnasium vergibt am Episodenende +100 fuer eine sichere Landung
       bzw. -100 fuer einen Crash (Rumpfkontakt). Ein Return > 100 ist nur
       erreichbar, wenn der Terminal-Bonus von +100 tatsaechlich vergeben wurde
       (Shaping-Rewards allein reichen dafuer nicht aus).
       `lastFitness != 10.0` (reiner Bein-Bodenkontakt) ist KEIN verlaesslicher
       Landeerfolg-Indikator, da ein Lander nach kurzem Beinkontakt trotzdem
       umkippen/crashen kann (fruehere Fehlklassifikation: tief-negative
       "gelandete" Punkte im Scatterplot).

    Gibt ein DataFrame mit (seed, pushVx, pushVy, pushMagnitude, fitness, landed)
    zurueck und druckt eine Zusammenfassung / optional einen Scatterplot.
    """
    diagPop = fn.Population(
        seed=seed,
        ni=1,
        jn=8,
        jnf=8,
        pn=4,
        pnf=5,
        fractalJudgment=True,
        useExperience=False,
        nFeatureValues=[5,5,5,5,5,5,2,2]
    )
    diagPop.individuals = [individual]

    rawEnv = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=0.5)
    runEnv = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                       enable_wind=False, wind_power=15.0, turbulence_power=0.5)

    rows = []
    for s in seeds:
        obs, _ = rawEnv.reset(seed=s)
        pushVx, pushVy = float(obs[2]), float(obs[3])
        pushMagnitude = math.sqrt(pushVx**2 + pushVy**2)

        # Same reward mode as the training loop -- otherwise the "difficulty" measured
        # here belongs to a different task (see renderVideos()).
        diagPop.gymnasium(
            runEnv,
            dMax=10,
            maxSteps=globals().get("maxSteps", 1000),
            maxConsecutiveP=5,
            worstFitness=0,
            seed=s,
            validation=True,
            absoluteImpulseCurriculum=globals().get("absoluteImpulseCurriculum", False),
            survivalMode=globals().get("survivalMode", False)
        )
        ind = diagPop.individuals[0]
        landed = ind.fitness > 100

        rows.append({
            "seed": s,
            "pushVx": pushVx,
            "pushVy": pushVy,
            "pushMagnitude": pushMagnitude,
            "fitness": ind.fitness,
            "landed": landed
        })

    rawEnv.close()
    runEnv.close()

    df = pd.DataFrame(rows)
    # Direction of the initial impulse, 0-360 degrees. Together with the magnitude these
    # are the two features a stratified seed draw could balance -- so the question this
    # analysis has to answer is how much of the landing outcome they actually explain.
    df["pushAngle"] = np.degrees(np.arctan2(df["pushVy"], df["pushVx"])) % 360.0
    landedPush = df.loc[df["landed"], "pushMagnitude"]
    crashedPush = df.loc[~df["landed"], "pushMagnitude"]
    corr = df["pushMagnitude"].corr(df["fitness"])

    # Landing rate per 45-degree sector: if direction drives the difficulty, the rates
    # must differ markedly between sectors.
    sectorSize = 45
    df["pushSector"] = (df["pushAngle"] // sectorSize).astype(int) * sectorSize
    sectorStats = df.groupby("pushSector").agg(
        n=("landed", "size"), landingRate=("landed", "mean"), meanFitness=("fitness", "mean"))

    # How well do the two features separate landed from crashed at all? A single split on
    # the magnitude, at the best possible threshold -- the accuracy of that is an upper
    # bound for what stratifying by magnitude alone could buy.
    thresholds = np.linspace(df["pushMagnitude"].min(), df["pushMagnitude"].max(), 200)
    baseRate = max(df["landed"].mean(), 1.0 - df["landed"].mean())
    bestAccuracy = max(((df["pushMagnitude"] <= t) == df["landed"]).mean()
                       for t in thresholds)

    print("="*50)
    print("Seed-Schwierigkeits-Analyse (Push-Impuls)")
    print("="*50)
    print(f"Seeds gesamt: {len(df)} | Gelandet: {df['landed'].sum()} | Abgestuerzt/Timeout: {(~df['landed']).sum()}")
    print(f"Push-Magnitude (gelandet):   mean={landedPush.mean():.3f}  std={landedPush.std():.3f}  n={len(landedPush)}")
    print(f"Push-Magnitude (abgestuerzt): mean={crashedPush.mean():.3f}  std={crashedPush.std():.3f}  n={len(crashedPush)}")
    print(f"Korrelation Push-Magnitude <-> Fitness: {corr:.3f}")
    print("-"*50)
    print("Landequote je Richtungssektor (45 Grad):")
    for sector, row in sectorStats.iterrows():
        print(f"  {sector:>3}-{sector+sectorSize:>3} Grad: n={int(row['n']):>4}  "
              f"Landequote={row['landingRate']*100:5.1f}%  mittlere Fitness={row['meanFitness']:7.1f}")
    print(f"  Spannweite der Landequote ueber die Sektoren: "
          f"{(sectorStats['landingRate'].max() - sectorStats['landingRate'].min())*100:.1f} Prozentpunkte")
    print("-"*50)
    print(f"Landequote insgesamt: {df['landed'].mean()*100:.1f}%")
    print(f"Trefferquote, wenn man immer die haeufigere Klasse raet: {baseRate*100:.1f}%")
    print(f"Beste Trefferquote mit einem Schnitt auf der Push-Magnitude: {bestAccuracy*100:.1f}%")
    print(f"  -> Gewinn durch die Magnitude: {(bestAccuracy-baseRate)*100:.1f} Prozentpunkte")
    print("     (Ist der Gewinn klein, kann eine Stratifizierung nach diesem Merkmal")
    print("      wenig bewirken -- es trennt gelandet und abgestuerzt schlicht nicht.)")
    print("="*50)

    if plot:
        figDiag = plt.figure(figsize=(15, 9))
        colors = df["landed"].map({True: "green", False: "red"})

        axA = figDiag.add_subplot(2, 3, 1)
        axA.scatter(df["pushMagnitude"], df["fitness"], c=colors, alpha=0.6)
        axA.set_xlabel("Push-Magnitude |v| nach Reset")
        axA.set_ylabel("Fitness")
        axA.set_title("Push-Staerke vs. Fitness (gruen=gelandet)")

        axB = figDiag.add_subplot(2, 3, 2)
        axB.boxplot([landedPush.dropna(), crashedPush.dropna()],
                    tick_labels=["gelandet", "nicht gelandet"])
        axB.set_ylabel("Push-Magnitude |v|")
        axB.set_title("Push-Staerke je Ausgang")

        # Richtung vs. Ausgang -- die Frage, die der bisherige Plot offen liess.
        axC = figDiag.add_subplot(2, 3, 3)
        axC.scatter(df["pushAngle"], df["fitness"], c=colors, alpha=0.6)
        axC.set_xlabel("Push-Richtung [Grad]")
        axC.set_ylabel("Fitness")
        axC.set_title("Push-Richtung vs. Fitness")
        axC.set_xticks(range(0, 361, 90))

        # Landequote je Sektor: eine flache Kurve heisst, dass die Richtung die
        # Schwierigkeit NICHT erklaert und Stratifizierung danach nichts bringt.
        axD = figDiag.add_subplot(2, 3, 4)
        axD.bar(sectorStats.index + sectorSize / 2, sectorStats["landingRate"] * 100,
                width=sectorSize * 0.85, color="steelblue")
        axD.axhline(df["landed"].mean() * 100, color="black", ls="--", lw=1,
                    label=f"Gesamt {df['landed'].mean()*100:.0f}%")
        axD.set_xlabel("Push-Richtung [Grad]")
        axD.set_ylabel("Landequote [%]")
        axD.set_title("Landequote je Richtungssektor")
        axD.set_xticks(range(0, 361, 90))
        axD.set_ylim(0, 100)
        axD.legend(fontsize=8)

        # Beide Merkmale zusammen: liegen die roten Punkte in einem klar abgegrenzten
        # Bereich, koennte eine Stratifizierung ueber BEIDE Achsen greifen.
        axE = figDiag.add_subplot(2, 3, 5)
        axE.scatter(df["pushVx"], df["pushVy"], c=colors, alpha=0.6)
        axE.axhline(0, color="grey", lw=0.5)
        axE.axvline(0, color="grey", lw=0.5)
        axE.set_xlabel("Push vx")
        axE.set_ylabel("Push vy")
        axE.set_title("Impuls-Vektor (gruen=gelandet)")
        axE.set_aspect("equal", adjustable="datalim")

        axF = figDiag.add_subplot(2, 3, 6)
        axF.hist([df.loc[df["landed"], "fitness"], df.loc[~df["landed"], "fitness"]],
                 bins=40, stacked=True, color=["green", "red"],
                 label=["gelandet", "nicht gelandet"])
        axF.set_xlabel("Fitness")
        axF.set_ylabel("Anzahl Seeds")
        axF.set_title("Fitness-Verteilung ueber die Seeds")
        axF.legend(fontsize=8)

        figDiag.tight_layout()
        plotPath = os.path.join(outputDir, "seed_difficulty_analysis.png")
        figDiag.savefig(plotPath)
        plt.close(figDiag)
        print(f"Plot gespeichert: {plotPath}")

    # Rohdaten und Sektorstatistik neben den Plot legen, damit die Diagnose eines Laufs
    # spaeter nachvollziehbar ist, ohne ihn zu wiederholen.
    df.to_csv(os.path.join(outputDir, "seed_difficulty.csv"), index=False)
    sectorStats.to_csv(os.path.join(outputDir, "seed_difficulty_sektoren.csv"))
    with open(os.path.join(outputDir, "seed_difficulty_zusammenfassung.txt"), "w") as f:
        f.write(f"Seeds gesamt: {len(df)}\n")
        f.write(f"Gelandet: {int(df['landed'].sum())} ({100*df['landed'].mean():.1f}%)\n")
        f.write(f"Push-Magnitude gelandet:    mean={landedPush.mean():.3f} "
                f"std={landedPush.std():.3f} n={len(landedPush)}\n")
        f.write(f"Push-Magnitude abgestuerzt: mean={crashedPush.mean():.3f} "
                f"std={crashedPush.std():.3f} n={len(crashedPush)}\n")
        f.write(f"Korrelation Push-Magnitude <-> Fitness: {corr:.3f}\n")
        f.write(f"Spannweite der Landequote ueber die Richtungssektoren: "
                f"{(sectorStats['landingRate'].max()-sectorStats['landingRate'].min())*100:.1f} "
                f"Prozentpunkte\n")
        f.write(f"Gewinn eines Schnitts auf der Push-Magnitude gegenueber Raten: "
                f"{(bestAccuracy-baseRate)*100:.1f} Prozentpunkte\n")
    print(f"Seed-Diagnose gespeichert in {outputDir}")

    return df

def plotNetwork(individual, filename, justUsedNodes=False, justUsedEdges=False):

    net = Network(notebook=True, directed=True, cdn_resources="in_line")
    net.force_atlas_2based()

    processingFunctionNames = ["do nothing", "fire left orientation engine", "fire main engine",
                                "fire right orientation engine", "last decision"]
    judgmentFunctionNames = ["x", "y", "vel x", "vel y", "angle", "ang vel", "left contact", "right contact"]
    judgmentFunctionNames += ["delta_" + name for name in judgmentFunctionNames]

    # --- Cluster fill palette ---
    CLUSTER_FILL_USED   = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12",
                           "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
                           "#e91e63", "#00bcd4", "#8bc34a", "#ff5722"]
    CLUSTER_FILL_UNUSED = ["#f1948a", "#a9dfbf", "#aed6f1", "#fad7a0",
                           "#d2b4de", "#a2d9ce", "#f0b27a", "#99a3a4",
                           "#f48fb1", "#80deea", "#dcedc8", "#ffab91"]
    UNCLUSTERED_USED   = "#717d7e"
    UNCLUSTERED_UNUSED = "#d5d8dc"

    def cluster_fill(cluster_id, used):
        if cluster_id < 0:
            return UNCLUSTERED_USED if used else UNCLUSTERED_UNUSED
        idx = cluster_id % len(CLUSTER_FILL_USED)
        return CLUSTER_FILL_USED[idx] if used else CLUSTER_FILL_UNUSED[idx]

    # --- Gateway border encoding ---
    BORDER_ENTRY    = "#f1c40f"
    BORDER_EXIT     = "#2980b9"
    BORDER_BOTH     = "#8e44ad"
    BORDER_INTERIOR = "#2c3e50"

    def gateway_border(node_idx):
        role = gateway_role.get(node_idx)
        if role == "entry": return BORDER_ENTRY,  4, "▶ Entry"
        if role == "exit":  return BORDER_EXIT,   4, "◀ Exit"
        if role == "both":  return BORDER_BOTH,   5, "⇄ Entry+Exit"
        return BORDER_INTERIOR, 1, ""

    def node_shape(node_type):
        return "diamond" if node_type in ("J", "JE") else "dot"

    # Cluster colouring was driven by the coactivation matrix, which has been removed;
    # nodes are now coloured by type/usage only.
    cluster_labels = []
    has_clusters   = False
    gateways       = {}

    gateway_role = {}
    for cid, (entry, exit_) in gateways.items():
        if entry == exit_:
            gateway_role[entry] = "both"
        else:
            if entry in gateway_role:
                gateway_role[entry] = "both"
            else:
                gateway_role[entry] = "entry"
            if exit_ in gateway_role:
                gateway_role[exit_] = "both"
            else:
                gateway_role[exit_] = "exit"

    # --- Start node ---
    net.add_node(-1, label=f"START\n{individual.startNode.type}",
                 color="#635b3e", shape="star", size=20)

    # --- Inner nodes ---
    for node in individual.innerNodes:
        if justUsedNodes and not node.used:
            continue

        cid          = cluster_labels[node.id] if has_clusters else -1
        fill_color   = cluster_fill(cid, node.used)
        border_color, border_width, gw_tag = gateway_border(node.id)
        cluster_tag  = f"\nCluster {cid}" if cid >= 0 else "\nUnclustered"
        gw_label     = f"\n{gw_tag}" if gw_tag else ""
        shape        = node_shape(node.type)

        vis_color = {
            "background": fill_color,
            "border": border_color,
            "highlight": {"background": fill_color, "border": border_color}
        }

        if node.type == "J":
            label = (f"ID:{node.id} J\n"
                     f"F:{judgmentFunctionNames[node.f]}\n"
                     f"TC:{node.traverseCounter}"
                     f"{cluster_tag}{gw_label}")
            net.add_node(node.id, label=label, color=vis_color,
                         shape=shape, borderWidth=border_width,
                         borderWidthSelected=border_width + 2)

        elif node.type == "JE":
            total_n     = sum(e.n for e in node.edgeExperience)
            n_active    = sum(1 for e in node.edgeExperience if e.n > 0)
            best_return = max((e.meanReturn for e in node.edgeExperience if e.n > 0), default=0.0)
            label = (f"ID:{node.id} JE\n"
                     f"F:{judgmentFunctionNames[node.f]}\n"
                     f"TC:{node.traverseCounter}\n"
                     f"α:{node.alpha:.2f} γ:{node.gamma:.2f}\n"
                     f"n_total:{total_n} active:{n_active}/{len(node.edges)}\n"
                     f"best_R:{best_return:.1f}"
                     f"{cluster_tag}{gw_label}")
            net.add_node(node.id, label=label, color=vis_color,
                         shape=shape, borderWidth=border_width,
                         borderWidthSelected=border_width + 2)

        elif node.type == "P":
            label = (f"ID:{node.id} P\n"
                     f"F:{processingFunctionNames[node.f]}\n"
                     f"TC:{node.traverseCounter}"
                     f"{cluster_tag}{gw_label}")
            net.add_node(node.id, label=label, color=vis_color,
                         shape="dot", borderWidth=border_width,
                         borderWidthSelected=border_width + 2)

    # --- Edges ---
    for node in individual.innerNodes:
        if justUsedNodes and not node.used:
            continue

        cid_u = cluster_labels[node.id] if has_clusters else -1

        for idx, edge in enumerate(node.edges):
            if justUsedEdges and not individual.innerNodes[edge].used:
                continue

            cid_v = cluster_labels[edge] if has_clusters else -1

            if cid_u >= 0 and cid_v >= 0 and cid_u == cid_v:
                edge_color  = CLUSTER_FILL_USED[cid_u % len(CLUSTER_FILL_USED)]
                edge_width  = 1
                edge_dashes = False
                zone_tag    = f"[Intra C{cid_u}]"
            elif cid_u >= 0 and cid_v >= 0 and cid_u != cid_v:
                edge_color  = CLUSTER_FILL_USED[cid_v % len(CLUSTER_FILL_USED)]
                edge_width  = 2
                edge_dashes = True
                zone_tag    = f"[Inter C{cid_u}→C{cid_v}]"
            else:
                edge_color  = UNCLUSTERED_USED
                edge_width  = 3
                edge_dashes = True
                zone_tag    = "[Unclustered]"

            if node.type == "J":
                edgeLabel = f"{node.boundaries[idx]:.2f}–{node.boundaries[idx+1]:.2f}\n{zone_tag}"
                net.add_edge(node.id, edge,
                             label=edgeLabel, title=edgeLabel,
                             color=edge_color, dashes=edge_dashes, width=edge_width)

            elif node.type == "JE":
                exp  = node.edgeExperience[idx]
                b_lo = node.boundaries[idx]
                b_hi = node.boundaries[idx + 1]

                if exp.n > 0:
                    var        = (exp.m2Obs / exp.n) if exp.n > 1 else 0.0
                    r_norm     = max(0.0, min(1.0, (exp.meanReturn + 200) / 400))
                    r_int      = int((1 - r_norm) * 255)
                    g_int      = int(r_norm * 255)
                    ec         = f"rgb({r_int},{g_int},80)"
                    edge_label = f"{b_lo:.2f}–{b_hi:.2f}\nn={exp.n} R={exp.meanReturn:.1f}\n{zone_tag}"
                    title      = (f"Boundary: {b_lo:.2f}–{b_hi:.2f}\n"
                                  f"n={exp.n}\nmeanReturn={exp.meanReturn:.1f}\n"
                                  f"meanObs={exp.meanObs:.3f}\nvar={var:.4f}\n{zone_tag}")
                else:
                    ec         = "#cccccc"
                    edge_label = f"{b_lo:.2f}–{b_hi:.2f}\nn=0\n{zone_tag}"
                    title      = f"Boundary: {b_lo:.2f}–{b_hi:.2f}\nn=0\n{zone_tag}"

                net.add_edge(node.id, edge,
                             label=edge_label, title=title,
                             color=ec, dashes=edge_dashes, width=edge_width)

            else:
                net.add_edge(node.id, edge,
                             color=edge_color, dashes=edge_dashes, width=edge_width)

    net.add_edge(-1, individual.startNode.edges[0])
    net.save_graph(filename)

    # ==========================================================================
    # Coactivation heatmap
    # ==========================================================================


def nUsedNodes(ind):
    counter = 0
    for node in ind.innerNodes:
        if node.used == True:
            counter += 1
    return counter


# ## Training the Population

import random
random.seed(42)

generations = 2000
cores = 4
batches = 1
# ── Seed schedule ────────────────────────────────────────────────────────────
# The panel does not stay at one size: it starts small and grows once progress stalls.
#
# Both measured runs point at this. With 10 seeds the validation moving average reached
# ~170 by generation 800 and then stood still; with 20 seeds it was slower early but was
# still climbing at generation 1496 (~150). Not a contradiction but two phases: in
# lexicase every seed is one more filter, so more seeds means more niches and more
# retained diversity -- slower, but sustained. Fewer seeds converge quickly and then stop.
#
# The schedule takes the fast early climb of the small panel and the sustained climb of
# the large one. Deliberately a fixed schedule and not a feedback controller: a control
# loop on a noisy signal can oscillate, and one mechanism at a time is easier to attribute.
#
# Trigger: the mean of the last seedStagePlateauWindow validation values improved by less
# than seedStagePlateauDelta over the window before it. seedStageMinGenerations is the
# hysteresis that stops an early dip from advancing a stage. Overfitting is handled by the
# adaptive rotation below, not here.
# Draw the panel so that the directions of the initial impulses are spread evenly over
# eight sectors instead of being left to chance. See stratifiedSeeds() for the reasoning
# and the measurement behind it.
stratifySeedsByDirection = True

# [10,20,30] is the setting of the best run so far (195 on 1000 unseen seeds). Measured
# against it, [10,20,40] was clearly worse at the same generation (~155 vs ~190 at 1496):
# more test cases mean more lexicase filter steps and slower convergence, and that
# outweighed the extra resolution.
seedSchedule = [10, 20, 30]      # panel sizes, in order; the panel only ever grows
seedStageMinGenerations = 150    # minimum generations before a stage may advance
seedStagePlateauWindow = 50      # generations compared against the window before them
seedStagePlateauDelta = 5.0      # reward improvement below this counts as "flat"
seedStage = 0
seedStageStart = 0

maxSeeds = seedSchedule[0]       # current panel size; raised by the schedule below
fitnessProgess = []
maxFitness = 0
nBatches = 0
pool = [random.randint(1, 10_000) for _ in range(500)]
worstFitness = -500

# ── Seed universe, split into a training pool and a fixed validation set ──────
# The validation seeds are removed from the training pool so the reported curve is
# never measured on a seed the evolution has optimised against.
seedUniverse = 10_000
# Reporting only -- this set is NEVER used to select anything, so the plotted curve stays
# an honest out-of-sample number. Since the elite is now chosen by its own re-evaluation
# (see eliteCandidates below), re-scoring elite[0] is enough for the curve.
validationSeedCount = 100
validationEliteCount = 1

# ── Elite re-evaluation ──────────────────────────────────────────────────────
# An individual only becomes elite if it is genuinely the best on a FRESH seed set, not
# if it won this generation's 20-seed lottery. Measured motivation: at 10 seeds the rank
# reliability is 0.60 and the champion survived only 32% of generations, while the
# reported champion was overrated by 27-38 reward. Re-scoring on eliteSeedCount fresh
# seeds lifts the reliability of that one decision to ~0.88 (50 seeds) / ~0.94 (100) and
# shrinks the winner's curse, because the maximum is taken over eliteCandidates instead
# of the whole population and over five times as many episodes.
#
# The seeds are drawn FRESH every generation, not fixed: a fixed set would be optimised
# against over hundreds of generations and stop being out-of-sample.
#
# Cost: eliteCandidates * eliteSeedCount episodes per generation.
eliteReevaluation = True
# 10 x 50 -- the setting of the run that reached 180 on 1000 unseen seeds, kept unchanged
# so the hall of fame below is the only difference against it. The pre-selection by
# training fitness is itself noisy (the honest winner sat at rank 4.5 of 10 on average),
# hence eliteCandidates larger than eliteSize. With the hall of fame the pre-selection
# matters less, so fewer candidates on more seeds (e.g. 6 x 150) is worth trying later --
# but not in the same run as the hall of fame itself.
eliteCandidates = 10   # top-k by training fitness that get re-scored
eliteSeedCount = 50 # fresh seeds per re-evaluation

# ── Hall of fame ─────────────────────────────────────────────────────────────
# The reigning champion is entered into the re-evaluation as an ADDITIONAL candidate,
# every generation, regardless of its training fitness.
#
# The gap this closes: today a champion has to get into the top eliteCandidates by
# training fitness before it is re-scored at all. That pre-selection runs on maxSeeds
# seeds with a measured rank reliability of ~0.6, so a champion that has one unlucky
# generation drops out, loses its elite status, gets mutated and is gone -- which is why
# the validation curve keeps falling back instead of ratcheting upwards.
#
# With the champion always in the field it can only be replaced by an individual that beat
# it on the SAME freshly drawn seeds. Because those seeds are redrawn every generation,
# nothing can be overfitted to them, and validationSeeds stays untouched -- the reported
# curve remains an honest out-of-sample number.
#
# The champion also occupies an elite slot, so its genes stay in the population as
# crossover material instead of only being archived.
hallOfFame = True
validationSeeds = random.sample(range(0, seedUniverse), validationSeedCount)
trainingSeedPool = sorted(set(range(0, seedUniverse)) - set(validationSeeds))

# ── Seed rotation ────────────────────────────────────────────────────────────
# How many of the maxSeeds seeds are replaced each generation.
#   0            -- full redraw: every generation gets maxSeeds completely fresh seeds
#   1..maxSeeds  -- rolling panel: only that many are exchanged, the rest carry over
#
# MEASURED over two runs of 2000 generations that differed in nothing else:
#   full redraw (0) -> 207.2 on 1000 unseen seeds, 84.8% landing rate, 15 nodes
#   rolling (1)     -> 189.8,                      78.4%,               21 nodes
# The full redraw is also faster to get there: at generation 500 it sat at 144 where the
# panel run reached 69, and at the end its training-to-validation gap was 0.3 points
# against 21.
#
# The panel was introduced because 95% of the wobble in the reported fitness curve comes
# from the seed batch being easy or hard. That measurement was right but applied wrongly:
# seed difficulty hits every individual equally and cancels out of every comparison WITHIN
# a generation -- it distorts the curve, not the selection. The panel smoothed the display
# and allowed overfitting in exchange (measured: 15.7 reward advantage of established over
# fresh panel seeds). A redrawn batch cannot be overfitted because it is gone next
# generation.
seedPanelRotation = 0            # 0 = full redraw each generation

# The adaptive rotation below is kept but switched off: it regulates a knob that the
# measurement says should simply stay at 0.
rotationAdaptive = False
rotationMin = 1
rotationMax = 10
rotationWindow = 25
rotationGapGrowth = 5.0
rotationGapStable = 0.0
rotationCooldown = 25
rotationLastChange = 0
rotationProgress = []            # rotation per generation, for the plot

# ── Diversity parameters ─────────────────────────────────────────────────────
# Measured (research/seed_noise_study.py): from generation 25 on only ~55 of 100
# individuals are distinct at all, and the best 10 lie 2-8 reward apart while a
# 10-seed score scatters by 25-33 reward. Selection among the top is therefore a
# lottery -- not because seeds are too few, but because the candidates are copies of
# each other. These three knobs push against that without changing any mechanism:
# a smaller elite protects fewer duplicates, a higher edge-mutation rate (k/N per
# edge) spreads a converged population out again, and more recombination creates new
# combinations instead of re-copying survivors.
# ── Lexicase test-case decomposition ─────────────────────────────────────────
# "seeds"             -- one test case per seed, criterion = raw reward, MAD-based cut
# "seedsStandardized" -- same test cases, rewards z-transformed per seed, tolerance in sigma
# "seedsRank"         -- same test cases, cut by rank; collapses after one test case, see below
# "objectives"        -- the 5 episode objectives, each averaged over all seeds
# "objectivesPerSeed" -- the (seed x objective) grid, 5*maxSeeds graded test cases
#
# Rationale for the grid: with a single scalar per seed the top of a converged
# population becomes indistinguishable -- measured (research/seed_noise_study.py), the
# best 10 individuals lie 2-8 reward apart while a 10-seed score scatters by 25-33. The
# five objectives (posture, descent safety, horizontal precision, fuel, gym reward)
# measure different things and still separate individuals whose totals no longer do.
# MEASURED RESULT -- the grid does what it was built for but must NOT be the default.
# It sharpens selection (champion held 64% of generations instead of 32%, winner's curse
# halved from 34 to 17 reward), yet the evolution stalls: over 391 generations the best
# honest score stays near 0 where "seeds" reaches +128 within 156.
#
# The grid is not broken -- it optimises exactly the test cases it is given, and three of
# the five are only loosely coupled to landing. Side-by-side over 40 generations, the
# grid population ends up burning less than half the fuel (-6.6 vs -16.3) while its gym
# reward lags far behind (-167 vs -35). Objectives 0 (posture) and 3 (fuel) are sums of
# non-positive per-step terms, so both are maximised by a short, engine-less episode --
# they pull against the task. Under "seeds" they were never consulted; in the grid they
# are 20 of the 50 test cases, and lexicase gives every test case a turn as the first
# filter. Retrying the grid means restricting it to the objectives that grow with task
# success (gym reward, descent safety) or normalising 0 and 3 per step first.
# MEASURED (evaluate_saved.py on the best net of run_20260908_180555, 1000 seeds):
# not a single failure runs out of time. 207 of 1000 fail, and they fail by position, not
# by control -- 143 come down at |x| = 0.63 with a touchdown speed of 2.01 (four times
# that of the successes), 64 touch down softly ON the pad and then slide off it with 0.39
# left. The successes end at exactly 0.00.
#
# "seedsStandardized" selects on the total reward alone, in which that lateral speed is
# one small term among many. The grid makes each objective its own test case, so a network
# that slides off the pad can no longer hide it behind a good vertical descent -- and
# objective 2 (-|vx_end|) exists separately for exactly that reason.
# ── Selection schedule ───────────────────────────────────────────────────────
# The type is not fixed for the whole run: it starts with the aggregated reward and
# switches to the objective grid once the population actually lands.
#
# Why. objectivesPerSeed failed twice, and both times for the same underlying reason: its
# end-state objectives are only meaningful for episodes that END in a landing. Measured
# unconditionally, a network hovering over the pad won them (it stands still, above the
# pad, and survives longest) -- 23.8% of episodes ran into the step limit and the fitness
# fell from -375 to -680 over 150 generations. Conditioned on landing, the opposite
# happens: while nobody lands, all four are constant, no test case filters anything, and
# only the raw reward is left -- with the MAD tolerance, which we measured as too loose.
# That run reached -32.6 after 2000 generations with a landing rate of exactly 0.
#
# Both failures are bootstrap failures, and neither applies once the population lands
# reliably. seedsStandardized gets it there (measured 85% landing rate, validation 195);
# the grid then resolves what the aggregate cannot -- the difference between landing and
# landing cleanly on the pad, which is where the remaining 21% are lost.
# The run that reached 207.2 lands on 84.8% of seeds, and its remaining failures are a
# pure aiming problem: they touch down softly (0.50 against 0.43 for the successes, lateral
# speed of the successes exactly 0.000) but 0.66 instead of 0.24 away from the pad. In the
# aggregate reward |x_end| is one term among many, so a network can miss the pad and make
# up for it elsewhere. The grid makes it a test case of its own.
#
# Both earlier attempts with the grid failed at the BOOTSTRAP, not at the idea: without
# landings its end-state criteria are either won by hovering (measured: 23.8% of episodes
# ran into the step limit) or constant across the population and filter nobody (landing
# rate exactly 0 after 2000 generations). At 84.8% neither applies -- which is why the
# switch happens only once selectionSwitchLandingRate is reached.
selectionSchedule = ["seedsStandardized", "objectivesPerSeed"]
# Landing rate of the elite over the reporting seeds at which the next stage begins.
# Below this the grid has too few landings to resolve anything.
selectionSwitchLandingRate = 0.70
selectionSwitchWindow = 25       # generations the rate has to hold, so one lucky batch
                                 # does not trigger the switch
selectionStage = 0

lexicaseType = selectionSchedule[0]   # current type; advanced by the schedule below

# Meaning of the epsilon passed to lexicaseSelection() depends on the type:
#   "seedsStandardized" -> tolerance in STANDARD DEVIATIONS of that seed's rewards
#   "seedsRank"         -> number of RANK POSITIONS that count as a tie
#   all others          -> reward margin; negative selects the automatic MAD tolerance
#
# Why not the plain MAD of "seeds": it is a spread estimate, and the reward is bimodal
# (landed ~+200 against crashed ~-200). The spread there mostly measures the distance
# between the two modes, not the resolution among the near-equal candidates selection
# actually has to separate -- epsilon comes out far too large and the test case barely
# filters.
#
# Why standardising and not ranking: both decouple the pressure from the shape and scale
# of the reward and both remove the seed difficulty. But an ABSOLUTE rank cut collapses
# the pool to epsilon+1 candidates after a SINGLE test case, after which the remaining 19
# seeds filter nobody -- one seed decides the parent and lexicase loses the property it is
# used for. Standardising keeps filtering along the whole chain, and unlike a rank
# transform it keeps the magnitudes, which is what epsilon is meant to judge. This is also
# what the epsilon-lexicase literature recommends (La Cava et al.).
#
# Standardising alone does NOT resolve the bimodality: at 0.5 sigma (~125 reward here)
# everyone who landed at all still passes every test case, so the filter only ever
# separates "landed" from "crashed". A small tolerance is what makes a test case
# discriminate WITHIN the landers, which is where the population sits.
#   0.5  -> reached ~185 and then went flat
#   0.1  -> still climbing after 1496 generations, the best run so far
# 0.05 continues that direction. Watch the diversity indicator in the log (number of
# distinguishable individuals) -- if it collapses to single digits, this is too tight and
# 0.1 was the better setting.
lexicaseEpsilon = 0.1

eliteSize = 2            # as in the run that reached 195
edgeMutationK = 2.0      # was 1.0 -- per-edge rate is k/N
# Expected number of mutated BOUNDARIES per individual and generation (k/N inside
# Node::boundaryMutationNormal, so it is a count, not a probability). Raised from 1: at 1 a
# given threshold was revisited only every 50-150 generations, which makes the position of
# the decision cuts an almost frozen degree of freedom. sigma stays at 0.2.
boundaryMutationK = 1.0
# ── Crossover ────────────────────────────────────────────────────────────────
# "semantic"   -- role-based: nodes are matched by (type, f, edges.size()) and paired
#                 within that role by the distance of their inner boundaries. A node is
#                 only transferred if ALL its edges translate through the matching table
#                 (all-or-nothing), so a gene always travels together with its wiring.
# "uniform"    -- by array position; degrades as networks grow and shrink
crossoverType = "semantic"

# CAREFUL, the meaning of this depends on crossoverType:
#   "uniform"                          -> rate PER NODE POSITION; every pair recombines
#   "semantic"/"onepoint"            -> gate PER PAIR; the pair recombines or it does not
#
# For "semantic" this is only the FIRST of two rates: it decides WHETHER a pair
# recombines. How much then moves inside that pair is crossoverNodeRate below. Until the
# two were separated, one value was applied twice and the effective rate per matched node
# was its square -- the run that reached a record of 185 nominally ran at 0.05, i.e.
# 0.0025 effective.
# Dose-response measured over three runs (effective rate is the product of the two):
#   0.001 / 0.001  -> record  ~85   (control, crossover effectively off)
#   0.05  / 0.05   -> record ~185   (best)
#   0.15  / 0.15   -> record ~105   (too much exchange, past the peak)
crossoverProbability = 0.05

# Only used by "semantic": share of the role-matched nodes transferred inside a pair that
# passed the gate above. Set explicitly so the two rates cannot silently collapse into one
# again (a negative value would restore the old coupling). Equal to crossoverProbability
# reproduces the historical behaviour exactly.
#   0.05 / 0.05  -> the 185-record run, the best of the three settings tried
#   0.15 / 0.15  -> 9x its effective rate, measurably worse
crossoverNodeRate = 0.05

# Only used by "semantic". Maximum normalised distance between the inner boundaries of
# two same-role nodes for them to still count as the same gene. Through the
# all-or-nothing rule this governs the node AND all its successors: a gene only travels
# if its whole direct neighbourhood has a close counterpart. 1.0 pairs everything within
# a role (that was the first, far too permissive test); lower values demand that both
# nodes actually cut the feature at a similar place. Watch the crossover panel in the
# plot -- if "uebersprungen" dominates, this is too tight.
boundaryTolerance = 1.0
# Dormant nodes take no part in the matching, so no dead weight is exchanged.
matchOnlyUsed = True
multiObjective = False 
nObjectives = 1
useLineageFitness = False  # zentraler Schalter: steuert Tracking (gymnasiumMultiSeed),
                          # Selektion (tournamentSelection) UND ob der Lineage-Fitness-
                          # Plot/Log unten mit ausgegeben wird.
minLineageN = 20  # Mindestanzahl Beobachtungen, ab der lineageMean in der Turnierauswahl
                  # statt der rohen Fitness verwendet wird (siehe tournamentSelection()).
lineagePriorCapMutation = 75  # priorCap fuer capLineageAfterMutation() -- wird unten am
                  # Aufrufort UND fuer die steady-state-Diagnose in Plot/Log verwendet,
                  # damit beide immer synchron bleiben.
lineageZ = 1.0  # Konfidenz-Multiplikator fuer die untere Konfidenzgrenze (LCB) des
                # Lineage-Mittelwerts (siehe Network::lineageLCB()): 0.0 = reiner
                # lineageMean (bisheriges Verhalten), >0 bestraft Linien mit kleinem
                # lineageN/hoher Varianz automatisch staerker -- verhindert, dass frisch
                # ueber minLineageN gekommene "Gluecks-Neulinge" durch Stichprobenrauschen
                # etablierte Champions bei der Elite-/Turnierauswahl verdraengen
                # (Winner's-Curse-Fix, siehe Analyse zu konstantem n=priorCap+maxSeeds).
poolSize = 10_000
stageTimer = 10000

# ## Gleichverteilte Impuls-Richtung (uniformDirectionCurriculum)
#
# Statt der natuerlichen (quadratisch verteilten, siehe Gymnasium-Quellcode)
# Zufallsrichtung des Anfangsimpulses bekommt jeder Seed eines Batches eine
# EXPLIZIT vorgegebene Richtung, sodass jede Generation garantiert alle
# Richtungen (0-360 Grad) gleichmaessig abdeckt -- unabhaengig vom Zufall.
# Motivation: Richtung ist die letzte grosse unkontrollierte Streuquelle (siehe
# Analyse zu Magnitude/Winkelgeschwindigkeit, die bereits fest sind). Da damit
# JEDE Generation bereits eine repraesentative Stichprobe aller Richtungen
# sieht, wird KEIN Gedaechtnis mehr ueber vergangene Generationen benoetigt --
# useLineageFitness ist daher bewusst deaktiviert (siehe oben), und die
# Seed-Hall-of-Shame (hardSeedFraction/seedStats) wurde aus demselben Grund
# entfernt: sie beruhte auf einer stabilen Seed->Schwierigkeit-Identitaet, die
# es nun nicht mehr gibt (dieselbe Seed-Nummer bekommt je nach Position im
# Batch/Generation eine andere Richtung zugewiesen).
uniformDirectionCurriculum = False # Parameter zur Nutzung: False = alte, natuerliche
                                    # (seed-zufaellige) Richtung.
absoluteImpulseCurriculum = False 
# SURVIVAL MODE: Network::fitGymnasium() ignoriert Gymnasiums eingebautes Reward
# (Distanz-/Geschwindigkeits-/Winkel-Shaping, Treibstoffkosten, Landing-Bonus,
# Crash-Strafe) komplett -- Fitness = Anzahl ueberlebter Steps. Sowohl Crash als
# auch eine (unerwuenschte) Landung beenden die Episode und stoppen damit die
# Belohnung gleichermassen, wodurch die Evolution ausschliesslich auf "so lange
# wie moeglich in der Luft/im Frame ueberleben" optimiert, ohne jeden Anreiz das
# Landefeld anzusteuern. Alle frueheren Reward-Shaping-Schalter (disableFuelCost,
# usePotentialShaping, useAirborneBalanceShaping, useSoftLandingBonus) wurden
# entfernt -- es gibt nur noch diese eine Variante, konsistent fuer Training
# (gymnasiumMultiSeed) UND Video-Rendering (gymnasium).

# ── Seed stratification by the direction of the initial impulse ──────────────
# Measured (analyzeSeedDifficulty, 600 seeds): the direction of the reset impulse explains
# a large part of the difficulty. Landing rate by 45 degree sector ranged from 52% (135-180
# degrees) to 95% (225-270), a spread of 43 percentage points, while the impulse MAGNITUDE
# barely separated landed from crashed at all.
#
# A random panel therefore over- or under-represents the hard sector by chance -- with 10
# to 30 seeds that is a real source of variance between generations. Stratifying draws the
# panel so that every sector appears equally often.
#
# Note this SELECTS seeds, it does not modify them: the environment, the natural impulse
# magnitude and the episode itself stay exactly as they are. That is the difference to
# uniformDirectionCurriculum, which overwrites the direction inside the environment (and is
# coupled to absoluteImpulseCurriculum, which would also force the magnitude).
directionSectorCount = 8         # 45 degree sectors
seedDirectionCache = {}          # seed -> sector, so a seed is reset at most once

def seedDirectionSector(seedValue):
    """Sector of a seed's initial impulse. One env.reset(), no episode -- cheap."""
    if seedValue not in seedDirectionCache:
        obs, _ = directionProbeEnv.reset(seed=seedValue)
        angle = math.degrees(math.atan2(float(obs[3]), float(obs[2]))) % 360.0
        seedDirectionCache[seedValue] = int(angle // (360.0 / directionSectorCount))
    return seedDirectionCache[seedValue]

def drawSeeds(nSeeds):
    if not stratifySeedsByDirection:
        return random.sample(trainingSeedPool, nSeeds)
    return stratifiedSeeds(nSeeds, [])

def stratifiedSeeds(nNew, alreadyInPanel):
    """Draw exactly nNew seeds so the WHOLE panel ends up evenly spread over the sectors.

    The quota is computed for the full panel (kept seeds plus the new ones), then reduced
    by what the kept seeds already cover. What remains is the shortfall the new seeds
    should fill. A seed is accepted only if its sector still has a shortfall.

    The loop counts the seeds drawn, NOT the remaining quota: with a small rotation the
    kept seeds can already satisfy every sector, and a quota-driven loop would then return
    nothing and silently shrink the panel (this happened -- 24 seeds became 2 over 30
    rotations). Whatever the quota says, exactly nNew seeds come back.
    """
    total = len(alreadyInPanel) + nNew
    quota = [total // directionSectorCount] * directionSectorCount
    for sector in random.sample(range(directionSectorCount),
                                total % directionSectorCount):
        quota[sector] += 1
    for seedValue in alreadyInPanel:
        quota[seedDirectionSector(seedValue)] -= 1

    chosen = []
    taken = set(alreadyInPanel)
    attempts = 0
    maxAttempts = 300 * max(1, nNew)
    while len(chosen) < nNew and attempts < maxAttempts:
        attempts += 1
        candidate = random.choice(trainingSeedPool)
        if candidate in taken:
            continue
        sector = seedDirectionSector(candidate)
        if quota[sector] <= 0:
            continue
        quota[sector] -= 1
        taken.add(candidate)
        chosen.append(candidate)

    # Safety net: a sector can be rare enough that the shortfall is not covered within the
    # attempt budget. Any seed is then better than a short panel -- a slightly unbalanced
    # panel still beats one that keeps losing seeds.
    while len(chosen) < nNew:
        candidate = random.choice(trainingSeedPool)
        if candidate in taken:
            continue
        taken.add(candidate)
        chosen.append(candidate)

    random.shuffle(chosen)
    return chosen

def rotateSeedPanel(panel, nReplace):
    """Drop the nReplace oldest seeds, append the same number of fresh ones (FIFO).

    With stratification the replacements are chosen so that the WHOLE panel stays balanced
    over the sectors, not just the new seeds -- otherwise the kept seeds would drift the
    balance over time.
    """
    if nReplace <= 0:
        return list(panel)
    kept = list(panel[nReplace:])
    if stratifySeedsByDirection:
        return kept + stratifiedSeeds(nReplace, kept)
    fresh = []
    while len(fresh) < nReplace:
        candidate = random.choice(trainingSeedPool)
        if candidate not in kept and candidate not in fresh:
            fresh.append(candidate)
    return kept + fresh

def drawDirectionAngles(nSeeds, excludedBands=None):
    # n gleichmaessig ueber den ERLAUBTEN Teil des Kreises verteilte Winkel, mit
    # zufaelligem Rotations-Offset (verhindert, dass exakt dieselben
    # Diskretisierungs-Grenzen jede Generation wiederkehren) und zufaelliger
    # Zuordnung zu den Batch-Positionen (verhindert eine feste Korrelation
    # zwischen Position im Batch und Winkel).
    #
    # excludedBands: optionale Liste von (start, ende)-Tupeln in RADIANT
    # (0 <= start < ende <= 2*pi, kein Wraparound ueber die 0/2*pi-Grenze
    # innerhalb eines einzelnen Bandes -- ein Band, das ueber 0 hinausgehen
    # soll, muss als zwei separate Baender uebergeben werden). In diesen
    # Baendern wirkt KEIN Richtungsimpuls -- sie werden aus der Ziehung
    # komplett ausgeschlossen. Wird excludedBands nicht uebergeben (None/leer),
    # gilt weiterhin der volle Kreis [0, 2*pi) wie bisher.
    if not excludedBands:
        offset = random.uniform(0, 2 * math.pi)
        angles = [(2 * math.pi * i / nSeeds + offset) % (2 * math.pi) for i in range(nSeeds)]
        random.shuffle(angles)
        return angles

    # Baender normalisieren (auf [0, 2*pi) clampen, sortieren) und
    # ueberlappende Baender zu einem gemeinsamen Band verschmelzen.
    bands = sorted((max(0.0, s), min(2 * math.pi, e)) for s, e in excludedBands if e > s)
    merged = []
    for s, e in bands:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Erlaubte Teilintervalle = Komplement der verschmolzenen Baender.
    allowed = []
    prev = 0.0
    for s, e in merged:
        if s > prev:
            allowed.append((prev, s))
        prev = max(prev, e)
    if prev < 2 * math.pi:
        allowed.append((prev, 2 * math.pi))

    totalLength = sum(e - s for s, e in allowed)
    if totalLength <= 0:
        raise ValueError("excludedBands decken den gesamten Kreis ab -- keine erlaubten Richtungen uebrig")

    # Stratifizierung erfolgt entlang der erlaubten Bogenlaenge (nicht entlang
    # des vollen Winkels), damit die Verteilung innerhalb der erlaubten
    # Bereiche weiterhin gleichmaessig bleibt.
    offset = random.uniform(0, totalLength)
    angles = []
    for i in range(nSeeds):
        s_param = (totalLength * i / nSeeds + offset) % totalLength
        cum = 0.0
        angle = allowed[-1][1]  # Fallback (numerische Rundung am Ende)
        for a_start, a_end in allowed:
            length = a_end - a_start
            if s_param < cum + length:
                angle = a_start + (s_param - cum)
                break
            cum += length
        angles.append(angle)
    random.shuffle(angles)
    return angles

# ## Simulation

envKwargs = dict(id="LunarLander-v3", continuous=False, gravity=-10.0,
                 enable_wind=False, wind_power=15.0, turbulence_power=0.5)
env = gym.make(**envKwargs)
# Own environment for seedDirectionSector(): resetting the training env between
# generations would leave it in a different state than the evaluation expects.
directionProbeEnv = gym.make(**envKwargs)

# ── Process-parallel evaluation ──────────────────────────────────────────────
# The C++ core calls back into Python for every Gym step, so the GIL serialises
# threads; only separate processes actually run in parallel. Individuals within a
# generation are independent, so they are split across workers, each with its own
# environment. Verified to reproduce the sequential fitness, fitnessValues,
# lexicaseObjectives, objectivesPerSeed and used flags exactly.
#
# Measured on this machine (4 physical cores, 8 logical), 100 individuals x 30 seeds:
#   sequential 10.98 s -> parallel 2.54 s  (4.3x); over full generations 2.5x.
#
# Verified: 40 generations of this script reproduce the sequential run exactly (training
# and validation curves compared as exact float reprs, not rounded).
#
# Getting there required carrying state that Network's pickle drops -- per-seed
# fitness, the objective vectors, the traversal flags and everything that accumulates
# across generations (traverseCounter, lastVisitStep, lineage statistics). A stale value
# in the parent changes mutation and selection one generation later while every fitness
# value stays bit-identical, so compare EXACTLY when checking this, never rounded.
# Population::rngState() pins such a divergence down to the operator that causes it.
useMultiprocessing = True
nWorkers = 8

evaluator = None
if useMultiprocessing:
    evaluator = ParallelEvaluator(envKwargs, populationKwargs, workers=nWorkers)


def evaluatePopulation(targetPop, seeds, **kwargs):
    """Evaluate targetPop on seeds -- across processes if enabled, else in-process."""
    if evaluator is not None and len(targetPop.individuals) > 1:
        evaluator.evaluate(targetPop, seeds=seeds, **kwargs)
    else:
        targetPop.gymnasiumMultiSeed(env, seeds=seeds, **kwargs)


start = time.perf_counter()

# ── Honest fitness curve: the elite is re-scored on seeds it was not selected on ──
# Measured (research/seed_noise_study.py): the champion of a generation is overrated by
# 27-38 reward, and the reported best fitness moves by sd 15-24 between generations
# without anything actually changing -- as much as the whole observed movement. The
# training fitness is therefore unusable for judging progress or for steering anything.
# A separate population holds copies of the elite, so re-scoring never touches the
# fitness/fitnessValues of the real population (assignment copies the networks).
popValidationElite = fn.Population(
    seed=seed,
    ni=1,
    jn=8,
    jnf=8,
    pn=4,
    pnf=4,
    fractalJudgment=False,
    useExperience=False,
    nFeatureValues=[5,5,5,5,5,5,2,2]
)
# Elite re-evaluation needs two copies of each candidate: one that gets re-scored (its
# fitness/fitnessValues are overwritten by the fresh seeds) and one that keeps its
# training state and is the copy actually placed into the elite slot.
popEliteScore = fn.Population(
    seed=seed, ni=1, jn=8, jnf=8, pn=4, pnf=4,
    fractalJudgment=False, useExperience=False, nFeatureValues=[5,5,5,5,5,5,2,2]
)
popEliteKeep = fn.Population(
    seed=seed, ni=1, jn=8, jnf=8, pn=4, pnf=4,
    fractalJudgment=False, useExperience=False, nFeatureValues=[5,5,5,5,5,5,2,2]
)
eliteHonestProgess = []     # honest score of the re-evaluated elite[0], per generation
eliteRankShiftProgess = []  # how far the honest ranking moved the training ranking
# The champion lives in its own Population: pop.individuals is an opaque vector, so
# indexing it yields a REFERENCE, and holding that across generations dangles once the
# vector is reassigned (this segfaulted). Assigning into a population's individuals copies,
# so popHallOfFame owns its entry and stays valid.
popHallOfFame = fn.Population(
    seed=seed, ni=1, jn=8, jnf=8, pn=4, pnf=4,
    fractalJudgment=False, useExperience=False, nFeatureValues=[5,5,5,5,5,5,2,2]
)
hasChampion = False         # popHallOfFame.individuals[0] is only meaningful once this is True
# ── Archive of the best individual ever validated ────────────────────────────
# Reporting only -- this NEVER feeds back into the evolution (that was the hall of fame,
# which did not help). The green "best so far" curve is the maximum over all validation
# values, but the individual behind it is not held anywhere: elite slots are recomputed
# every generation, so the champion of generation 700 is usually long gone by the end.
# The final 1000-seed validation then measures whatever happens to be best at generation
# 2000, not the best the run ever produced.
#
# Selection happens on the 100 reporting seeds, so the archived individual's score there is
# optimistic (a maximum over ~2000 noisy values). The final validation runs on 1000
# INDEPENDENT seeds though, so that number is unbiased for this individual -- it answers
# "how good is the net that looked best on the 100 seeds", which is exactly the question.
# Its own Population, for the same reason the hall of fame needs one: indexing the opaque
# individuals vector yields a reference that dangles once the vector is reassigned.
popBestEver = fn.Population(
    seed=seed, ni=1, jn=8, jnf=8, pn=4, pnf=4,
    fractalJudgment=False, useExperience=False, nFeatureValues=[5,5,5,5,5,5,2,2]
)
hasBestEver = False
bestEverScore = None        # its score on the 100 reporting seeds
bestEverGeneration = None   # when it was archived

hallOfFameScore = None      # its score on the seeds of the generation it last won on
hallOfFameHeldProgess = []  # 1 if the champion defended its title this generation, else 0

landingRateProgess = []     # landing rate of elite[0] on the reporting seeds
validationProgess = []      # honest score of the best individual, per generation
validationEliteMeanProgess = []  # mean over the re-scored elite, per generation
validationBestSoFarProgess = []  # running maximum of validationProgess (see plot comment)
# Panel overfitting probe, free of charge: the last seedPanelRotation entries of the
# panel were introduced THIS generation and nobody has been selected on them yet, while
# the older ones have been optimised against for up to maxSeeds/seedPanelRotation
# generations. A systematic gap between the two groups is the price of the panel.
panelFreshGapProgess = []

fitnessProgess = []
fitnessProgessII = []
fitnessProgessIII = []
fixedProgess = []
objectivesPerIndvidualProgress = []
variableProgress = []
lineageMeanProgress = []
lineageNProgress = []
lineageSuccessRateProgress = []
# ── Netzgroessen-Verlauf (siehe Groessen-Plot weiter unten) ──────────────────────
# Pro Generation wird die Verteilung der Knotenzahlen ueber die Population zu einer
# Handvoll Kennzahlen verdichtet, statt alle Groessen zu speichern: Median und das
# 10.-90.-Perzentil zeigen, wohin sich die Population als Ganzes bewegt, das Maximum
# deckt einzelne aufgeblaehte Netze auf (seedSpecialist haengt ganze Sub-Graphen an),
# und die Groesse des besten Individuums beantwortet die eigentliche Frage: waechst
# die Loesung mit, oder waechst nur der Ballast?
networkSizeMedianProgress = []
networkSizeP10Progress = []
networkSizeP90Progress = []
networkSizeMaxProgress = []
networkSizeBestProgress = []
# Anteil nie betretener Knoten -- die Groesse allein sagt nicht, ob ein Netz gewachsen
# ist, WEIL es mehr rechnet, oder weil dormante Transplantate liegen bleiben.
unusedShareProgress = []
# Crossover-Diagnose je Generation (nur type="semantic" befuellt sie, siehe
# Population::crossoverNodes*). "matched" ist das Angebot, "exchanged" der tatsaechliche
# Austausch, "skipped" das, was die Alles-oder-nichts-Regel verworfen hat.
crossoverMatchedProgress = []
crossoverExchangedProgress = []
crossoverSkippedProgress = []
nSameFitnessPerSeed = [0 for _ in range(maxSeeds)]
fitnessPerSeedBefore = [worstFitness for _ in range(maxSeeds)]
g = 0
totalSeeds = maxSeeds
markerPoolExtension = []

plt.ion()  # Interaktiven Modus aktivieren
fig = plt.figure(figsize=(12, 11))
if multiObjective:
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(2, 2, (1, 2), projection='3d')  # obere Reihe, volle Breite
    ax2 = fig.add_subplot(2, 3, 4)
    ax3 = fig.add_subplot(2, 3, 5)
    ax4 = fig.add_subplot(2, 3, 6)
else:
    # Feste Grenzen der Seed-Heatmap: worstFitness ist der Boden, den fitGymnasium() fuer
    # Abstuerze/ungueltige Netze vergibt, +300 liegt knapp ueber dem, was eine saubere Landung
    # in LunarLander einbringt (~200-300). Der Farbwechsel liegt bei 0 -- die inhaltlich
    # richtige Grenze zwischen "hat sich geschadet" und "hat etwas erreicht" -- statt in der
    # geometrischen Mitte der Skala (-100), wo er inhaltlich nichts bedeutet.
    heatVmin = worstFitness   # -500
    heatVmax = 300
    heatNorm = mcolors.TwoSlopeNorm(vmin=heatVmin, vcenter=0.0, vmax=heatVmax)

    ax = fig.add_subplot(4, 1, 1)
    axSize = fig.add_subplot(4, 1, 2, sharex=ax)  # visualisiert die Netzgroessen ueber die Generationen
    axSizeRight = axSize.twinx()                  # Anteil dormanter Knoten, einmal erzeugt und
                                                  # bei jedem Update nur geleert (clear()), damit
                                                  # sich keine Twin-Achsen anhaeufen
    # Crossover-Wirkung: unterscheidet "der Operator bewirkt nichts" von "der Operator
    # feuert nie". Bei type="semantic" wandert ein Knoten nur, wenn er UND alle seine
    # Nachfolger eine Entsprechung innerhalb von boundaryTolerance haben -- wird
    # "uebersprungen" zur dominanten Kurve, ist die Toleranz zu eng.
    axCross = fig.add_subplot(4, 1, 3, sharex=ax)
    axCrossRight = None   # Twin-Achse fuer "zugeordnet"; vor jedem Update entfernt, sonst
                          # haeufen sich bei jedem Plot unsichtbare Achsen an
    axHeat = fig.add_subplot(4, 1, 4)          # visualisiert Populations-Erfolg je Seed (aktuelle Generation)
    axRightSuccess = None  # twinx() fuer Lineage-Erfolgsrate, wird bei jedem Plot-Update
    axRightN = None        # neu erzeugt -- vorherige Instanz muss vorher entfernt werden
    heatColorbar = None    # Colorbar der Seed-Heatmap, muss vor jedem Update entfernt werden
                            # (siehe Plot-Block), sonst haeufen sich Twin-Achsen an.

done = False
seeds = drawSeeds(maxSeeds)   # initial panel; rotated by seedPanelRotation each generation
maxSteps = 1000
curriculumLevel = 1  # startet mit reduziertem Anfangsimpuls (siehe GymnasiumWrapper.hpp),
                       # steigt weiter unten automatisch an, sobald moving_avg > 200
recoveryPause = 0
survivalMode = False # siehe oben: Gymnasium-Reward wird ignoriert, Fitness = ueberlebte Steps
landingQuote = False 
# SEED-SPECIALIST CROSSOVER (Population::crossover(type="seedSpecialist")):
# Fuer jedes Individuum (Host, auch Elite) wird populationsweit der Donor gesucht, der auf
# den Seeds, wo der Host schwaechelt, am meisten besser abschneidet (hoechste summierte
# positive fitnessValues-Differenz). Dessen tatsaechlich durchlaufener Sub-Graph auf genau
# diesen Defizit-Seeds (Network::visitedNodesPerSeed) wird additiv an den Host angehaengt --
# der Host verliert dabei nichts, der Donor bleibt unveraendert. Der neue Block bleibt
# zunaechst unerreichbar/fitness-neutral ("Junk-DNA") und wird erst durch eine spaetere
# zufaellige Mutation (edgeMutation()) ueber seinen einzigen Entry-Knoten erschlossen.
# Rein additives Feature -- per Flag komplett abschaltbar, ohne bestehende Crossover-Typen
# (onepoint/randomWidth) zu beeinflussen.
useSeedSpecialistCrossover = False
# Zwei Varianten, gleiche Donor-Suche, unterschiedlicher Einbau:
#   "seedSpecialistAppend"  -- der Donor-Sub-Graph wird als dormanter Block ANGEHAENGT und
#                              erst durch eine spaetere Mutation ueber seinen Entry-Knoten
#                              erschlossen. Fitness-neutral, laeuft auch fuer Elite, laesst
#                              die Netze aber wachsen.
#   "seedSpecialistReplace" -- der Donor-Sub-Graph ERSETZT den defizit-exklusiven Sub-Graphen
#                              des Hosts (die Knoten, die der Host nur auf den schlechten
#                              Seeds benutzt). Das Backbone bleibt erhalten, die Groesse
#                              aendert sich nur um die Differenz der beiden Sub-Graphen.
#                              Destruktiv, deshalb werden Elite-Hosts uebersprungen.
seedSpecialistType = "seedSpecialistReplace"
seedSpecialistProtection = 5  # Generationen, waehrend derer der transplantierte Block vor
                                # weiterem (onepoint/randomWidth/clusterAware) Crossover
                                # geschuetzt ist (siehe crossoverProtection/generationReceived).
adaptiveK = False
stage = 0
for g in range(generations):

    if g > 0:
        # seedPanelRotation == 0 means a full redraw -- see the rotation block above.
        seeds = (drawSeeds(maxSeeds) if seedPanelRotation <= 0
                 else rotateSeedPanel(seeds, seedPanelRotation))
    if  g == 0:
        directionAngles = (
            drawDirectionAngles(
                len(seeds),
                excludedBands=[
                    # (math.radians(60), math.radians(120)),
                    # (math.radians(240), math.radians(300)),
                ],
            )
            if uniformDirectionCurriculum
            else []
        )

    if g % stageTimer == 0 and g > 0:
        maxSteps += 100

    if done:
        break

    # 1. Multi-Seed Evaluation 
    evaluatePopulation(
        pop,
        dMax=10,
        maxSteps=maxSteps,
        maxConsecutiveP=5,
        worstFitness=-500,
        seeds=seeds,
        curriculumLevel=curriculumLevel,
        useLineageFitness=useLineageFitness,
        uniformDirectionCurriculum=uniformDirectionCurriculum,
        absoluteImpulseCurriculum=absoluteImpulseCurriculum,
        # directionAngles=directionAngles,
        survivalMode=survivalMode,
        landingQuote = landingQuote
    )

    # Grace-period counters must advance while Node::used still holds THIS generation's
    # flags -- the next evaluation clears them (fitGymnasium(newRun=True)).
    if nodeGracePeriod >= 0:
        pop.callAgeUnusedNodes()

    if multiObjective:
        # 2. Pareto Objectives berechnen
        pop.calculateParetoObjectives(landingThreshold=100)
        # 3. Pareto Tournament Selection
        pop.paretoTournamentSelection(N=5, E=5)
    else:
        # ── Elite re-evaluation ──────────────────────────────────────────────
        # Candidates are pre-selected by training fitness (that filter is still noisy, so
        # eliteCandidates is deliberately larger than eliteSize), then re-scored on seeds
        # drawn fresh this generation and disjoint from the current panel. Only the winners
        # of THAT comparison get the elite slots. Parent selection is untouched: lexicase
        # reads fitnessValues/objectivesPerSeed, never fitness.
        eliteOrder = []
        if eliteReevaluation:
            trainFitness = [ind.fitness for ind in pop.individuals]
            candIdx = sorted(range(len(trainFitness)),
                             key=lambda i: trainFitness[i], reverse=True)[:eliteCandidates]

            eliteSeeds = [s for s in drawSeeds(eliteSeedCount + maxSeeds)
                          if s not in seeds][:eliteSeedCount]

            # The reigning champion joins the field as an extra candidate, whatever its
            # training fitness says. That is the whole point of the hall of fame: it can
            # only lose to someone who beat it on the SAME seeds, never by dropping out of
            # a noisy pre-selection.
            candidates = [pop.individuals[i] for i in candIdx]
            championIndex = None
            if hallOfFame and hasChampion:
                championIndex = len(candidates)
                candidates.append(popHallOfFame.individuals[0])

            popEliteKeep.individuals = candidates
            popEliteScore.individuals = candidates
            evaluatePopulation(
                popEliteScore,
                dMax=10,
                maxSteps=maxSteps,
                maxConsecutiveP=5,
                worstFitness=worstFitness,
                seeds=eliteSeeds,
                curriculumLevel=curriculumLevel,
                useLineageFitness=False,
                uniformDirectionCurriculum=uniformDirectionCurriculum,
                absoluteImpulseCurriculum=absoluteImpulseCurriculum,
                survivalMode=survivalMode,
                landingQuote=landingQuote
            )
            honestScores = [ind.fitness for ind in popEliteScore.individuals]
            eliteOrder = sorted(range(len(honestScores)),
                                key=lambda j: honestScores[j], reverse=True)
            eliteHonestProgess.append(honestScores[eliteOrder[0]])
            # How far the fresh seeds moved the training ranking: 0 means the training
            # batch already had the honest winner on top. The champion sits outside that
            # ranking, so it is reported as the last position.
            eliteRankShiftProgess.append(float(eliteOrder[0]))

            if hallOfFame:
                # The winner of this comparison becomes the champion. Everyone was scored
                # on the same freshly drawn seeds, so the comparison is fair; the score is
                # kept only for reporting, never carried over as a threshold (the seeds
                # change, so an old score would not be comparable).
                winner = eliteOrder[0]
                hallOfFameHeldProgess.append(
                    1 if (championIndex is not None and winner == championIndex) else 0)
                # Assigning into .individuals copies the network, so the champion survives
                # the next reassignment of popEliteKeep.
                popHallOfFame.individuals = [popEliteKeep.individuals[winner]]
                hasChampion = True
                hallOfFameScore = honestScores[winner]

        pop.lexicaseSelection(
            E=eliteSize,
            type=lexicaseType,
            # -1 keeps the MAD automatic for the plain value-based types; the two
            # scale-free variants take their tolerance in their own unit (see above).
            # lexicaseType changes during the run (see the selection schedule), so the
            # unit of epsilon changes with it: sigma for the standardised type, rank
            # positions for the rank type, and the MAD automatic for the grid -- whose
            # test cases have very different scales (reward, speeds 0-4, a 0/1 flag) and
            # therefore need a tolerance computed per test case.
            epsilon=(lexicaseEpsilon
                     if lexicaseType in ("seedsStandardized", "seedsRank") else -1.0)
        )

        if eliteReevaluation and eliteOrder:
            # Overwrite the provisional elite (chosen on the training batch) with the
            # individuals that actually won on the fresh seeds. The copies come from
            # popEliteKeep, so they carry their TRAINING fitnessValues -- the heatmap,
            # the per-seed printout and the crossover partner criterion keep working.
            for slot, j in zip(pop.indicesElite, eliteOrder[:len(pop.indicesElite)]):
                pop.individuals[slot] = popEliteKeep.individuals[j]

    best = pop.individuals[pop.indicesElite[0]]
    maxFitness = best.fitness

    # ── Honest score of this generation's elite ──────────────────────────────
    # Runs on a fixed seed set the elite was NOT selected on, in a separate
    # population (the assignment copies the networks, so pop.individuals keeps its
    # training fitness/fitnessValues for the heatmap, printing and crossover).
    nValidate = min(validationEliteCount, len(pop.indicesElite))
    popValidationElite.individuals = [pop.individuals[i] for i in pop.indicesElite[:nValidate]]
    evaluatePopulation(
        popValidationElite,
        dMax=10,
        maxSteps=maxSteps,
        maxConsecutiveP=5,
        worstFitness=worstFitness,
        seeds=validationSeeds,
        curriculumLevel=curriculumLevel,
        useLineageFitness=False,
        uniformDirectionCurriculum=uniformDirectionCurriculum,
        absoluteImpulseCurriculum=absoluteImpulseCurriculum,
        survivalMode=survivalMode,
        landingQuote=landingQuote
    )
    eliteValidationScores = [ind.fitness for ind in popValidationElite.individuals]
    # Landing rate of elite[0] over the reporting seeds -- the quantity the selection
    # schedule below switches on, and the one the objective grid needs to be meaningful.
    eliteSeedRewards = list(popValidationElite.individuals[0].fitnessValues)
    landingRateProgess.append(
        sum(1 for r in eliteSeedRewards if r > 100.0) / max(1, len(eliteSeedRewards)))
    # Deliberately the score of elite[0], not the maximum over the elite: taking the
    # max would re-introduce the same selection bias the validation is meant to remove.
    validationProgess.append(eliteValidationScores[0])
    validationEliteMeanProgess.append(statistics.mean(eliteValidationScores))
    validationBestSoFarProgess.append(max(validationProgess))

    # Archive the individual behind the "best so far" curve. Assigning into .individuals
    # copies, so the archived net survives the reassignment of popValidationElite.
    if not hasBestEver or validationProgess[-1] > bestEverScore:
        popBestEver.individuals = [popValidationElite.individuals[0]]
        hasBestEver = True
        bestEverScore = validationProgess[-1]
        bestEverGeneration = g

    # ── Selection schedule: switch once the population reliably lands ────────
    # The objective grid needs landings to be able to resolve anything: without them its
    # end-state criteria are constant across the population and filter nobody. The rate
    # has to hold over a whole window, so a single lucky batch does not trigger the
    # switch. The type only ever moves forward.
    if (selectionStage + 1 < len(selectionSchedule)
            and len(landingRateProgess) >= selectionSwitchWindow):
        recentLandingRate = statistics.mean(landingRateProgess[-selectionSwitchWindow:])
        if recentLandingRate >= selectionSwitchLandingRate:
            selectionStage += 1
            lexicaseType = selectionSchedule[selectionStage]
            markerPoolExtension.append(g)
            print(f"[selection schedule] generation {g}: landing rate "
                  f"{recentLandingRate*100:.1f}% over the last {selectionSwitchWindow} "
                  f"generations (>= {selectionSwitchLandingRate*100:.0f}%) "
                  f"-> selection switched to {lexicaseType}")

    # ── Adaptive rotation: how fast the panel is exchanged ───────────────────
    # Reads the generalisation gap (training fitness of elite[0] minus the same individual
    # on unseen seeds) over two consecutive windows. Growing gap -> the population is
    # adapting to the panel -> rotate faster. Stable gap -> rotate slower and regain
    # comparability between generations. See the rotation block in the configuration.
    rotationProgress.append(seedPanelRotation)
    if (rotationAdaptive
            and g - rotationLastChange >= rotationCooldown
            and len(validationProgess) >= 2 * rotationWindow):
        recentValidation = statistics.mean(validationProgess[-rotationWindow:])
        earlierValidation = statistics.mean(
            validationProgess[-2 * rotationWindow:-rotationWindow])
        # fitnessProgess and validationProgess are index-aligned: both get one entry per
        # generation for the same elite[0].
        recentGap = statistics.mean(fitnessProgess[-rotationWindow:]) - recentValidation
        earlierGap = statistics.mean(
            fitnessProgess[-2 * rotationWindow:-rotationWindow]) - earlierValidation
        gapGrowth = recentGap - earlierGap

        previousRotation = seedPanelRotation
        if gapGrowth > rotationGapGrowth:
            seedPanelRotation = min(rotationMax, seedPanelRotation + 1)
        elif gapGrowth < rotationGapStable:
            seedPanelRotation = max(rotationMin, seedPanelRotation - 1)

        if seedPanelRotation != previousRotation:
            rotationLastChange = g
            direction = "faster" if seedPanelRotation > previousRotation else "slower"
            print(f"[rotation] generation {g}: gap {round(earlierGap, 1)} -> "
                  f"{round(recentGap, 1)} (growth {round(gapGrowth, 1)}) -> rotate {direction}: "
                  f"{previousRotation} -> {seedPanelRotation} of {maxSeeds} seeds per generation")

    # ── Seed schedule: grow the panel once the honest curve stops improving ──
    if seedStage + 1 < len(seedSchedule):
        generationsInStage = g - seedStageStart
        if (generationsInStage >= seedStageMinGenerations
                and len(validationProgess) >= 2 * seedStagePlateauWindow):
            recentMean = statistics.mean(validationProgess[-seedStagePlateauWindow:])
            earlierMean = statistics.mean(
                validationProgess[-2 * seedStagePlateauWindow:-seedStagePlateauWindow])
            if (recentMean - earlierMean) < seedStagePlateauDelta:
                seedStage += 1
                seedStageStart = g
                # Grow, do not redraw: the seeds already in the panel stay, so the task
                # does not change abruptly in the generation of the switch. The added seeds
                # go through the same stratification, so the enlarged panel is balanced as
                # a whole and not just in its original part.
                nAdd = seedSchedule[seedStage] - len(seeds)
                if nAdd > 0:
                    if stratifySeedsByDirection:
                        seeds = seeds + stratifiedSeeds(nAdd, seeds)
                    else:
                        while len(seeds) < seedSchedule[seedStage]:
                            candidate = random.choice(trainingSeedPool)
                            if candidate not in seeds:
                                seeds.append(candidate)
                maxSeeds = seedSchedule[seedStage]
                markerPoolExtension.append(g)   # marks the switch in the fitness plot
                print(f"[seed schedule] generation {g}: validation flat "
                      f"({round(earlierMean, 1)} -> {round(recentMean, 1)} over "
                      f"{seedStagePlateauWindow} generations) -> panel grown to {maxSeeds} seeds")

    if seedPanelRotation > 0 and len(best.fitnessValues) == maxSeeds:
        freshMean = statistics.mean(best.fitnessValues[-seedPanelRotation:])
        establishedMean = statistics.mean(best.fitnessValues[:-seedPanelRotation])
        panelFreshGapProgess.append(establishedMean - freshMean)
    else:
        panelFreshGapProgess.append(0.0)

    # if g > 0 and best.fitnessValues:
    #     idxBestSeed = max(range(len(best.fitnessValues)), key=lambda i: best.fitnessValues[i])
    #     newSeed = random.randint(1, 10_000)
    #     while newSeed in seeds:
    #         newSeed = random.randint(1, 10_000)
    #     seeds[idxBestSeed] = newSeed

    if nObjectives == 3:
        fitnessProgess.append(best.objectives[0])
        fitnessProgessII.append(best.objectives[1])
        fitnessProgessIII.append(best.objectives[2])
    elif nObjectives == 2:
        fitnessProgess.append(best.objectives[0])
        fitnessProgessII.append(best.objectives[1])
        fitnessProgessIII.append(0)
    else:
        fitnessProgess.append(best.fitness)

    fixedProgess.append(statistics.mean(best.fitnessValues[0:maxSeeds]))
    if useLineageFitness:
        lineageMeanProgress.append(best.lineageMean)
        lineageNProgress.append(best.lineageN)
        lineageSuccessRateProgress.append(best.lineageSuccessRate)


    if multiObjective:
        if nObjectives == 3:
            objectivesPerIndvidual = [[ind.objectives[0], ind.objectives[1], ind.objectives[2]] for ind in pop.individuals]
        elif nObjectives == 2:
            objectivesPerIndvidual = [[ind.objectives[0], ind.objectives[1], 0] for ind in pop.individuals]

        objectivesPerIndvidualProgress.append(objectivesPerIndvidual)

    # ── Mutation & Crossover ──────────────────────────────────────────────────
    if useSeedSpecialistCrossover:
        # WICHTIG: "seedSpecialist" MUSS vor callAddDelNodes() laufen. Es liest
        # Network::visitedNodesPerSeed, das in gymnasiumMultiSeed() weiter oben mit
        # Node-Indizes relativ zum DAMALIGEN innerNodes-Stand befuellt wird. Wuerde
        # hier zuerst callAddDelNodes() laufen, koennte es bei jedem potenziellen
        # Donor Knoten loeschen -- das verschiebt/entfernt Indizes in innerNodes
        # (siehe Network::addDelNodes()), waehrend visitedNodesPerSeed unveraendert
        # (also veraltet) bleibt. Die spaeter kopierten "besuchten" Knoten waeren dann
        # falsch oder (nach dem Bounds-Check) komplett leer -- der Transplant faellt
        # in dem Fall still aus, ohne dass fitnessValues/visitedNodesPerSeed das anzeigen.
        # probability wirkt bei "seedSpecialist" PRO INDIVIDUUM: mit dieser Wahrscheinlichkeit
        # erhaelt ein Host in dieser Generation ueberhaupt ein Transplantat (Ziehung erfolgt vor
        # der Donor-Suche). Bei 0.05 bekommen also im Mittel nur 5% der Population einen Block.
        pop.crossover(
            probability = 0.1,
            type = seedSpecialistType,
            crossoverProtection = seedSpecialistProtection,
            lineagePriorCap = 75,
            currentGeneration = g
        )
        pop.callAddDelNodes(
            minFeatures,
            maxFeatures,
            junk=junkShare,
            noElite=True,
            currentGeneration=g,
            nodeGracePeriod=nodeGracePeriod,
            crossoverProtection = seedSpecialistProtection
        )
    else:
        pop.callAddDelNodes(
            minFeatures,
            maxFeatures,
            junk=junkShare,
            noElite=True,
            currentGeneration=g,
            nodeGracePeriod=nodeGracePeriod,
            crossoverProtection = 0
        )
        # Rollenbasiertes Crossover (siehe crossoverType oben). probability wirkt PRO PAAR:
        # mit dieser
        # Wahrscheinlichkeit wird ein (nicht beidseitig elitaeres) Paar ueberhaupt
        # rekombiniert. Der schwaechere Partner (mittlere fitnessValues) wird vom Kind
        # ersetzt, der fittere bleibt unveraendert; Elite gibt immer nur ab.
        pop.crossover(
            probability = crossoverProbability,   # Rate pro zugeordnetem Knoten, wie bei uniform
            type = crossoverType,
            crossoverProtection = 0,
            lineagePriorCap = 75,
            currentGeneration = g,
            boundaryTolerance = boundaryTolerance,
            matchOnlyUsed = matchOnlyUsed,
            nodeExchangeRate = crossoverNodeRate
        )

    crossoverMatchedProgress.append(pop.crossoverNodesMatched)
    crossoverExchangedProgress.append(pop.crossoverNodesExchanged)
    crossoverSkippedProgress.append(pop.crossoverNodesSkipped)

    if g < stageTimer:
        # Self-adaptive mutation strength using a CONTINUOUS, scale-independent seed-
        # consistency score (Population::seedConsistencyRatio): min(fitnessValues) /
        # max(fitnessValues) per individual, averaged over the population and smoothed
        # over `windowSize` generations. Meaningful from generation 1 onward -- unlike
        # threshold-based criteria (batchThresholdSuccess()) that require near-maximal
        # absolute fitness to ever register as "successful" and would stay pinned at one
        # extreme until that late-training level is reached. Lineage-based alternatives
        # (seedConsistentSuccess()) are unusable here since useLineageFitness=False keeps
        # lineageN permanently at 0 (see uniformDirectionCurriculum notes above).
        #
        # Direction is INVERTED (invert=True) relative to classical Rechenberg: our goal
        # is stability (perfect results across all seeds), not raw convergence speed.
        # With invert=True: score > target -> shrink k (decFactor, preserve/stabilize a
        # structure that already works everywhere); score < target -> grow k (incFactor,
        # explore out of an insufficient/fragile structure).
        #
        # CALIBRATION NOTE: targetSuccessRate must be close to the ACTUAL long-run average
        # of seedConsistencyRatio for k to genuinely oscillate (rather than saturating at
        # kMin/kMax) -- read pop.successRateHistory (printed as [k-debug] below) after a
        # few hundred generations and re-tune targetSuccessRate to match its observed mean.
        # windowSize is deliberately kept SHORT (not 20-30) so the rolling average reacts
        # to real swings instead of averaging them away.
        if adaptiveK == True:
            pop.updateAdaptiveK(
                successCriterion=lambda ind: fn.Population.seedConsistencyRatio(ind),
                invert=True,
                targetSuccessRate=0.5,  # start guess -- recalibrate from observed successRateHistory
                windowSize=5,
                incFactor=1.18,   # grows k (applied BELOW target when invert=True)
                decFactor=0.88,   # shrinks k (applied ABOVE target when invert=True)
                kMin=0.15,
                kMax=1
            )
        print(f"[k-debug] gen={g} adaptiveK={pop.adaptiveK:.4f} history={list(pop.successRateHistory)[-6:]}")

        pop.callEdgeMutation(
            probInnerNodes=0,
            probStartNode=0,
            justUsedNodes=True,
            k=edgeMutationK
        )
    else:
        pop.callClusterAwareEdgeMutation(0.04, freezeIntraCluster=True)

    # ── Netzgroessen erfassen (nach allen strukturveraendernden Operatoren dieser
    # Generation: crossover + callAddDelNodes; Mutation aendert die Groesse nicht) ──
    sizes = np.array(pop.networkSizes(), dtype=float)
    unused = np.array(pop.unusedNodeCounts(), dtype=float)
    networkSizeMedianProgress.append(float(np.median(sizes)))
    networkSizeP10Progress.append(float(np.percentile(sizes, 10)))
    networkSizeP90Progress.append(float(np.percentile(sizes, 90)))
    networkSizeMaxProgress.append(float(sizes.max()))
    networkSizeBestProgress.append(float(sizes[pop.indicesElite[0]]) if len(pop.indicesElite) > 0 else float(np.median(sizes)))
    unusedShareProgress.append(float((unused.sum() / sizes.sum()) * 100.0) if sizes.sum() > 0 else 0.0)

    # CAREFUL: k overrides probability inside Node::boundaryMutationNormal --
    # probability = min(1, k/N) with N = number of eligible boundaries. The probability
    # argument above is therefore ignored whenever k > 0; k is the expected number of
    # MUTATED BOUNDARIES per individual and generation.
    #
    # At k = 1 and roughly 50-150 boundaries per network (17-24 nodes with 2-10 boundaries
    # each) a single boundary is touched about every 50-150 generations. The thresholds are
    # what decides WHERE a judgment node cuts its feature -- for a controller that has to
    # react at a particular sink rate that is the governing quantity, and it was being
    # explored very slowly.
    pop.callBoundaryMutationNormal(
        probability=0,
        sigma=0.2,
        justUsedNodes=True,
        k=boundaryMutationK
    )
    #pop.callBoundaryMutationEdgeSizeDependingSigma(
    #    probability=0.05,
    #    sigma=1,
    #    justUsedNodes=True
    #)
    pop.callBoundaryMutationUniform(
        probability=0,
        justUsedNodes=True,
        k=1
    )
    #
    # pop.callBoundaryMutationFractal(
    #         probability=0.0,
    #         minF=minFeatures,
    #         maxF=maxFeatures,
    #         justUsedNodes=True,
    #         k=1
    #         )
    #
    #pop.callGammaMutation(0.03, justUsedNodes=True)
    #pop.callAlphaMutation(0.03, justUsedNodes=True)

    # Mutations-Pendant zum Crossover-priorCap: verhindert, dass reine
    # Mutations-Linien ihre lineageMean unbegrenzt akkumulieren und dadurch
    # gegenueber neuen (durch Mutation veraenderten) Beobachtungen traege werden.
    pop.capLineageAfterMutation(priorCap=lineagePriorCapMutation)

    if g%50 == 0 and g > 0:
        # ── Plot ──────────────────────────────────────────────────────────────────
        if multiObjective:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            # ── Multi-Objective: 3D upper chart + 3 objective plots below ─────────
            for idx, progess in enumerate(objectivesPerIndvidualProgress):
                x, y, z = [], [], []
                for obj in progess:
                    x.append(obj[0])
                    y.append(obj[1])
                    z.append(obj[2])
                a = (1 + idx) / len(objectivesPerIndvidualProgress)
                if idx == len(objectivesPerIndvidualProgress) - 1:
                    ax1.scatter(x, y, z, alpha=a, color="orange")
                    if nObjectives == 3:
                        ax1.scatter(
                            best.objectives[0],
                            best.objectives[1],
                            best.objectives[2],
                            alpha=a, color="red", s=80
                        )
                    elif nObjectives == 2:
                        ax1.scatter(
                            best.objectives[0],
                            best.objectives[1],
                            alpha=a, color="red", s=80
                        )

            ax1.set_xlabel("Objective 1")
            ax1.set_ylabel("Objective 2")
            ax1.set_zlabel("Objective 3")
            ax1.set_title(
                f"Fitness Progress (Generations: {g+1}) | Batches: {nBatches+1} | "
                f"Best Fitness: {round(maxFitness, 2)} | NN: {len(best.innerNodes)} | "
                f"Used: {nUsedNodes(best)} | Time.: {round(time.perf_counter()-start, 2)} | "
                f"Seeds: {len(seeds)} | "
                f"Crossover: {best.nCrossovers}"
            )

            ax2.scatter(range(len(fitnessProgess)), fitnessProgess, color='black', label="Objective 1", s=10)
            fitness_series = pd.Series(fitnessProgess)
            moving_avg = fitness_series.rolling(window=25).mean()
            ax2.plot(range(len(fitnessProgess)), moving_avg, color='orange', linewidth=1, label="MA (25)")
            moving_avgII = fitness_series.rolling(window=50).mean()
            ax2.plot(range(len(fitnessProgess)), moving_avgII, color='red', linewidth=1, label="MA (50)")
            moving_avgIII = fitness_series.rolling(window=100).mean()
            ax2.plot(range(len(fitnessProgess)), moving_avgIII, color='green', linewidth=2, label="MA (100)")

            if moving_avgIII[0] > 200:
                done = True

            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Objective 1")
            ax2.set_title("Objective 1 Progress")
            ax2.ticklabel_format(style='plain', axis='x', useOffset=False)
            ax2.legend()

            ax3.scatter(range(len(fitnessProgessII)), fitnessProgessII, color='black', label="Objective 2", s=10)
            fitness_series = pd.Series(fitnessProgessII)
            moving_avg = fitness_series.rolling(window=25).mean()
            ax3.plot(range(len(fitnessProgessII)), moving_avg, color='orange', linewidth=1, label="MA (25)")
            moving_avgII = fitness_series.rolling(window=50).mean()
            ax3.plot(range(len(fitnessProgessII)), moving_avgII, color='red', linewidth=1, label="MA (50)")
            moving_avgIII = fitness_series.rolling(window=100).mean()
            ax3.plot(range(len(fitnessProgessII)), moving_avgIII, color='green', linewidth=2, label="MA (100)")
            fitnesses = [ind.objectives[1] for ind in pop.individuals]
            for marker in markerPoolExtension:
                ax3.vlines(marker, ymin=min(fitnessProgessII), ymax=max(fitnessProgessII), alpha=0.5)
            ax3.set_xlabel("Generation")
            ax3.set_ylabel("Objective 2")
            ax3.set_title("Objective 2 Progress")
            ax3.ticklabel_format(style='plain', axis='x', useOffset=False)
            ax3.legend()

            ax4.scatter(range(len(fitnessProgessIII)), fitnessProgessIII, color='black', label="Objective 3", s=10)
            fitness_seriesIII = pd.Series(fitnessProgessIII)
            moving_avg_III_25 = fitness_seriesIII.rolling(window=25).mean()
            ax4.plot(range(len(fitnessProgessIII)), moving_avg_III_25, color='orange', linewidth=1, label="MA (25)")
            moving_avg_III_50 = fitness_seriesIII.rolling(window=50).mean()
            ax4.plot(range(len(fitnessProgessIII)), moving_avg_III_50, color='red', linewidth=1, label="MA (50)")
            moving_avg_III_100 = fitness_seriesIII.rolling(window=100).mean()
            ax4.plot(range(len(fitnessProgessIII)), moving_avg_III_100, color='green', linewidth=2, label="MA (100)")
            for marker in markerPoolExtension:
                ax4.vlines(marker, ymin=min(fitnessProgessIII), ymax=max(fitnessProgessIII), alpha=0.5)
            ax4.set_xlabel("Generation")
            ax4.set_ylabel("Objective 3")
            ax4.set_title("Objective 3 Progress")
            ax4.ticklabel_format(style='plain', axis='x', useOffset=False)
            ax4.legend()

        else:
            ax.clear()
            # ── Single-Objective: 1x1 Layout (Fitness + Lineage-Fitness zusammengefuehrt) ──
            ax.scatter(range(len(fitnessProgess)), fitnessProgess, color='black', label="Fitness", s=10)
            fitness_series = pd.Series(fitnessProgess)
            moving_avg = fitness_series.rolling(window=25).mean()
            ax.plot(range(len(fitnessProgess)), moving_avg, color='orange', linewidth=2, label="MA (25)")
            moving_avgII = fitness_series.rolling(window=50).mean()
            ax.plot(range(len(fitnessProgess)), moving_avgII, color='red', linewidth=2, label="MA (50)")
            # Honest curve: elite[0] re-scored on validationSeedCount seeds it was not
            # selected on. The gap to the black training points IS the winner's curse.
            ax.plot(range(len(validationProgess)), validationProgess,
                    color='blue', linewidth=1, alpha=0.6,
                    label=f"Validation ({validationSeedCount} unseen seeds)")
            validation_series = pd.Series(validationProgess)
            ax.plot(range(len(validationProgess)),
                    validation_series.rolling(window=25).mean(),
                    color='blue', linewidth=2.5, label="Validation MA (25)")
            # The validation number carries NO measurement noise: the environment is
            # deterministic given a seed, the seed set is fixed, and elite individuals are
            # exempt from mutation -- re-scoring the same individual returns the identical
            # value. Every jump in the thin blue line therefore means a DIFFERENT
            # individual is being validated, not a different measurement of the same one.
            ax.plot(range(len(validationEliteMeanProgess)), validationEliteMeanProgess,
                    color='teal', linewidth=1, alpha=0.7, label="Validation, elite mean")
            # Best honest score reached so far. Exact on this seed panel; still optimistic
            # about the full seed universe, because it is a maximum over many individuals
            # scored on the same 50 seeds.
            ax.plot(range(len(validationBestSoFarProgess)), validationBestSoFarProgess,
                    color='darkgreen', linewidth=2, linestyle='--',
                    label="Validation, best so far")
            for marker in markerPoolExtension:
                ax.vlines(marker, ymin=min(fitnessProgess), ymax=max(fitnessProgess), alpha=0.5)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            # Verhindert Matplotlibs automatische wissenschaftliche Notation/Offset auf der
            # x-Achse (z.B. "1e3" + Offset-Label), die bei hohen Generationszahlen sonst
            # eine verwirrende Skalierung erzeugt.
            ax.ticklabel_format(style='plain', axis='x', useOffset=False)

            # Alte Twin-Achsen aus dem vorherigen Plot-Update entfernen, bevor neue
            # erzeugt werden -- sonst haeufen sich bei jedem Update (alle 50 Gen.)
            # zusaetzliche, unsichtbar uebereinanderliegende Achsen an.
            if axRightSuccess is not None:
                axRightSuccess.remove()
            if axRightN is not None:
                axRightN.remove()
            axRightSuccess = None
            axRightN = None
            if useLineageFitness:
                # Lineage-Mean auf derselben (linken) Achse wie die rohe Fitness, da
                # beide dieselbe Reward-Skala teilen -- macht Ueberschwingen/Truegen
                # der Batch-Fitness gegenueber dem robusteren Lineage-Mittel sichtbar.
                ax.plot(range(len(lineageMeanProgress)), lineageMeanProgress,
                        color='blue', linewidth=1.5, label="Lineage-Mean")

                # Lineage-Erfolgsrate (rechte Achse, 0-1): Anteil gelandeter Seeds ueber
                # die gesamte Lineage-Historie -- robusterer Generalisierungs-Indikator
                # als die Batch-Fitness.
                axRightSuccess = ax.twinx()
                axRightSuccess.plot(range(len(lineageSuccessRateProgress)), lineageSuccessRateProgress,
                                     color='green', linewidth=1.5, label="Lineage-Erfolgsrate")
                axRightSuccess.set_ylabel("Erfolgsrate")
                axRightSuccess.set_ylim(0, 1)

                # lineageN (weitere, nach aussen versetzte rechte Achse): zeigt direkt im
                # Chart, ob die aktuelle Elite ueber mehrere Generationen hinweg Elite
                # geblieben ist (n waechst dann ueber den steady state von
                # priorCap(lineagePriorCapMutation) + maxSeeds hinaus, da
                # capLineageAfterMutation() Elite-Individuen bewusst NICHT deckelt), oder
                # ob nahezu jede Generation eine neue (gerade erst gedeckelte) Linie zur
                # Elite aufsteigt (n bleibt dann konstant bei ca. steady state).
                axRightN = ax.twinx()
                axRightN.spines["right"].set_position(("outward", 60))
                axRightN.plot(range(len(lineageNProgress)), lineageNProgress,
                              color='purple', linewidth=1.2, alpha=0.7, label="Lineage-N")
                axRightN.set_ylabel("Lineage-N")

            steadyStateN = lineagePriorCapMutation + maxSeeds
            turnoverHint = ""
            if useLineageFitness:
                turnoverHint = (
                    " | hohe Elite-Fluktuation" if best.lineageN <= steadyStateN
                    else " | Elite bleibt bestehen"
                )
            ax.set_title(
                f"Fitness Progress (Generations: {g+1}) | Batches: {nBatches+1} | "
                f"Best Fitness: {round(maxFitness, 2)} | NN: {len(best.innerNodes)} | "
                f"Used: {nUsedNodes(best)} | Time.: {round(time.perf_counter()-start, 2)} | "
                f"Seeds: {len(seeds)} (panel, {seedPanelRotation} rotated/gen) | "
                f"Validation: {round(validationProgess[-1], 2)}\n"
                + (f"Lineage-Mean: {round(best.lineageMean, 2)} | n: {best.lineageN} "
                   f"(steady-state={steadyStateN}{turnoverHint}) | "
                   f"Erfolgsrate: {round(best.lineageSuccessRate, 2)}" if useLineageFitness else "")
            )

            linesLeft, labelsLeft = ax.get_legend_handles_labels()
            linesRight, labelsRight = [], []
            if axRightSuccess is not None:
                l, lab = axRightSuccess.get_legend_handles_labels()
                linesRight += l
                labelsRight += lab
            if axRightN is not None:
                l, lab = axRightN.get_legend_handles_labels()
                linesRight += l
                labelsRight += lab
            ax.legend(linesLeft + linesRight, labelsLeft + labelsRight, loc="upper left")

            # ── Netzgroessen ueber die Generationen ──
            # Linke Achse: Knoten je Netz -- Median mit 10.-90.-Perzentilband (wohin
            # bewegt sich die Population), Maximum (einzelne aufgeblaehte Netze) und die
            # Groesse des aktuell besten Individuums (waechst die Loesung oder nur der
            # Ballast?). Rechte Achse: Anteil nie betretener Knoten ueber die gesamte
            # Population -- steigt er zusammen mit der Groesse, ist das Wachstum dormante
            # Masse (seedSpecialist-Transplantate, Junk-DNA) und kein echter Zugewinn an
            # Struktur; addDelNodes sollte diesen Anteil je Generation wieder abbauen.
            axSize.clear()
            axSizeRight.clear()
            gensSize = range(len(networkSizeMedianProgress))
            axSize.fill_between(gensSize, networkSizeP10Progress, networkSizeP90Progress,
                                color='steelblue', alpha=0.2, label="10.-90. Perzentil")
            axSize.plot(gensSize, networkSizeMedianProgress, color='steelblue', linewidth=1.5, label="Median")
            axSize.plot(gensSize, networkSizeMaxProgress, color='indianred', linewidth=1.0,
                        linestyle='--', label="Maximum")
            axSize.plot(gensSize, networkSizeBestProgress, color='black', linewidth=1.2, label="bestes Individuum")
            for marker in markerPoolExtension:
                axSize.vlines(marker, ymin=min(networkSizeP10Progress), ymax=max(networkSizeMaxProgress), alpha=0.5)
            axSize.set_xlabel("Generation")
            axSize.set_ylabel("Knoten je Netz")
            axSize.ticklabel_format(style='plain', axis='x', useOffset=False)
            axSize.set_title(
                f"Netzgroessen (Median {round(networkSizeMedianProgress[-1], 1)}, "
                f"Max {int(networkSizeMaxProgress[-1])}, "
                f"dormant {round(unusedShareProgress[-1], 1)}%)"
            )

            axSizeRight.plot(gensSize, unusedShareProgress, color='darkorange', linewidth=1.2,
                             linestyle=':', label="dormante Knoten [%]")
            # clear() setzt bei einer twinx()-Achse die Position von Ticks und Achsenbeschriftung
            # auf links zurueck -- ohne diese beiden Zeilen liegt "dormante Knoten [%]" nach dem
            # ersten Update auf der Beschriftung der linken Achse.
            axSizeRight.yaxis.set_label_position("right")
            axSizeRight.yaxis.set_ticks_position("right")
            axSizeRight.set_ylabel("dormante Knoten [%]")
            axSizeRight.set_ylim(0, 100)
            # Rotation auf derselben rechten Achse: sie teilt sich mit dem Dormant-Anteil
            # nur den Wertebereich, nicht die Bedeutung -- beides sind kleine Zahlen, und
            # der Zusammenhang "schneller rotieren, wenn die Population das Panel ausnutzt"
            # liest sich neben der Netzgroesse am besten.
            if rotationAdaptive:
                axSizeRight.plot(range(len(rotationProgress)), rotationProgress,
                                 color='purple', linewidth=1.2, alpha=0.8,
                                 label="Rotation [Seeds/Gen]")

            linesSize, labelsSize = axSize.get_legend_handles_labels()
            linesDormant, labelsDormant = axSizeRight.get_legend_handles_labels()
            axSize.legend(linesSize + linesDormant, labelsSize + labelsDormant, loc="upper left")

            # ── Populations-Erfolg je Seed (Heatmap, aktuelle Generation) ──
            # Zeigt fuer jedes Individuum (Zeile, sortiert nach Fitness) und jeden Seed
            # (Spalte) den erzielten Reward -- macht sichtbar, welche Seeds fuer die
            # gesamte Population schwer sind (durchgehend rote Spalte) vs. welche nur
            # fuer einzelne Individuen ein Problem darstellen (vereinzelte rote Zellen in
            # einer sonst gruenen Spalte).
            if heatColorbar is not None:
                heatColorbar.remove()
                heatColorbar = None
            # ── Crossover-Wirkung ──────────────────────────────────────────
            axCross.clear()
            if axCrossRight is not None:
                axCrossRight.remove()
            # Zwei Achsen, weil die Groessenordnungen um den Faktor 20-50 auseinanderliegen:
            # "zugeordnet" ist das Angebot der Rollenpaarung (Dutzende), "ausgetauscht" und
            # "uebersprungen" sind das Ergebnis der Ziehung (einstellig). Auf einer gemeinsamen
            # Achse kleben die beiden interessanten Kurven auf der Null. Zusaetzlich geglaettet,
            # die Streuung von Generation zu Generation ist sonst nicht lesbar.
            crossWindow = 25
            axCross.plot(range(len(crossoverExchangedProgress)),
                         pd.Series(crossoverExchangedProgress).rolling(crossWindow, min_periods=1).mean(),
                         color='green', linewidth=2, label=f"ausgetauscht (MA {crossWindow})")
            axCross.plot(range(len(crossoverSkippedProgress)),
                         pd.Series(crossoverSkippedProgress).rolling(crossWindow, min_periods=1).mean(),
                         color='red', linewidth=2, label=f"uebersprungen (MA {crossWindow})")
            axCross.set_xlabel("Generation")
            axCross.set_ylabel("Knoten je Generation")

            axCrossRight = axCross.twinx()
            axCrossRight.plot(range(len(crossoverMatchedProgress)),
                              pd.Series(crossoverMatchedProgress).rolling(crossWindow, min_periods=1).mean(),
                              color='grey', linewidth=1.2, alpha=0.7,
                              label=f"zugeordnet / Angebot (MA {crossWindow})")
            axCrossRight.set_ylabel("zugeordnete Knoten")
            recentExchanged = sum(crossoverExchangedProgress[-25:])
            recentSkipped = sum(crossoverSkippedProgress[-25:])
            recentDrawn = recentExchanged + recentSkipped
            axCross.set_title(
                f"Crossover-Wirkung ({crossoverType}, Paar {crossoverProbability} x "
                f"Knoten {crossoverNodeRate}) | letzte 25 Gen.: "
                f"{recentExchanged} ausgetauscht, {recentSkipped} uebersprungen"
                + (f" ({100.0 * recentExchanged / recentDrawn:.0f}% Erfolgsquote)" if recentDrawn > 0 else "")
            )
            crossLines, crossLabels = axCross.get_legend_handles_labels()
            rightLines, rightLabels = axCrossRight.get_legend_handles_labels()
            axCross.legend(crossLines + rightLines, crossLabels + rightLabels,
                           fontsize=7, loc="upper left")
            axCross.ticklabel_format(style='plain', axis='x', useOffset=False)

            axHeat.clear()
            fitnessMatrix = np.array([
                ind.fitnessValues for ind in pop.individuals
                if len(ind.fitnessValues) == len(seeds)
            ])
            if fitnessMatrix.size > 0:
                order = np.argsort(-fitnessMatrix.mean(axis=1))  # bestes Individuum oben
                # FESTE Farbskala statt der per-Generation automatisch skalierten: sonst wird in
                # jeder Generation das jeweils beste Individuum gruen und das schlechteste rot
                # eingefaerbt, egal auf welchem absoluten Niveau -- eine Population, die
                # durchgehend abstuerzt, sieht dann genauso aus wie eine, die durchgehend landet,
                # und die Farben sind ueber Generationen hinweg nicht vergleichbar. Mit festen
                # Grenzen bedeutet eine Farbe immer denselben Reward.
                im = axHeat.imshow(fitnessMatrix[order], aspect='auto', cmap='RdYlGn',
                                   interpolation='nearest', norm=heatNorm)
                axHeat.set_xlabel("Seed-Index (aktuelle Generation)")
                axHeat.set_ylabel("Individuum\n(sortiert nach Fitness)")
                axHeat.set_title(
                    f"Populations-Erfolg je Seed (aktuelle Generation, Skala {heatVmin} bis {heatVmax})"
                )
                heatColorbar = fig.colorbar(im, ax=axHeat, fraction=0.025, pad=0.02,
                                            extend='both',  # Werte ausserhalb der Skala als Pfeil
                                            ticks=[heatVmin, -250, 0, 150, heatVmax])

        plt.pause(0.1)  # Wichtig: Erlaubt GUI-Update ohne die Schleife zu stoppen

    print(f"Generation: {g}")
    print("Per-seed rewards:", np.array(best.fitnessValues).round(0))
    print("Fitness:", round(maxFitness,2))
    championHeld = (len(validationProgess) > 1
                    and validationProgess[-1] == validationProgess[-2])
    print(f"Validation ({validationSeedCount} unseen seeds): {round(validationProgess[-1], 2)} "
          f"| elite mean: {round(validationEliteMeanProgess[-1], 2)} "
          f"| best so far: {round(validationBestSoFarProgess[-1], 2)} "
          f"| optimism of the training score: {round(maxFitness - validationProgess[-1], 2)}")
    if landingRateProgess:
        recent = statistics.mean(landingRateProgess[-selectionSwitchWindow:])
        print(f"  landing rate of elite[0]: {landingRateProgess[-1]*100:.0f}% "
              f"(mean over last {selectionSwitchWindow}: {recent*100:.1f}%) | "
              f"selection: {lexicaseType}"
              + ("" if selectionStage + 1 >= len(selectionSchedule)
                 else f" -> switches at {selectionSwitchLandingRate*100:.0f}%"))
    if hasBestEver:
        print(f"  best ever: {round(bestEverScore, 2)} on the {validationSeedCount} reporting "
              f"seeds, archived at generation {bestEverGeneration} "
              f"({g - bestEverGeneration} generations ago)")
    if hallOfFame and hallOfFameScore is not None:
        heldRun = 0
        for held in reversed(hallOfFameHeldProgess):
            if held != 1:
                break
            heldRun += 1
        print(f"  hall of fame: champion scored {round(hallOfFameScore, 2)} on "
              f"{eliteSeedCount} fresh seeds | "
              f"{'defended' if hallOfFameHeldProgess[-1] else 'DETHRONED'} "
              f"(held {heldRun} generations in a row, "
              f"{round(100 * statistics.mean(hallOfFameHeldProgess[-25:]))}% over the last 25)")
    if eliteReevaluation and eliteHonestProgess:
        print(f"  elite re-evaluated on {eliteSeedCount} fresh seeds: "
              f"{round(eliteHonestProgess[-1], 2)} | the honest winner was ranked "
              f"#{int(eliteRankShiftProgess[-1]) + 1} of {eliteCandidates} by the training batch "
              f"(mean over last 25: "
              f"{round(statistics.mean(eliteRankShiftProgess[-25:]) + 1, 1)})")
    if rotationAdaptive:
        print(f"  rotation: {seedPanelRotation} of {maxSeeds} seeds per generation "
              f"(panel fully exchanged every "
              f"{maxSeeds / seedPanelRotation:.0f} generations)")
    print(f"  champion {'held' if championHeld else 'REPLACED'} this generation "
          f"| panel overfit probe (established minus fresh seeds): "
          f"{round(panelFreshGapProgess[-1], 2)} "
          f"(mean over last 25: {round(statistics.mean(panelFreshGapProgess[-25:]), 2)})")
    print("Min:", round(min(best.fitnessValues), 2))
    print("Max:", round(max(best.fitnessValues), 2))
    print("Poolsize:", len(pool))
    print("Curriculum Level:", round(curriculumLevel, 2))
    if useLineageFitness:
        print("Lineage-Fitness (best):", round(best.lineageMean, 2), "| n:", best.lineageN,
              "| Lineage-Erfolgsrate:", round(best.lineageSuccessRate, 2))
        steadyStateN = lineagePriorCapMutation + maxSeeds
        turnoverHint = "hohe Elite-Fluktuation (neue Elite-Linie)" if best.lineageN <= steadyStateN else "Elite bleibt seit mehreren Generationen bestehen"
        print(f"  -> steady-state n = {steadyStateN} (priorCap {lineagePriorCapMutation} + {maxSeeds} Seeds/Gen): {turnoverHint}")
        print(f"  -> LCB (z={lineageZ}): {round(best.lineageLCB(lineageZ), 2)} "
              f"(Std.Fehler: {round(math.sqrt(best.lineageVariance() / max(best.lineageN, 1)), 2)}) "
              f"-- tatsaechliches Selektionskriterium, sobald n >= minLineageN({minLineageN})")
    # Haerteste Seeds DIESER Generation (kein stabiles Gedaechtnis mehr noetig, siehe
    # uniformDirectionCurriculum-Kommentar oben): einfach die niedrigsten Fitnesswerte
    # im aktuellen Batch, gepaart mit dem jeweils zugewiesenen Richtungswinkel.
    seedFitnessThisGen = sorted(
        zip(seeds, best.fitnessValues, directionAngles if directionAngles else [None]*len(seeds)),
        key=lambda t: t[1]
    )[:5]
    print("Haerteste Seeds dieser Generation (Seed, Fitness, Winkel[deg]):",
          [(s, round(f, 2), round(math.degrees(a), 1) if a is not None else None) for s, f, a in seedFitnessThisGen])
    print("_"*20)


    if g > 50:
        # Steering (curriculum stages, stage transitions) reads the VALIDATION curve:
        # the training fitness moves by as much as its own noise between generations,
        # so thresholds on it fire on luck rather than on progress.
        fitness_series = pd.Series(validationProgess)
        moving_avg = fitness_series.rolling(window=50).mean()
        recoveryPause -= 1

        

        if stage == 0:

            if moving_avg.iloc[-1] >= maxSteps and curriculumLevel < 1.0 and recoveryPause <= 0:

                curriculumLevel += 0.1
                curriculumLevel = min(curriculumLevel, 1.0)
                recoveryPause = 100 

                if curriculumLevel >= 1.0:
                    stage = 1
                    survivalMode = False
                    landingQuote = True 

        elif stage == 1:

            if moving_avg.iloc[-1] >= 1 and recoveryPause <= 0:
                survivalMode = False
                landingQuote = False
                stage = 2


    if (g % 250 == 0) and g > 0:
        renderVideos(pop, g, seeds, curriculumLevel=curriculumLevel, validation=False,
                     uniformDirectionCurriculum=uniformDirectionCurriculum, directionAngles={})
        plotNetwork(pop.individuals[pop.indicesElite[0]], filename = f"generation_{g}_lunarlander_solution_used_py_fitness:{pop.individuals[pop.indicesElite[0]]}.html", justUsedNodes = True, justUsedEdges = True)
        plotNetwork(pop.individuals[pop.indicesElite[0]], filename = f"generation_{g}_lunarlander_solution_all_py_fitness:{pop.individuals[pop.indicesElite[0]]}.html", justUsedNodes = False, justUsedEdges = False)

# Everything below runs on single individuals, so the worker pool is no longer needed.
if evaluator is not None:
    evaluator.close()
    evaluator = None

# ## Inspecting the Best Individual

individual = pop.individuals[pop.indicesElite[0]]
plotNetwork(individual, filename = f"lunarlander_solution_py_fitness:{individual.fitness}.html", justUsedNodes = False)

individual = pop.individuals[pop.indicesElite[0]]
plotNetwork(individual, filename = f"lunarlander_solution_used_py_fitness:{individual.fitness}.html", justUsedNodes = True, justUsedEdges = True)

## Validation of the Best Individual

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=0.5)
# Keep only the best individual
popValidation = fn.Population(
    seed=seed,
    ni=1,       # number of individuals
    jn=8,         # judgment nodes
    jnf=8,        # judgment node functions
    pn=4,         # processing nodes
    pnf=5,        # processing node functions (0-3: actions, 4: "last decision" -> enables judgment-node hysteresis)
    fractalJudgment=False,
    nFeatureValues=[5,5,5,5,5,5,2,2]
)
# Both the individual the run ENDED with and the best one it ever produced are validated
# on the same 1000 seeds. The gap between them is what gets lost when a good champion is
# not held on to -- elite slots are recomputed every generation, so the best net of
# generation 700 is usually gone by generation 2000.
seedsValidation = [random.randint(0, 10_000) for _ in range(1000)]

def validateOnSeeds(individual, label):
    popValidation.individuals = [individual]
    results = []
    for v in seedsValidation:
        popValidation.gymnasium(
            env,
            dMax=10,
            maxSteps=1500,
            maxConsecutiveP=5,
            worstFitness=0,
            seed=v,
            validation=True
        )
        results.append(popValidation.individuals[0].fitness)
    landingRate = sum(1 for r in results if r > 100) / len(results)
    print(f"[{label}] mean over {len(results)} seeds: {statistics.mean(results):.2f} "
          f"| landing rate {landingRate*100:.1f}%")
    return results

validationResults = validateOnSeeds(pop.individuals[pop.indicesElite[0]], "final individual")
finalMean = statistics.mean(validationResults)

if hasBestEver:
    bestEverResults = validateOnSeeds(popBestEver.individuals[0], "best ever archived")
    bestEverMean = statistics.mean(bestEverResults)
    print(f"\nbest ever was archived at generation {bestEverGeneration}, scoring "
          f"{round(bestEverScore, 2)} on the {validationSeedCount} reporting seeds")
    print(f"  on the independent 1000 seeds it reaches {bestEverMean:.2f} "
          f"(the reporting score is optimistic: it is a maximum over ~{len(validationProgess)} "
          f"noisy values)")
    print(f"  difference to the final individual: {bestEverMean - finalMean:+.2f} "
          f"-- this is what holding on to the best net is worth")
    # Everything downstream should look at whichever net is actually better.
    if bestEverMean > finalMean:
        bestIndividual = popBestEver.individuals[0]
        print("  -> the archived individual is the better one, using it below")
    else:
        bestIndividual = pop.individuals[pop.indicesElite[0]]
        print("  -> the final individual is the better one, using it below")
else:
    bestIndividual = pop.individuals[pop.indicesElite[0]]

print(f"Average Fitness of Validations: {finalMean}")

# ── Persist the results ──────────────────────────────────────────────────────
# After a run of many hours the numbers must not live only in the terminal: the macOS
# matplotlib backend floods stdout with harmless event errors ('cmd' is not a valid
# signal) and pushes the output out of the scrollback. Everything worth keeping goes to
# disk, and the networks are pickled so a run can be re-evaluated without retraining.
runDir = os.path.join("runs", time.strftime("run_%Y%m%d_%H%M%S"))
os.makedirs(runDir, exist_ok=True)

def saveIndividual(individual, name):
    with open(os.path.join(runDir, f"{name}.pkl"), "wb") as f:
        pickle.dump(individual, f)

saveIndividual(pop.individuals[pop.indicesElite[0]], "final")
if hasBestEver:
    saveIndividual(popBestEver.individuals[0], "bestEver")

summary = {
    "generations": len(fitnessProgess),
    "validationSeedCount": validationSeedCount,
    "finalValidationMean": finalMean,
    "finalLandingRate": sum(1 for r in validationResults if r > 100) / len(validationResults),
    "bestEver": ({
        "archivedAtGeneration": bestEverGeneration,
        "scoreOnReportingSeeds": bestEverScore,
        "validationMean": bestEverMean,
        "landingRate": sum(1 for r in bestEverResults if r > 100) / len(bestEverResults),
    } if hasBestEver else None),
    "settings": {
        "seedSchedule": seedSchedule, "maxSeeds": maxSeeds,
        "seedPanelRotation": seedPanelRotation,
        "stratifySeedsByDirection": stratifySeedsByDirection,
        "lexicaseType": lexicaseType, "lexicaseEpsilon": lexicaseEpsilon,
        "eliteSize": eliteSize, "eliteCandidates": eliteCandidates,
        "eliteSeedCount": eliteSeedCount, "hallOfFame": hallOfFame,
        "crossoverType": crossoverType, "crossoverProbability": crossoverProbability,
        "crossoverNodeRate": crossoverNodeRate, "boundaryTolerance": boundaryTolerance,
        "matchOnlyUsed": matchOnlyUsed,
        "edgeMutationK": edgeMutationK, "boundaryMutationK": boundaryMutationK,
        "maxSteps": maxSteps, "populationSize": pop.ni,
        # Network shape and input encoding. These were missing and are exactly what
        # distinguished two runs that otherwise looked identical in this file -- the
        # difference (fixed 5 edges per judgment node against a random count) had to be
        # reconstructed from the pickled networks afterwards.
        "jn": pop.jn, "jnf": pop.jnf, "pn": pop.pn, "pnf": pop.pnf,
        "nFeatureValues": nFeatureValues,
        "minFeatures": minFeatures, "maxFeatures": maxFeatures,
        "fractalJudgment": pop.fractalJudgment, "useExperience": pop.useExperience,
        "junk": junkShare,
        "selectionSchedule": selectionSchedule,
        "seedStagePlateauDelta": seedStagePlateauDelta,
        "rotationAdaptive": rotationAdaptive,
        "generationsConfigured": generations,
        "populationSeed": seed,
        "nodeGracePeriod": nodeGracePeriod,
        "useMultiprocessing": useMultiprocessing,
        "nWorkers": nWorkers if useMultiprocessing else 1,
    },
    "networkSize": {
        "final": len(pop.individuals[pop.indicesElite[0]].innerNodes),
        "medianLast": networkSizeMedianProgress[-1] if networkSizeMedianProgress else None,
        "dormantShareLast": unusedShareProgress[-1] if unusedShareProgress else None,
    },
}
with open(os.path.join(runDir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Per-generation curves, so a run can be re-plotted or compared later.
curveNames = ["generation", "trainingFitness", "validation", "validationBestSoFar",
              "networkSizeMedian", "dormantShare", "crossoverExchanged", "crossoverSkipped"]
curves = [range(len(fitnessProgess)), fitnessProgess, validationProgess,
          validationBestSoFarProgess, networkSizeMedianProgress, unusedShareProgress,
          crossoverExchangedProgress, crossoverSkippedProgress]
nRows = min(len(c) for c in curves)
with open(os.path.join(runDir, "curves.csv"), "w") as f:
    f.write(",".join(curveNames) + "\n")
    for i in range(nRows):
        f.write(",".join(str(c[i]) for c in curves) + "\n")

# Raw per-seed rewards of both networks on the 1000 validation seeds.
with open(os.path.join(runDir, "validation_per_seed.csv"), "w") as f:
    f.write("seed,final" + (",bestEver" if hasBestEver else "") + "\n")
    for i, sv in enumerate(seedsValidation):
        row = f"{sv},{validationResults[i]}"
        if hasBestEver:
            row += f",{bestEverResults[i]}"
        f.write(row + "\n")

fig.savefig(os.path.join(runDir, "progress.png"), dpi=150, bbox_inches="tight")
print(f"\nrun saved to {runDir}/ "
      f"(final.pkl, bestEver.pkl, summary.json, curves.csv, validation_per_seed.csv, progress.png)")

# ## Seed-Schwierigkeits-Diagnose: korreliert der initiale Impuls mit Abstuerzen?
diagnosticSeeds = [random.randint(0, 10_000) for _ in range(500)]
seedDifficultyDf = analyzeSeedDifficulty(bestIndividual, diagnosticSeeds, outputDir=runDir)

# ## Rendering and Recording the Best Run
renderVideos(pop=pop, seeds=seeds)
plt.ioff()
plt.show(block=True)
# Manche Backends (z.B. macOS) betrachten das Figure-Fenster nach dem interaktiven
# ion()/pause()-Loop bereits als "gezeigt" und plt.show() kehrt sofort zurueck, wodurch
# das Skript endet und das Fenster mitsamt schliesst. Haelt den Prozess daher explizit
# offen, bis das Fenster manuell geschlossen wird.
while plt.fignum_exists(fig.number):
    plt.pause(0.5)
