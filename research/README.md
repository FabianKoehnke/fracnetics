# research/ — measurement studies for the LunarLander GNP run

Small, self-contained studies that measure a property of the existing experiment.
They never modify `include/` or `examples/lunarlander.py`; they import `fracnetics`
and reproduce the training loop so that what is measured is the real thing.

## seed_noise_study.py — how much of the fitness is luck?

Ten seeds are drawn fresh every generation and the mean over them becomes the
fitness. This study quantifies what that costs.

The starting observation is that **an episode is fully determined by its seed**
(`enable_wind=False`), so replaying a seed produces the identical reward. There is
no measurement noise to average away — the entire fluctuation comes from *which*
seeds were drawn. That makes it a sampling problem, not a measurement problem, and
the study is built to separate the two ways sampling hurts.

### What it measures

1. **Test-retest reliability** — the population is scored on two disjoint seed
   batches; how much of the ranking survives. Reported as a score correlation
   (Pearson) and a rank correlation (Spearman), plus the practical version: how many
   of the four elite individuals would be the same, and how much true reward is lost
   by crowning the batch winner.
2. **Seed budget** — the reliability curve over batch size, and the extrapolation to
   a target reliability. Given twice, because the two numbers answer different
   questions: at score level (does the *value* reproduce) and at rank level (does the
   *order* reproduce, which is all selection cares about).
3. **Variance decomposition** — a two-way split of the reward matrix into
   * real skill differences (the signal),
   * seed difficulty, shared by the whole population — this shifts the reported
     fitness curve between generations but cancels out of every within-generation
     comparison, since all individuals are scored on the same batch,
   * individual × seed interaction — the term that actually corrupts selection.

   Plus, as a by-product, the winner's curse: how much the reported champion score
   exceeds what the same individual scores on seeds it has not seen.

### Method

Run the ordinary evolution loop (same operators, parameters and selection as
`examples/lunarlander.py`) and at each checkpoint generation score the **entire**
population on a large seed set (default 60). That reward matrix is written to CSV;
every statistic is then derived from it by resampling, so the expensive part happens
once per checkpoint and the analysis can be repeated for free.

### Usage

```bash
source venv/bin/activate

python research/seed_noise_study.py --quick                 # ~1 min smoke test
python research/seed_noise_study.py                         # 60 generations, 3 checkpoints
python research/seed_noise_study.py --checkpoints 0 25 50 100 --measure-seeds 60

# same protocol, but with potential-based reward shaping switched on
python research/seed_noise_study.py --checkpoints 0 25 50 100 --measure-seeds 60 --potential

# re-analyse stored matrices without running a single episode
python research/seed_noise_study.py --analyze-only research/results/run_main
```

Batch sizes in the reliability curve are capped at `3 * S <= measure-seeds`, so batch
A, batch B and an untouched held-out set all fit. Two disjoint batches drawn from a
finite pool are negatively dependent; at `S = M/2` they are exact complements, and a
seed that flatters an individual in A is by construction missing from B. That barely
touches the correlations but destroys the elite-overlap statistic, which lives exactly
where real skill differences are negligible.

Cost per checkpoint is `population × measure-seeds` episodes (default 6 000, about
six generations' worth); the evolution itself costs `generations × 100 × 10`.

### Output (in `research/results/<run>/`)

| File | Contents |
|---|---|
| `report.txt` | the readable summary, also printed to stdout |
| `summary.json` | every number, for further analysis |
| `matrix_gen<g>.csv` | raw reward matrix, rows = individuals, cols = seeds (header: seed ids) |
| `training_curve.csv` | best/mean fitness per generation of the embedded run |
| `fig1_reliability.png` | reliability over batch size, measured vs. model |
| `fig2_variance_components.png` | where the scatter comes from, per checkpoint |
| `fig3_selection_quality.png` | elite stability and champion regret over batch size |
| `fig4_curve_noise.png` | wobble of the reported curve vs. its observed movement |
| `fig5_reward_shape.png` | reward distribution and seed-difficulty spread |

### Reading the result

* **Pearson ≫ Spearman** means the scores reproduce but the order does not: many
  individuals sit so close together that noise decides who wins. More seeds help
  only slowly here; a different selection (or more diversity) helps more.
* **Large seed share in the decomposition** means the *curve* is unreliable while
  selection is fine. Fix the reporting (a fixed validation set, elite re-scored on
  unseen seeds), not the seed count.
* **Large interaction share** means selection itself is guessing. That is what more
  seeds, a slowly rotating seed panel, or stratified seed draws address.
* **A crowded top plateau** (`crowding at the top` far below the noise per individual)
  makes the elite statistics degenerate: when the best 10 individuals lie within a few
  reward of each other but a batch score scatters by tens of reward, *no* seed budget
  identifies "the best four", and the champion regret bottoms out at the noise of the
  ground-truth estimate itself rather than at zero. That is a diversity finding, not a
  seed-budget finding.
* `requiredSeeds` at rank level is usually the sobering number — if it runs into the
  hundreds, buying reliability with more episodes is hopeless and the variance has to
  be removed by design instead (seed panel, per-seed baseline, stratification).

### Result: potential-based shaping (`run_main` vs. `run_pbrs`)

Both runs use an identical configuration; `run_pbrs` adds `--potential`. Up to
generation 55 the two evolutions are the same run: every variance component agrees to
the last decimal and the best fitness differs by 0.36 reward on average -- the constant
`-phi(s_0)` offset. That is the expected outcome and not a coincidence: with
`phi(terminal) = 0` and `gamma = 1` the shaping telescopes to a per-seed constant, so no
comparison an EA makes can be affected by it (`tests/test_potential_shaping.py` pins
this down directly). From generation 56 the trajectories separate, because a
float-rounding tie flips on the crowded plateau -- an incidental illustration of how
weakly determined the selection is up there.

The practical conclusion: exact potential-based shaping cannot help an EA that selects
on the episode return. Dense per-step information only reaches selection if it is kept
as a *separate criterion* rather than summed away -- which is what
`lexicaseSelection(type="objectives")` and `Network::lexicaseObjectives` already do.

`noise_analysis.py` holds the statistics and has no dependency on `fracnetics`, so it
can be pointed at any reward matrix.
