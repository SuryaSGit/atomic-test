# Plan: a publishable result from atomic chess

Status as of 2026-07-28. Every claim is annotated with its source — `file:line`
for code, "measured" for numbers from this project's runs, "MultiAra" for
[Gehrke 2021](https://ml-research.github.io/papers/gehrke2021assessing.pdf), the
only published study covering atomic.

Paths are relative to the repo root; `$SP` = `/u/$USER/scratch/open_spiel`,
`$SCRATCH` = `/u/$USER/scratch`, `$DATA` = `~/atomic_sf_data` locally.

---

## 0. Where we are

The project reached a working end-to-end pipeline: an atomic-chess AlphaZero net,
a Fairy-Stockfish distillation dataset, a statistically sound match harness, and
a supervised trainer that shares the RL loop's loss, optimizer and checkpoint
format. What it has *not* reached is playing strength.

**Measured, 2026-07-27, 60 paired games vs Fairy-Stockfish:**

| | Score | Record |
|---|---|---|
| as White | 23.3% ± 15.4 | 7W 0D 23L |
| as Black | **0.0% ± 0.0** | 0W 0D 30L |
| overall | 11.7% ± 8.2 | 7W 0D 53L |

**Measured, sims sweep:** 13 / 8.8 / 10 / 11.2% at 200 / 400 / 800 / 1600
simulations. An 8× increase in search buys nothing, so the network — not the
search — is the binding constraint.

**Measured 2026-08-01, the ablation (two arms, equal budget ~66.3M positions,
4 epochs each):**

| Arm | Teacher holdout EDGE | T3 own-play EDGE | retained | policy_top1 (teacher → T3) |
|---|---|---|---|---|
| A4 — base only | 0.19654 | 0.16849 | **86%** | 0.4095 → 0.3499 |
| B — 50/50 base + diversified | 0.19408 | 0.16844 | **87%** | 0.4060 → 0.3567 |

Two results, both negative for the original hypothesis:

1. **Opening diversification changes nothing.** The arms differ by 0.000045 in
   EDGE. Untargeted coverage neither helped nor hurt, on either distribution.
2. **Distribution shift is modest, not catastrophic.** 86% of the edge survives
   on own-play positions.

⚠️ **A retracted claim.** Earlier readings of "0.618 against a 0.617 base rate"
(no information) and "39% retained" were **measurement artifacts**, not model
behaviour. Two errors compounded:

- The `az_vs_sf` hook scored only positions where **our bot moved** (~20k),
  while `Validate()` scores **all** positions (44,633 decisive).
- Its base rate was computed **per game** by hand — 703 of 1,153 decisive games
  won by White = 0.610 — where the correct **position-weighted** rate on the
  same games is **0.537**. Black-won games are systematically longer, so
  weighting by position dilutes White's share.

The wrong denominator inflated the apparent shift roughly twofold. Trust the
`Validate()` numbers: same code path as the teacher measurement, larger n, base
rate computed on exactly the positions scored.

**Both arms are at the label ceiling on value** (0.7285 / 0.7261 against a
measured ceiling of 0.725) — no headroom there. **The policy head is not:**
`policy_top1` of 0.41 means it disagrees with Stockfish on ~3 of 5 moves.

**Measured 2026-08-02, the strength ladder** — arm A4 at 1300 sims vs
Fairy-Stockfish 13.1, classical evaluation, Hash=256MB, paired openings:

| Fairy-SF nodes | book openings (n=200) | random 4-ply (n=400) |
|---|---|---|
| 10,000 | **80.8% ± 5.3** | 74.4% ± 4.1 |
| 50,000 | **27.5% ± 5.8** | 32.9% ± 4.4 |
| 200,000 | **6.8% ± 3.0** | 18.1% ± 3.6 |
| 650,000 | **0.8% ± 0.8** | — |

**Crossover ~25,000 nodes** under the book, ~26,000 under random openings. Two
opening regimes agreeing makes that the citable strength anchor. This is the
primary result and nothing below disturbs it.

**The colour gap was mostly the opening protocol.** 27.7 / 30.2 / 27.3pp under
random openings; 5.5 / 9.0 / −2.4pp under the book. Fairy-SF against *itself*
shows 15–20pp from random openings alone. Earlier framing of this as a defect
of our network was wrong.

**Measured, sims sweep** (arm A4, vs Fairy-SF at 50k, n=80 each):
24.4% at 400 sims, **35.0% at 800** — roughly **+88 Elo per doubling**. The
original flat sweep (13/8.8/10/11.2%) was the *pre-retrain* net; search does
convert on arm A4, so "the network is the bottleneck, not the search" no longer
holds and the ladder at 1300 sims understates the model.

### MultiAra: the released RL model does not work

Five independent measurements, after matching engine version, evaluation,
opening book, search budget, and move history:

1. It scores **8.75%** (n=40) against **its own supervised predecessor**, where
   Gehrke reports the RL model as **+150 Elo better**.
2. Raw network, no search — on a won rook endgame (K+R vs bare k) it returns
   **+9cp**, and on the mirrored lost one **−3cp**. The SL model returns
   **+897** and **−803**.
3. It evaluates the initial position at **+957cp**, where SL gives +102 and
   Fairy-SF 13.1 gives +254 at depth 20 (true White edge: 55.7%).
4. It does not improve from 1300 to 6400 nodes — the signature of a value head
   carrying no rankable signal.
5. It scores 40.2 / 16.7 / 9.0% on the ladder above, far under its published
   parity claim.

The value head is not saturated; it is *anti-correlated with reality* —
confident on balanced positions, blind on decided ones. That pattern fits an
input-layout mismatch: a near-empty endgame fills almost no planes, so a model
reading shifted planes sees zeros. Both models receive the same 63-plane tensor
through the same binary, and the SL model works perfectly through all of it.

**Consequence: the ~82% head-to-head against the RL model is void**, not
caveated. It measured a non-functioning network.

**MultiAra-SL is the real opponent.** It is the supervised-initialised model
from the same release (lichess games, Aug 2013 – Jul 2020), demonstrably
working, and per §5.2 about 40 Elo below the published RL figure against
Fairy-Stockfish. It is NOT "MultiAra" — that name means the RL model in the
literature — so results must say "MultiAra's supervised model".

**Datasets in hand** (measured, `$DATA`):

| Set | Games | Positions | White/Black/Draw |
|---|---|---|---|
| `atomic.*` (`random_plies=8`) | 574,858 | 18,967,523 | 52.7 / 44.8 / 2.6% |
| `hard/atomic.*` (`random_plies=20`) | 561,731 | 14,760,838 | 51.3 / 47.0 / 1.7% |
| total | 1,136,589 | **33,728,361** | 1.59 GB |

Both generated at `label_nodes=10000`, `multipv=4`, NNUE on.

### What changed since the last plan

Phases 1 and 2 of the previous plan are **done** (config retune at
[train_atomic_az.sh:82-105](train_atomic_az.sh#L82-L105), resume guard at
[:26](train_atomic_az.sh#L26), harness rebuild in [eval/match.h](eval/match.h)).
Phase 3 **pivoted** from lichess PGN supervised learning to Fairy-Stockfish
distillation (5ba3ab0). Phase 4's ladder gates were bypassed — we ran Stockfish
matches directly.

---

## 1. The target, and why it is not "beat Fairy-Stockfish"

**Verified against the thesis 2026-08-01** (Gehrke 2021 §5.2, quoted):

> "In 3check, **Atomic** and King of the Hill both engines **were about even**"
> — vs **Fairy-Stockfish 13.1, classical evaluation**
>
> "Fairy-Stockfish now won in 3check, **Atomic** and King of the hill by about
> 65, **300** and 320 Elo" — with NNUE

So the parity and −300 Elo figures in [pretrain/README.md:20-21](pretrain/README.md#L20-L21)
are correct. Two qualifiers the summary tables omit, both of which soften the
milestone:

**1. It is a blitz result.** Figure 5.1: "each player had **10/60 seconds for
the game** plus 0.1/0.6 seconds per move." And atomic is named as a variant
where the advantage *shrinks* with thinking time — "MultiAra won significantly
more games against Fairy-Stockfish in the **fast** time control than in the long
time control, when playing Horde, Crazyhouse and **Atomic**." Parity at 10-second
games is not parity at tournament time controls.

**2. RL bought ~40 Elo against a real opponent, not 150.** The "+150 Elo fast
TC / +115 long" in our table is the **self-relative** gain (Figure 5.1c, Elo
between MultiAra's own model updates, supervised init pinned at zero). Measured
against Fairy-Stockfish, §5.2 reports the gain as **"40 Elo for Atomic"** —
against 330 for 3check, 370 for Antichess, 490 for King of the Hill. Atomic was
the worst variant but one. A ~4x gap between self-measured and externally
measured Elo is the classic self-play ladder inflation signature.

**Consequence:** the only published evidence says RL in atomic is worth ~40 Elo
against an external opponent, after 26 days on 4 V100s, and it stalled — "the
models of Atomic and Racing kings improved at the beginning, but stopped after
the 26th and 4th model update". That is a substantive reason to deprioritise RL,
independent of our compute constraints.

Our RL compute is worse than that by more than an order of magnitude. Measured
over job 9664549: **3h 23m of GPU time across 59h 27m of wall clock** on the
preemptible `scavenger` partition — a 5.7% duty cycle, four preemptions, two of
which ran for 20 seconds and 11 minutes. Chasing a strength result through
self-play RL on that budget is not a plan.

**But the interesting question here was never the RL.** It is this: distillation
from a strong engine produces a network that is excellent on the engine's
positions and uninformative on its own, and that is measurable, fixable, and
unpublished. Supervised training costs *hours*, not weeks (MultiAra: "several
hours on a single GPU"), so the paper fits the compute we actually have.

### Contributions, in descending order of confidence

**C1 — Outcome calibration of Fairy-Stockfish's atomic evaluation.**
33.7M labelled positions with known game outcomes. Chess convention maps
centipawns to win probability with `tanh(cp/400)`; **for atomic the fitted
constant is ~1580** (measured, refit on both datasets), i.e. atomic evaluations
are roughly 4× less decisive per centipawn than chess. Also measured:
mate-scored positions predict the winner with mean result ±0.92 (n=271k); a
~0.10 residual White advantage at *equal* evaluation; and an apparent
non-monotonicity above 1600cp. Nobody has published any of this, and anyone
training an atomic NNUE or AZ needs the constant. **Status: essentially done.**

**C2 — How to measure distribution shift, and why naive accuracy misleads.**
Reframed 2026-08-01 after the ablation came back null. The contribution is no
longer "shift is the bottleneck" — it isn't — but the methodology and the
negative result:

- Distillation transfers **better than expected**: 86% of the value head's edge
  survives on positions the net reaches in its own play.
- Untargeted opening diversification (14.8M extra positions, deliberately wilder
  openings) produced **no measurable change** at equal training budget.
- The **same underlying data** yielded verdicts of "no information at all",
  "moderately degraded", and "comparable to training" depending purely on the
  baseline used. Accuracy without a base rate is uninterpretable, and a
  game-weighted base rate applied to a position-level metric is wrong by ~2x.

Negative results with a clean methodology are publishable, and this one is
cheap to defend: everything is measured through one code path with the baseline
computed on the scored positions. **Status: complete.**

**C3 — Artifact release.** 33.7M-position atomic dataset, generator, readers,
and a match harness with per-colour scoring, paired openings and CIs. **Status:
exists, needs packaging.**

**Stretch — strength.** Place the net on the Fairy-SF `UCI_Elo` scale rather
than claiming a win. "Our net plays at approximately Elo X at atomic" is a
reportable result; "we lost to Fairy-Stockfish" is not.

Venues that fit: IEEE Conference on Games (CoG), Advances in Computer Games
(ACG), AIIDE, or an ML workshop track. All accept focused empirical variant-game
work; none require beating the state of the art.

---

## Phase 1 — Lock down C1 (~1 day, CPU only)

The calibration result is nearly finished. It needs verification, not compute.

### 1.1 Fix the documented recipe — it returns noise

[pretrain/README.md:85-89](pretrain/README.md#L85-L89) buckets raw `f[2]`
(**side-to-move** centipawns, per the format header at
[sf_label.cc:33-34](pretrain/sf_label.cc#L33-L34)) against `result_p0`
(**player-0** relative) without correcting for whose turn it is. Verified
2026-07-28: the curve is flat at ≈ −0.02 across every bucket from −29,600 to
+2,400 — no signal at all. The cited fits (1620, then 1580) cannot have come
from this recipe.

Token *i* (1-based) has White to move when *i* is odd, and White is player 1
([chess.h:73-78](https://github.com/deepmind/open_spiel)), so the correction is
`cp_p0 = (i % 2 == 1) ? -cp : cp`. With it, the curve is monotonic:

| bucket cp | mean result | n | tanh(cp/1580) |
|---|---|---|---|
| 400 | 0.206 | 193,239 | 0.248 |
| 800 | 0.443 | 126,161 | 0.467 |
| 1200 | 0.604 | 72,279 | 0.641 |
| 1600 | 0.712 | 35,518 | 0.767 |
| 2000 | 0.640 | 17,951 | 0.853 |
| 2800 | 0.535 | 7,161 | 0.944 |

Commit the corrected recipe. It is the method of record for a constant that
drives 70% of every value target (`sf_lambda=0.7`).

### 1.2 The turn-over is a game-phase effect (measured 2026-07-28)

Conditioning the same buckets on ply index resolves it. Mean result by cp
bucket, `hard/atomic.0.tsv`:

| cp bucket | early (ply ≤20) | mid (21–40) | late (>40) |
|---|---|---|---|
| 800 | 0.282 | 0.415 | 0.606 |
| 1200 | 0.447 | 0.573 | 0.723 |
| 1600 | 0.551 | 0.675 | 0.798 |
| 2000 | **0.172** | 0.710 | 0.840 |
| 2800 | **0.157** | 0.568 | 0.813 |

Late in the game the relationship is monotonic to the top of the range; early it
collapses above 1600cp. So the aggregate turn-over was mixing phases:
**Fairy-Stockfish's atomic evaluation is unreliable in early, unstable positions
and reliable once the game resolves.** That is a stronger and more useful claim
than the original one.

Two confounders still to rule out before publishing:

1. **Horizon, not phase.** More plies remaining means more chances to squander
   an advantage. Re-bin by *plies remaining* rather than ply index; if the
   effect tracks plies-remaining rather than absolute depth, describe it that
   way.
2. **Random-opening contamination.** "Early" tokens follow a `uniform(0,20)`
   random opening, so early labelled positions are also the ones closest to
   random play. Re-run the split on the `random_plies=8` dataset, where the
   opening is shorter, and check the effect persists.

**Consequence for our own training:** `sf_lambda=0.8` weights the engine score
heavily, so early-position value targets are systematically overconfident. This
is a candidate contributor to arm A's failure that is *independent* of
distribution shift, and it suggests a phase-dependent lambda or down-weighting
early positions. Worth an arm of its own if Phase 2 has budget.

### 1.3 Anchor the first-move advantage

**Measured 2026-07-28: White's edge at the true initial position is +11.4pp**
(53.9/42.8 on `base` n=64,628; 54.2/42.2 on `book` n=18,599; combined n=83,227).
Direct measurement, not an extrapolation — the datasets record their random
opening as bare tokens, so games with **zero** random plies can be selected out
of any shard.

⚠️ **A retracted claim.** Aggregate per-dataset balances (52.7/44.8 at
`random_plies=8`, 51.3/47.0 at `random_plies=20`) appear to show the first-move
advantage decaying with opening depth. It does not. Those aggregates average
over opening-depth **parity**, and the per-depth breakdown is alternating:

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| edge | +11.2 | +15.5 | +13.6 | −4.8 | +17.0 | −9.8 | +19.8 | −14.2 | +22.6 |

At depth *d*, White has played ⌈d/2⌉ random moves to Black's ⌊d/2⌋, so odd
depths cost White a move *and* give Black the turn. Magnitudes grow with depth
in both directions: the more chaotic the opening, the more decisive it is to be
on move. Deeper `random_plies` simply mixes more odd depths in, cancelling the
even ones — that cancellation was the apparent "decay".

**Never compare colour balance across datasets with different `random_plies`.**
Condition on depth first.

**Unresolved:** depth 1 breaks the parity pattern (+15.5, favouring White, where
every other odd depth favours Black), and it replicates on both datasets so it
is not noise. The parity result does not go in the writeup until this is
explained. Candidate checks: whether depth-1 games differ in length or draw
rate, and whether `explore_prob` deviations are distributed evenly by colour.

⚠️ `--random_plies=0 --explore_prob=0` is **degenerate**: SF-vs-SF from the
initial position is deterministic, so all games would be identical. The depth-0
sample above comes from the `random_plies=2` run, where a third of games draw
zero random plies:

```bash
./sf_label --out=book/atomic --shard=$i --num_shards=8 --games=7000 \
           --label_nodes=10000 --multipv=4 --random_plies=2 --explore_prob=0.05
```

Measured: 8 shards, 44m 33s, 55,985 games, 2.08M positions, zero parse failures.

### 1.4 Report `cp_scale` per dataset

1580 was validated on `hard/` (measured 2026-07-28) and holds for |cp| ≤ 1600,
which covers the bulk of the data. Report the fit, its range of validity, and
the residual White offset. **Do not** use `--sf_max_abs_cp` to drop mate
positions: at ±0.92 they are the cleanest signal in the set (17.2% of
positions).

### Acceptance criteria

- [ ] Corrected recipe committed; rerunning it reproduces 1580 ± 50 on both sets
- [ ] Turn-over above 1600cp either survives the mate-conditioning check or is
      dropped from the writeup
- [ ] First-move advantage reported at 3+ opening depths including 0
- [ ] Mate-band reliability and the equal-eval White offset both reported with n

---

## Phase 2 — The C2 experiment (~1 week, this is the paper)

A three-arm ablation, all supervised, all on one GPU. The design matters more
than the compute.

### 2.1 Arms — status after the 2026-08-01 results

| Arm | Training data | Result |
|---|---|---|
| **A4** | base only, 66.3M positions | EDGE 0.1965 teacher / 0.1685 T3 |
| **B** | 50/50 base + diversified, 66.4M | **identical to A4** (Δ EDGE 0.000045) |
| **C** | + own-play DAgger | **deprioritised** — only 14% of edge is lost to shift, so there is almost nothing to recover. Data is generated and relabelling is ~7 min, so run it for completeness, but do not expect movement. |
| **D** | phase-conditioned value targets | **promoted** — see below |
| **E** | larger network | **promoted to top priority** — see §2.6 |

Arm D's justification strengthened rather than weakened: `value_mse` is 0.126 on
T3 against 0.069 on the teacher holdout — nearly 2x worse — while *sign*
accuracy holds up. The net knows who is winning off-distribution but its
magnitudes are miscalibrated, which is exactly what a single global
`cp_scale` of 1580 predicts when the true fit runs 2800-3000 early and
1100-1200 late (§1.2). It needs no new data.

### 2.6 Arm E — the network is sized for a constraint that no longer applies

`nn_width=128 nn_depth=10` (~6M params) was chosen in Phase 1 of the *previous*
plan to raise self-play throughput, because in RL the actor data rate is the
scarce resource ([train_atomic_az.sh:49-51](train_atomic_az.sh#L49-L51)). **In
supervised distillation that constraint does not exist** — the data is already
on disk and the GPU is the only cost.

The evidence says capacity is now the binding constraint on the part that
matters:

- The **value head is at the label ceiling** (0.726 vs 0.725). More capacity
  cannot help there, and neither can more data.
- The **policy head is nowhere near a ceiling** at `policy_top1` 0.41, and
  policy quality is what MCTS actually consumes. A search that starts from a
  bad move ordering wastes its simulations — which is the likely explanation for
  the flat sims sweep.

**→ Train 256x20 (~24M params) on the same 33.7M positions.** Roughly 4x the
compute per position, so ~12-16h on an A100 for a comparable budget. Score it on
the same T3 and the same holdout, and re-run the sims sweep: if search finally
scales with a better policy, that is strength for free.

⚠️ `nn_width`/`nn_depth` can only be set in a run directory with no `vpnet.pb`.
Use a fresh `RUN_DIR`.

### 2.2 Two design rules that decide whether this is publishable

**Equalise samples seen, not epochs.** Arm A has 18.97M positions and arm B has
33.7M. Training each for the same number of *epochs* confounds data composition
with data volume, and the paper's claim is about composition. Fix the number of
gradient steps across all arms and report it.

**Freeze the own-play test set.** Generate own-play positions **once**, from arm
A's net, relabel them, and evaluate every arm on those identical positions. If
each arm is tested on its own self-generated positions, the arms are scored on
different distributions and the comparison means nothing. This is the easiest
mistake available here and it invalidates the whole experiment.

### 2.3 Test sets

All held out, none overlapping training:

| Set | Source | Answers |
|---|---|---|
| **T1** | shard 7 of `atomic.*` | did it forget the base distribution? |
| **T2** | shard 7 of `hard/atomic.*` | did it learn the diversified coverage? |
| **T3** | frozen relabelled own-play (§2.2) | **did any of it transfer to our own positions?** |

T3 is the headline metric. Report every figure against the **base rate of that
test set** — 0.618 read as a success until we computed the 0.617 base rate, and
that mistake must not reach a paper.

### 2.4 Prerequisites (small, blocking, do first)

1. **Validation-only path in `az_pretrain`.** `--epochs=0` never reaches a
   validation call ([az_pretrain.cc:142](pretrain/az_pretrain.cc#L142) runs on
   the `val_every` schedule). Needed for the "before" measurement, which is
   only obtainable while the current checkpoint is the current checkpoint.
2. **Base-rate baseline in the harness.** Majority-class accuracy plus mean, SD
   and fraction-positive of predicted values, in both `Validate()` and the
   `az_vs_sf` calibration block.
3. **Colour-gap threshold.** [match.h:261](eval/match.h#L261) fires above 25pp;
   the 23.3/0.0 run did not trip it. Lower to ~15pp, or fire whenever either
   colour is 0% at n ≥ 20.
4. **Commit the relabeller.** It is load-bearing for arm C and is not in the
   repo.
5. **Hold out shard 7** of both datasets before anything is shipped or trained.

### 2.5 Strength measurement

Per arm, several hundred paired games. Note that distinguishing arms 5pp apart
needs ~400–800 games ([match.h:44-51](eval/match.h#L44-L51) gives ±8pp at
n=60), so **lower `sf_nodes` for the comparison matches** — relative ranking
does not need a full-strength opponent, and the harness is reproducible under
`go nodes`.

Separately, run a `UCI_Elo` sweep with `sf_bridge` to place the best arm on the
Elo scale. That number, with a CI, is the strength claim.

### Acceptance criteria

- [ ] Three arms trained to an equal number of samples seen
- [ ] All arms scored on frozen T1/T2/T3, each against its base rate
- [ ] Per-colour playing strength per arm, n ≥ 400 paired games, with CIs
- [ ] Best arm placed on the `UCI_Elo` scale with a CI
- [ ] The Black collapse either fixed or characterised — 0/30 is a finding in
      its own right given atomic is near colour-balanced at engine strength
      (measured: 51.3/47.0)

---

## Phase 3 — Packaging (C3)

- Dataset card: generation parameters, 33.7M positions, the format spec at
  [sf_label.cc:23-37](pretrain/sf_label.cc#L23-L37), colour balance, calibration
  constant, and the known miscalibrated band.
- `sf_data_check` output over the released shards, so the artifact ships with
  its own validation.
- The harness as the reusable piece: per-colour scoring, paired openings,
  `ucinewgame`, unfinished-game exclusion, CIs. The README already argues why
  each matters; that argument is a section of the paper.

---

## Phase 4 — RL, only if the queue allows

Not on the critical path for publication. If it runs, it runs as a bonus arm.

1. **Stop `run_v2`.** It is the from-zero arm — the first launch printed
   `Creating model` / `Loading model from step 0` — getting 5.7% of the clock,
   competing with work that has a model. MultiAra scores from-zero ~220 Elo
   below supervised init at an order of magnitude more compute.
2. **Move off `scavenger`** to `IllinoisComputes-GPU` / `sridhar-ic` (72h, no
   preemption); the swap is documented at
   [atomic_az.slurm:51-54](atomic_az.slurm#L51-L54). Drop `--requeue`.
3. **Lower `checkpoint_freq`** from 25 ([train_atomic_az.sh:104](train_atomic_az.sh#L104)).
   Preemptions of 20s and 11m banked nothing.
4. Launch from the best Phase 2 checkpoint via `bootstrap_pretrained_run.sh`,
   and confirm `Loading model from step -1` + `Using existing model` on the
   **first** launch — `run_v2` did not show this.

MultiAra's atomic RL stalled after 26 updates. Budget for that recurring, and do
not let it block the paper.

---

## Phase 5 — Beat MultiAra's supervised model

The goal, stated plainly. Transitivity puts the gap at **~145 Elo**: SL beats
the RL model by +410 (91.25%, n=40), we beat it by +265 (~82%), so SL sits
roughly 145 Elo above us — about a **30% score**. Confirm against the direct
match at n=200 before trusting that.

Levers, ordered by Elo per unit of effort:

### 5.1 More simulations — free, and already measured

+88 Elo per doubling on arm A4 (24.4% → 35.0% from 400 → 800 vs Fairy-SF@50k).
Two more doublings would be ~+176 Elo, which alone exceeds the gap.

⚠️ **The catch: equal-simulation matching means SL scales too.** What matters is
*relative* scaling. Two facts suggest this favours us: our search converts at
+88/doubling, and the RL model from the same codebase was completely flat from
1300 to 6400. If SL also scales poorly, raising the budget is pure gain.

**→ Sweep the head-to-head over simulations before anything else.** 800 / 1600 /
3200 / 6400 a side, ~80 games each. It is the cheapest possible test and it
either wins outright or tells us the gap is structural.

### 5.2 `uct_c` — untested, free

Fixed at 2.0 throughout, never tuned for this network's value scale. Standard
AlphaZero practice sweeps it; tens of Elo are typical. Costs only match time.

### 5.3 More epochs — the policy head was still improving

`policy_top1` rose 0.397 → 0.406 between step 259k and the epoch-4 boundary, so
4 epochs was not convergence. MultiAra found 7 generalised better than 30.
Cheap relative to a new architecture.

### 5.4 Arm E — the larger network

256x20, ~24M params, ~16h. The value head is pinned at the label ceiling
(0.726 vs 0.725) but `policy_top1` sits at 0.41 with room, and policy is what
MCTS consumes. Biggest single lever, highest cost. Do 5.1-5.3 first: if
simulations and tuning close the gap, this is unnecessary.

### 5.5 Arm D — phase-conditioned value targets

No new data. Early-position evaluations explain ~5% of outcome variance at any
`cp_scale` (S1.2), yet `sf_lambda=0.8` weights them at 80%. A phase-dependent
scale or lambda fixes targets we already know are wrong.

### Acceptance criteria

- [ ] Direct match vs MultiAra-SL at n=200 establishes the actual gap
- [ ] Simulation sweep of that match, 800 through 6400, ~80 games each
- [ ] `uct_c` swept at the best simulation count
- [ ] Report per-colour with intervals, book openings, and state the simulation
      count — a score without it is uninterpretable (measured: the same match
      moved 63.5% to 87.1% across opening regimes alone)
- [ ] Any claim says "MultiAra's supervised model", never "MultiAra"

## What would falsify the plan

- **Arm C does not lift T3 above its base rate.** Then distribution shift is not
  the binding constraint, and the suspects become capacity (6M params at
  128×10) and policy-target quality (`multipv=4` may be too coarse). This is
  still a reportable negative result, but the framing changes.
- **The 1600cp turn-over is a mate-conditioning artifact** (§1.2). C1 narrows
  but survives.
- **The Black collapse is a harness or convention bug**, not a model property.
  Worth ruling out early: the value sign convention is player-0-relative in
  three places (`vpevaluator.cc:73-77`, the training targets, the dump format),
  and a sign error somewhere would look exactly like "cannot play Black."
  Cheapest check: score arm A's value head separately by colour on T3.

---

## Operational rules

**Metrics live in files, not stdout.** `FileLogger` writes only to disk
(`logger.h:43-58`), so a SLURM `.out` showing startup lines and nothing else is
not a hang.

```bash
R=$SCRATCH/atomic_az/run_v2
jq -c '{step, t:.time_rel, states:.total_states, sps:.states_per_s,
        glen:.game_length.avg, outcomes:.outcomes.counts,
        eval:.eval.results, evalN:.eval.count,
        loss:.loss.sum, pol:.loss.policy, val:.loss.value}' \
   $R/learner.jsonl | tail -20
```

`outcomes.counts` is `[Player1, Player2, Draw]`; **player 0 is Black**, so index
0 is Black's win count. A white win stores `result_p0 = -1`.

**Empty eval windows read as 0.0, not "no data."** `AvgResults()` returns 0 for
an empty buffer (`alpha_zero.cc:250-252`), indistinguishable from a genuine 0.0
mean. Always cross-check `eval.count`.

**Never change on a resume:** `nn_width` / `nn_depth` (`vpnet.pb` is rewritten
with a shape the checkpoints do not match) or `replay_buffer_size` (`LoadBuffer`
fatally errors on a max-size mismatch). Bump to a new run directory instead.

**`--sf_cp_scale` defaults to 400** at
[az_pretrain.cc:65](pretrain/az_pretrain.cc#L65) — the chess value. The slurm
wrapper passes 1580, but any hand-run invocation must pass it explicitly or the
value targets are silently wrong. Change the default.

**Preemption can land mid-write** of `replay_buffer.data` (~1.2GB, rewritten
every learn step). If a resume ever dies inside buffer deserialisation, delete
the buffer and let it refill.

---

## Reference

Reading order for the MultiAra thesis: §4.3 (supervised setup, Elo thresholds),
§4.4.5 (the atomic MCGS NaN failure), §5.1 (per-variant RL Elo gains), §5.2
(strength vs Fairy-Stockfish), §5.3 (game balance and White/Black win rates),
§5.5 and §6.4 (the zero-knowledge ablation), §6.1 (throughput recommendations).

Sources: [Gehrke 2021](https://ml-research.github.io/papers/gehrke2021assessing.pdf)
· [Czech et al. 2020, CrazyAra](https://arxiv.org/abs/1908.06660)
· [CrazyAra repo](https://github.com/QueensGambit/CrazyAra)
· [variant-nnue-pytorch](https://github.com/fairy-stockfish/variant-nnue-pytorch)
· [lichess open database](https://database.lichess.org/)
· Ross et al. 2011, [DAgger](https://arxiv.org/abs/1011.0686) — the reduction
  that arm C implements
