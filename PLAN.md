# Plan: AlphaZero on atomic chess

Status as of 2026-07-25. Every claim below is annotated with the source that
establishes it — `file:line` for code, "measured" for numbers derived from this
run's logs, "MultiAra" for [Gehrke 2021](https://ml-research.github.io/papers/gehrke2021assessing.pdf),
the only published study covering atomic.

Paths are relative to the repo root; `$SP` = `/u/$USER/scratch/open_spiel`,
`$SCRATCH` = `/u/$USER/scratch`.

---

## 0. Where we are

Job `9658911` is training from zero on one H200 with chess-scale
hyperparameters. It works end to end — self-play, GPU inference, evaluator
threads, checkpointing — but has produced no learn step yet, and four config
values are actively wasting the run.

**Eval ladder after 25 games at step 0** (measured, `log-evaluator-{0,1}.txt`):

| MCTS sims | AZ record | mean return |
|---|---|---|
| 300 | 2W / 2L | 0.00 |
| 948 – 300000 | 0W / 21L | −1.00 |

Total **2W/23L = 8%**, mean return −0.84, no draws. Both wins came as White.
That is the expected floor for a randomly-initialised net, not a signal.

**Game lengths: mean 15.2 plies, median 11, range 5–42** (measured, n=25). This
single number is what invalidates most of the current config — the defaults
assume chess-length games of 80–150 plies.

### Revised target

MultiAra reached **parity with Fairy-Stockfish 13.1 classical eval** at atomic,
and lost to **Fairy-Stockfish with the atomic NNUE by ~300 Elo** (the atomic
NNUE alone gave Fairy-SF ~500 Elo). So:

- **Milestone: parity with Fairy-Stockfish, `Use NNUE=false`.** This equals the
  published state of the art and is what "beat MultiAra" amounts to — the two
  are the same rung, and Fairy-SF is already built while MultiAra needs
  MXNet 1.8 / TensorRT 8 and model downloads.
- Beating NNUE-enabled Fairy-SF at atomic would be **new**, and is a stretch
  goal, not a plan.

---

## Phase 1 — Stop wasting the current run (do first, ~1 hour)

### 1.1 `temperature_drop=20` exceeds the average game length

`PlayGame` samples moves ∝ visit counts until `history.size() >= temperature_drop`,
then plays greedily (`alpha_zero.cc:135-139`). With `temperature_drop=20` and a
15.2-ply mean game, **the greedy phase never happens** — every self-play game is
sampled at full temperature start to finish.

Why it matters: atomic games are decided by single moves. Sampling ∝ visits
regularly plays a move the search knows loses to an explosion mate, so the value
head learns that sound positions are lost. The value targets are corrupted by
self-inflicted blunders.

**→ `--temperature_drop=6`** (~⅓ of a typical game). Highest-value single change.

### 1.2 `learn_rate` is 262,144 states per step

`learn_rate = replay_buffer_size / replay_buffer_reuse` = 1048576/4
(`alpha_zero.cc:324`, consumed at `:355`). At 15.2 states/game that is
**~17,200 self-play games per learn step** — roughly 5× what the config intends,
because atomic games are ~5× shorter than chess games.

**→ `--replay_buffer_size=262144`** (~4,300 games/step, ~6× more gradient steps
per GPU-hour). Side benefit: the buffer is serialised **every** learn step
(`alpha_zero.cc:403`) at ~4.7KB/state ≈ 4.7GB/step, dropping to ~1.2GB.

⚠️ **Requires a fresh run directory.** `LoadBuffer` fatally errors if the saved
buffer's `max_size` differs (`serializable_circular_buffer.h:53-57`), so
changing this mid-run guarantees a crash on the next requeue.

### 1.3 The net is oversized for this game

ResNet 256×20 ≈ **24.3M params** (~97MB fp32, ~290MB per checkpoint pair with
optimizer state). Every actor move costs 300 forward passes, so net size
throttles the self-play data rate — the scarce resource. MultiAra used
RISEv2-mobile with **13 blocks**, smaller and cheaper.

**→ `--nn_width=128 --nn_depth=10`** (~6M params, ~4× the games/hour). Scale up
only after the small net plateaus.

### 1.4 The eval ladder eats ~85% of evaluator CPU

Opponent sims are `max_simulations × 10^(level/2)` → 300, 948, 3000, 9486,
30000, 94868, 300000 (`alpha_zero.cc:276`). `EvalResults::Next()` round-robins
over `eval_levels × 2` slots, so every level gets an **equal game count** and
wall-clock is dominated by the slowest. Measured: levels 4–6 consumed ~85% of 37
minutes for 9 of 19 games, and evaluator-1 spent **29+ minutes on one
300,000-sim game**. The opponent is `RandomRolloutEvaluator` — pure CPU, so both
evaluator threads peg a core producing no training data.

**→ `--eval_levels=4 --evaluators=1 --evaluation_window=30`**

### 1.5 Consolidated fresh-start config

Apply to [train_atomic_az.sh](train_atomic_az.sh):

```
--nn_width=128 --nn_depth=10       # was 256 / 20
--temperature_drop=6               # was 20
--replay_buffer_size=262144        # was 1048576
--eval_levels=4                    # was 7
--evaluators=1                     # was 2
--evaluation_window=30             # was 100
--checkpoint_freq=25               # steps are ~6x faster now
```

Leave `max_simulations=300`, `uct_c=2.0`, `replay_buffer_reuse=4`,
`policy_alpha/epsilon`, `learning_rate`, `weight_decay` alone — those are sound.

### 1.6 Request more CPU

`--cpus-per-task=16` with `--gres=gpu:H200:1`
([atomic_az.slurm:10-11](atomic_az.slurm#L10-L11)). OpenSpiel runs MCTS tree
search on **CPU** and uses the GPU only for batched inference, so self-play is
core-limited while the H200 has headroom. MultiAra independently hit this —
their GPUs used under 950MB of 32GB at batch size 8, and "run multiple
self-play processes on one GPU" was their top recommendation.

**→ Raise `--cpus-per-task` to 32–64** if the partition allows, and scale
`--actors` with it. Likely the largest throughput win available.

### 1.7 Fix the preemption crash

[train_atomic_az.sh:20](train_atomic_az.sh#L20) resumes whenever `config.json`
exists. But `config.json` is written at **startup** (`alpha_zero.cc:552-555`)
while `learner.jsonl` is not written until the first learn step **completes** —
currently 90+ minutes. Preempted at minute 40, the requeued job finds
`config.json` with an empty/absent `learner.jsonl` and dies in
`json::FromString("").value()` (`alpha_zero.cc:82-83`).

**→ Guard the resume condition:**

```bash
if [[ -f "${CONFIG}" && -s "${RUN_DIR}/learner.jsonl" ]]; then
  # resume
else
  # fresh start
fi
```

### Acceptance criteria

- [ ] `log-learner.txt` shows `Step: 1` within ~20 minutes of launch
- [ ] `jq '.game_length.avg' learner.jsonl` ≈ 15–25
- [ ] `eval.results[0]` climbing above 0 within a few hours
- [ ] A killed-and-requeued job resumes without crashing

---

## Phase 2 — Make measurement trustworthy (~2 hours, parallel with training)

Nothing below changes training; it changes whether we can tell progress from
noise. Currently we cannot.

### 2.1 Build `az_vs_sf` (it has never been compiled)

```bash
cp eval/az_vs_sf.cc $SP/open_spiel/examples/
```
Add to `$SP/open_spiel/examples/CMakeLists.txt`:
```cmake
if (${OPEN_SPIEL_BUILD_WITH_LIBTORCH})
  add_executable(az_vs_sf az_vs_sf.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(az_vs_sf ${TORCH_LIBRARIES})
endif()
```
```bash
cd $SP/build && make -j az_vs_sf
```

### 2.2 Three harness fixes

1. **Per-colour scores.** Most important. `harnesses/README.md` records that
   after `1.Nf3`, 17 of 20 Black replies **lose by force**. If atomic is
   heavily White-favoured, an aggregate 50% could be 100% as White and 0% as
   Black — indistinguishable from "equal strength". Print White score, Black
   score, and n separately, always.
2. **Unfinished games are scored as draws.**
   [az_vs_sf.cc:107](eval/az_vs_sf.cc#L107) maps a non-terminal state (600-ply
   cap, or an unparseable SF reply) to `r = 0.0` → 0.5 points, silently
   inflating the score. Count and report them separately.
3. **`ucinewgame` between games.** One `UCIBot` is reused for the whole match
   with no reset ([az_vs_sf.cc:85](eval/az_vs_sf.cc#L85)), so SF carries hash
   and history across games. Send `ucinewgame` + `isready` per game.

### 2.3 Deterministic, finer-grained opponent strength

- **Nodes, not milliseconds.** `go movetime` results depend on machine load.
  Switch to `go nodes N` for reproducibility across runs and hardware.
- **`UCI_Elo` instead of Skill Level.** Fairy-SF supports
  `UCI_LimitStrength=true` + `UCI_Elo=<n>` for a smooth ~1350–2850 dial rather
  than 21 coarse, blunder-injecting steps:
  ```cpp
  uci::Options opts = {{"UCI_Variant", "atomic"},
                       {"UCI_LimitStrength", "true"},
                       {"UCI_Elo", std::to_string(target_elo)}};
  ```
- **Fix the defaults.** [az_vs_sf.cc:39-40](eval/az_vs_sf.cc#L39-L40) default to
  skill 20 / 200ms — full strength with *more* thinking time than the baseline
  that already scored ~0%. Default to the bottom rung instead.

### 2.4 Determine the real target strength

```bash
./stockfish
setoption name UCI_Variant value atomic
uci        # look for EvalFile / NNUE
```
If an atomic NNUE is loaded, `setoption name Use NNUE value false` is the
milestone opponent. Record which build and which NNUE file, since MultiAra's
numbers are against **Fairy-SF 13.1** and current versions differ.

### 2.5 Enough games, paired openings

The README's "~17% (1W/6)" has a 95% CI of roughly 0–50% — it is noise. At n=40
the standard error on a proportion is ±8pp; distinguishing 40% from 50% needs
several hundred games.

- Play **paired openings** (same opening, colours swapped) and score the pair.
- Always report `score% ± 1.96·√(p(1−p)/n)`.
- Never act on a single 40-game run.

### Acceptance criteria

- [ ] `az_vs_sf` builds and completes a 20-game match
- [ ] Output shows per-colour scores, unfinished-game count, and a CI
- [ ] Two identical invocations produce statistically consistent results

---

## Phase 3 — Supervised pretraining (built, needs cluster build)

Code is complete in [pretrain/](pretrain/). See
[pretrain/README.md](pretrain/README.md) for full detail.

### Evidence

| MultiAra finding | Number |
|---|---|
| From-zero vs supervised-init (King of the Hill ablation) | from-zero **~220 Elo weaker**, never caught up |
| Cost of from-zero | **26 days on 4 Tesla V100s** |
| Cost of supervised model | "several hours on a single GPU" |
| RL gain over supervised init, **atomic** | **+150 Elo** fast TC / +115 long |
| RL gain, other variants | +300 to +690 Elo |
| Atomic supervised data | 36M samples ≈ 530k games |
| Epoch count | **7 epochs generalised better than 30** |

We are currently on the from-zero path, on one GPU. The published evidence says
that is ~220 Elo worse *and* an order of magnitude more expensive.

### Steps

1. **Verify the reader** (already passing locally, re-run on the cluster):
   recipe in [pretrain/README.md](pretrain/README.md) §1. Expect
   `ALL TESTS PASSED`.
2. **Build the trainer:**
   ```bash
   cp pretrain/az_pretrain.cc pretrain/pgn_atomic.h $SP/open_spiel/examples/
   # add the az_pretrain target to examples/CMakeLists.txt (README §2)
   cd $SP/build && make -j az_pretrain
   ```
3. **Get data.** Atomic files are per-month at
   `https://database.lichess.org/atomic/`. Hold out a **separate month** for
   validation — never a random split, since positions within a game are highly
   correlated and would leak. MultiAra used April 2018 as test, August 2018 as
   validation, and Elo ≥ 1900 for atomic (top 10th percentile).
4. **Run:**
   ```bash
   export TRAIN_PGN=$SCRATCH/data/lichess_db_atomic_rated_2020-06.pgn
   export VAL_PGN=$SCRATCH/data/lichess_db_atomic_rated_2018-08.pgn
   export RUN_DIR=$SCRATCH/atomic_az/run_pretrained
   bash pretrain/bootstrap_pretrained_run.sh
   ```
5. **Pick the best epoch** by `policy_top1` on the held-out month, not the
   default. A checkpoint is written per epoch; copy the winner over
   `checkpoint--1.pt` (README §5).

### Why the bootstrap script has four phases

A **fresh** RL start destroys pretrained weights: with `resuming == false` it
calls `SaveCheckpoint(0)` from random init then loads it back
(`alpha_zero.cc:585-595`), and rewrites `vpnet.pb` whenever `config.graph_def`
is empty. The **resume** path does neither. So injection must go through resume:
generate `vpnet.pb`+`config.json` from the real flags → pretrain against that
exact `vpnet.pb` → seed `learner.jsonl` with a step-0 record → launch with the
positional `config.json`.

### Acceptance criteria

- [ ] `ALL TESTS PASSED` on the cluster build
- [ ] `games_used` is a large fraction of `games_seen` (high `skipped_parse`
      means the reader is wrong — investigate, don't proceed)
- [ ] `policy_top1` reaches ≥ 0.35 and `value_sign_acc` ≥ 0.60 on held-out data
- [ ] The RL job prints **`Loading model from step -1`** and
      **`Using existing model`** — `step 0` or `Overwriting existing model`
      means the pretrained weights were discarded
- [ ] `eval.results[0]` starts well above the from-zero run's −1.0

---

## Phase 4 — Climb the ladder

Advance only when the current rung is cleared with error bars that exclude the
threshold.

| Rung | Opponent | Target | Gate |
|---|---|---|---|
| 0 | rollout MCTS, 300 sims | >80% | `eval.results[0]` > +0.6 sustained |
| 1 | rollout MCTS, 3000 sims | >60% | `eval.results[2]` > +0.2 |
| 2 | Fairy-SF `UCI_Elo=1400`, 100k nodes | >50% | 200+ paired games, CI excludes 50% |
| 3 | Fairy-SF `UCI_Elo=1800` | >50% | same |
| 4 | Fairy-SF `UCI_Elo=2200` | >40% | same |
| 5 | **Fairy-SF full, `Use NNUE=false`** | parity | **the milestone — equals MultiAra** |
| 6 | Fairy-SF with atomic NNUE | any wins | stretch; beyond published results |

Do not run a Stockfish match before rung 0 is cleared. A net that loses to
300-sim rollout MCTS will score 0% and teach us nothing.

### Levers if we plateau

1. **More sims at eval than training.** Train at 300, evaluate at 800–1600
   (`--az_sims`). Often worth 100+ Elo for free, no retraining.
2. **More CPU / more actors** (Phase 1.6) — likely still the binding constraint.
3. **Scale the net** once throughput can feed it: a fresh 256×20 run will pass
   the small net eventually.
4. **Opening diversity.** If self-play collapses onto a few forcing lines the
   net gets narrow and SF punishes it off-book. Check `game_length_hist` and
   action variety in `log-actor-*.txt`; add random opening plies if collapsed.
5. **`--replay_buffer_reuse=8`** for more gradient steps per game; watch the
   value-loss / value-accuracy gap for overfitting.

### Known risk

MultiAra's atomic RL **stalled after 26 model updates** and no later model could
beat update 25, even with a reduced learning rate. That was not a compute limit,
so it may recur here. Mitigations to try in order: more training samples per
update (their own diagnosis), larger Dirichlet `policy_alpha`, opening
diversity. Budget for the possibility that atomic self-play plateaus early.

---

## Operational rules

**Metrics live in files, not stdout.** `FileLogger` writes only to disk
(`logger.h:43-58`), so the SLURM `.out` shows startup lines and nothing else —
that is not a hang. [README.md:128](README.md#L128) is wrong on this.

```bash
R=$SCRATCH/atomic_az/run_main
tail -30 $R/log-learner.txt
jq -c '{step, t:.time_rel, states:.total_states, sps:.states_per_s,
        glen:.game_length.avg, outcomes:.outcomes.counts,
        eval:.eval.results, evalN:.eval.count,
        loss:.loss.sum, pol:.loss.policy, val:.loss.value}' \
   $R/learner.jsonl | tail -20
```

`outcomes.counts` is `[Player1, Player2, Draw]`; **player 0 is Black**
(`chess.h:73-78`), so index 0 is Black's win count.

**Empty eval windows read as 0.0, not "no data."** `AvgResults()` returns 0 for
an empty buffer (`alpha_zero.cc:250-252`), indistinguishable from a genuine 0.0
mean (= 50%). Always cross-check `eval.count`.

**Never change on a resume:**
- `nn_width` / `nn_depth` — `vpnet.pb` gets rewritten with a shape the
  checkpoints do not match
- `replay_buffer_size` — `LoadBuffer` fatally errors on a max-size mismatch

**Disk.** ~290MB per checkpoint pair at 256×20 (~75MB at 128×10), kept forever
every `checkpoint_freq` steps, plus `replay_buffer.data` rewritten every step
(~4.7GB at buffer 2^20, ~1.2GB at 2^18). Watch the scratch quota.

**Delete or update the stale clone.** `~/atomic-test` is a second clone of the
same GitHub repo, 6 commits behind, still containing the `/gpu:0` bug in 6
places including its `train_atomic_az.sh` and `smoke_test.slurm`. Two clones
with one remote is how a job gets launched from the broken copy.

---

## Reference: the numbers that shaped this plan

Reading order for the MultiAra thesis: §4.3 (supervised setup and Elo
thresholds), §4.4.5 (the atomic MCGS NaN failure — they fell back to plain MCTS,
the only variant needing it), §5.1 (per-variant RL Elo gains), §5.2 (strength vs
Fairy-Stockfish), §5.3 (game balance and White/Black win rates — relevant to
Phase 2.2's per-colour reporting), §5.5 and §6.4 (the zero-knowledge ablation),
§6.1 (their own throughput recommendations).

Sources: [Gehrke 2021](https://ml-research.github.io/papers/gehrke2021assessing.pdf)
· [Czech et al. 2020, CrazyAra](https://arxiv.org/abs/1908.06660)
· [CrazyAra repo](https://github.com/QueensGambit/CrazyAra)
· [variant-nnue-pytorch](https://github.com/fairy-stockfish/variant-nnue-pytorch)
· [lichess open database](https://database.lichess.org/)
