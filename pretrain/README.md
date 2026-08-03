# Supervised pretraining (RL finetuning setup)

Trains the AlphaZero net on human lichess atomic games first, then hands the
weights to self-play RL. This is the MultiAra recipe: supervised initialisation
followed by reinforcement learning.

## Why

From [Gehrke 2021, *Assessing Popular Chess Variants Using Deep Reinforcement
Learning*](https://ml-research.github.io/papers/gehrke2021assessing.pdf) (TU
Darmstadt), the only published study that covers atomic:

| Finding | Number |
|---|---|
| From-zero vs supervised-init (King of the Hill ablation) | from-zero ended **~220 Elo weaker** and never caught up |
| Cost of the from-zero run | **26 days on 4 Tesla V100s** (80 updates) |
| Cost of the supervised model | "only takes several hours on a single GPU" |
| RL gain over supervised init, **atomic** | **+150 Elo** fast TC / +115 long TC |
| RL gain, other variants | +300 to +690 Elo |
| MultiAra vs Fairy-Stockfish **classical** eval, atomic | roughly **even** |
| MultiAra vs Fairy-Stockfish **NNUE**, atomic | **−300 Elo** |

Atomic was the weakest variant for self-play RL in that study — its RL run
stalled after 26 model updates and no later model could beat update 25. That
makes the supervised starting point *more* important here, not less.

## Two teachers

The trainer accepts either data source, or both at once into the same shuffle
buffer:

- **Human games** (`--pgn`) — lichess atomic at Elo ≥ 1900, the MultiAra recipe.
- **Fairy-Stockfish distillation** (`--sf_tsv`) — generated locally by
  `sf_label`. For atomic this is the **stronger teacher**: Fairy-Stockfish is
  far above 1900-Elo human play, the data is free to generate on CPU, and the
  same approach produced Fairy-SF's own atomic NNUE (~500 Elo).

Distillation caps out at the teacher, so it gets you *to* Fairy-Stockfish
parity — the Phase 4 milestone — and RL has to do any surpassing.

## Files

| file | needs LibTorch | what it does |
|---|---|---|
| `sample.h` | no | the `Sample` type every reader emits |
| `pgn_atomic.h` | no | streaming PGN reader → samples |
| `sf_data.h` | no | sf_label `.tsv` reader → samples |
| `pgn_atomic_test.cc` | no | tests for the PGN reader |
| `sf_data_test.cc` | no | tests for value/policy targets and sign conventions |
| `sf_label.cc` | no | **generates** the distillation dataset (CPU, laptop-friendly) |
| `sf_data_check.cc` | no | replays a dataset to validate it |
| `run_sf_label_mac.sh` | — | builds + shards `sf_label` on macOS |
| `az_pretrain.cc` | **yes** | trains `VPNetModel`, writes `checkpoint--1.pt` |
| `bootstrap_pretrained_run.sh` | — | wires the pretrained net into an RL run |

## Distillation flow

```bash
# 1. Generate on any CPU machine (measured: 530 pos/s on an M4, 49 bytes each)
bash pretrain/run_sf_label_mac.sh            # ~10h -> ~18M positions, ~900MB

# 2. Validate (drop the last line if the run is still going)
$OUT/sf_data_check $OUT/atomic.*.tsv         # expect DATASET OK

# 3. Ship to the cluster and train
rsync -av $OUT/atomic.*.tsv cc-login1:/u/$USER/scratch/data/
az_pretrain --path=$RUN_DIR --sf_tsv=$(ls -m /path/atomic.*.tsv | tr -d ' \n') \
            --val_sf_tsv=/path/holdout.tsv --sf_lambda=0.7 --sf_policy_temp=100
```

### Target construction

```
value  = sf_lambda * tanh(stm_cp / sf_cp_scale) + (1 - sf_lambda) * game_result
policy = softmax(multipv_cp / sf_policy_temp)      # or one-hot when temp = 0
```

- `--sf_lambda=0.7` follows the Fairy-SF NNUE convention. Pure search scores
  inherit the teacher's biases; pure outcomes are noisy.
- `--sf_cp_scale=1620` is calibrated for atomic by fitting `tanh(cp/K)` to
  observed outcomes over 579k positions. The chess default of 400 saturates:
  `tanh(1000/400)=0.99` where the real win rate at +1000cp is 0.56. Recalibrate
  if the generator config changes:
  ```bash
  grep -hv '^#' atomic.0.tsv | awk -F'\t' '{r=$1; n=split($2,p," ");
    for(i=1;i<=n;i++){split(p[i],f,"|"); if(f[2]!=""){b=int(f[2]/200)*200;
    s[b]+=r; c[b]++}}} END{for(b in s) printf "%6d %7.3f %d\n", b, s[b]/c[b], c[b]}' \
    | sort -n
  ```
  Pick the scale where `tanh(cp/scale)` best tracks the mean result per bucket.
- `--sf_policy_temp=100` gives soft targets across the MultiPV list, which
  carry more information per position than one-hot. Requires `--multipv > 1`
  at generation time.
- `--sf_max_abs_cp` optionally drops mate-scored positions (~16% of the data)
  if you would rather spend capacity on unresolved play.

## 1. Verify the PGN reader (any machine, no GPU)

The parsing is where the risk is, so it is testable against a plain CPU build:

```bash
SP=/path/to/open_spiel; AZ=/path/to/atomic_az
cd $SP/build/games
FLAGS="-O2 -w -std=gnu++20 -I$SP/open_spiel/abseil-cpp \
  -I$SP/open_spiel/json/include -I$SP -I$SP/open_spiel -I$AZ/pretrain"
clang++ $FLAGS -c $AZ/pretrain/pgn_atomic_test.cc -o /tmp/pgn_test.o
LINK=$(cat CMakeFiles/atomic_chess_test.dir/link.txt)
LINK=${LINK/CMakeFiles\/atomic_chess_test.dir\/atomic_chess\/atomic_chess_test.cc.o//tmp/pgn_test.o}
LINK=${LINK/-o atomic_chess_test/-o /tmp/pgn_test}
eval "$LINK" && /tmp/pgn_test
```

Expected tail: `ALL TESTS PASSED`. The suite covers the value-sign convention,
castling, `1.d4` no-space numbering, `+`/`#`/`?!` suffixes, `{clk}` comments,
variations, NAGs, the Elo/variant/result filters, and that a game containing an
illegal move contributes **zero** samples rather than a mislabelled prefix.

## 2. Build the trainer (cluster, LibTorch on)

```bash
# az_pretrain.cc includes all three headers, so copy them together
cp pretrain/{az_pretrain.cc,pgn_atomic.h,sf_data.h,sample.h} \
   $SCRATCH/open_spiel/open_spiel/examples/
```

Add to `open_spiel/examples/CMakeLists.txt`:

```cmake
if (${OPEN_SPIEL_BUILD_WITH_LIBTORCH})
  add_executable(az_pretrain az_pretrain.cc
                ${OPEN_SPIEL_OBJECTS}
                $<TARGET_OBJECTS:alpha_zero_torch>)
  target_link_libraries(az_pretrain ${TORCH_LIBRARIES})
endif()
```

**Apply the vpnet patch first.** `az_pretrain` calls `SetLossWeights()` and
`SetLearningRate()`, which upstream OpenSpiel does not have -- AlphaZero sums the
policy and value losses equally and fixes the learning rate at whatever the graph
def carried. Both matter for distillation: measured on atomic, value accuracy is
pinned at the label ceiling (0.726 vs 0.725) while policy_top1 sits at 0.41, so
equal weighting spends trunk capacity on a head with no headroom; and a constant
2e-4 is three orders of magnitude below the one-cycle peak used by the published
MultiAra supervised models (Gehrke 2021 S4.3.2).

The patch is kept HERE rather than committed to the open_spiel fork, so nothing
in this project ever writes to that repository.

```bash
cd $SCRATCH/open_spiel && git apply --check $AZ/pretrain/vpnet-recipe.patch \
  && git apply $AZ/pretrain/vpnet-recipe.patch
```

`--check` first: applying twice fails noisily, but a half-applied patch is worse.
If it reports the patch is already applied, skip it. Leave the change uncommitted
in that tree.

```bash
cd $SCRATCH/open_spiel/build && make -j az_pretrain
```

## 3. Get the data

```bash
mkdir -p $SCRATCH/data && cd $SCRATCH/data
# Atomic files are per-month: lichess_db_atomic_rated_YYYY-MM.pgn.zst
wget https://database.lichess.org/atomic/lichess_db_atomic_rated_2020-06.pgn.zst
zstd -d lichess_db_atomic_rated_2020-06.pgn.zst
```

Hold out a **separate month** for validation — never a random split, since
positions from one game are highly correlated and would leak across the split.
MultiAra used April 2018 as test and August 2018 as validation.

The `--min_elo=1900` default is that study's atomic threshold (top 10th
percentile). Both players must clear it.

## 4. Run it

```bash
export TRAIN_PGN=$SCRATCH/data/lichess_db_atomic_rated_2020-06.pgn
export VAL_PGN=$SCRATCH/data/lichess_db_atomic_rated_2018-08.pgn
export RUN_DIR=$SCRATCH/atomic_az/run_pretrained
bash pretrain/bootstrap_pretrained_run.sh
```

Or standalone, against an existing run directory:

```bash
$BUILD/examples/az_pretrain --path=$RUN_DIR --pgn=a.pgn,b.pgn \
    --val_pgn=val.pgn --epochs=7 --min_elo=1900 --device=/cuda:0
```

Large datasets can stream instead of being decompressed to disk:

```bash
zstdcat $SCRATCH/data/*.pgn.zst | $BUILD/examples/az_pretrain --path=$RUN_DIR --pgn=-
```

(stdin cannot be re-read, so this forces one epoch.)

## 5. Choosing an epoch

MultiAra found **7 epochs generalised better than 30** — hence the default. A
checkpoint is written per epoch (`checkpoint-1.pt` … `checkpoint-7.pt`) plus
validation metrics, so pick the best epoch rather than trusting the default:

```
[pretrain] EPOCH 3 VAL  value_mse 0.71  value_sign_acc 0.63  policy_top1 0.41
```

`policy_top1` is agreement with the human move; `value_sign_acc` is whether the
predicted winner is right, over decisive games only (comparable to the RL loop's
`value_accuracy`). When `policy_top1` stops improving while training loss keeps
falling, you are overfitting — use the earlier checkpoint:

```bash
cp $RUN_DIR/checkpoint-3.pt $RUN_DIR/checkpoint--1.pt
cp $RUN_DIR/checkpoint-3-optimizer.pt $RUN_DIR/checkpoint--1-optimizer.pt
```

## How it stays compatible with the RL loop

Verified against the OpenSpiel source rather than assumed:

- **Value sign.** The value head is player-0-relative — `VPNetEvaluator::Evaluate`
  returns `{v, -v}` for players `{0, 1}` (`vpevaluator.cc:73-77`) and the learner
  feeds `returns[0]` for every state in a trajectory. In OpenSpiel chess **player
  0 is Black** (`chess.h:73-78`), so a `1-0` PGN result becomes `value = -1`.
  Getting this backwards would train the net to invert every evaluation;
  `pgn_atomic_test.cc` asserts it explicitly.
- **Loss and optimizer.** Training goes through `VPNetModel::Learn`, so the
  masked policy loss, value loss, L2 term and Adam state are identical to RL's.
- **Checkpoint format.** Written by `VPNetModel::SaveCheckpoint`, which is what
  `LoadCheckpoint` reads (a model `.pt` + optimizer `.pt` pair, libtorch archive
  format — not a Python `state_dict`).
- **Move legality.** Moves are matched by comparing the PGN SAN token against
  `ActionToString()` over the state's own `LegalActions()`, never via
  `ParseSANMove`. Atomic's extra rules (no self-king explosion, exploding the
  enemy king wins even while in check) live in `AtomicChessState::LegalActions()`,
  so going through the state is the only way to be sure an action is legal.

## Gotchas

- **A fresh RL start destroys pretrained weights.** `alpha_zero.cc:585-595`
  calls `SaveCheckpoint(0)` from random init then loads it whenever `resuming`
  is false. Injection therefore *must* go through the resume path — that is what
  `bootstrap_pretrained_run.sh` sets up. Confirm the RL job prints
  `Loading model from step -1`, not `step 0`.
- **Never change `nn_width`/`nn_depth` afterwards.** `vpnet.pb` would be
  rewritten with a shape the checkpoints do not match.
- **Adam state carries over** from pretraining into RL. Harmless, but if you
  pretrained at a very different learning rate and RL looks unstable for the
  first few steps, that is why.
- The optimizer `.pt` roughly doubles checkpoint size (~2× the parameter bytes).
- Human data is not self-play data: the thesis notes humans resign often, so
  draw rates and value targets differ in character from self-play trajectories.
