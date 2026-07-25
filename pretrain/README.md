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

## Files

| file | needs LibTorch | what it does |
|---|---|---|
| `pgn_atomic.h` | no | streaming PGN reader → training samples |
| `pgn_atomic_test.cc` | no | tests for the above; run this first |
| `az_pretrain.cc` | **yes** | trains `VPNetModel`, writes `checkpoint--1.pt` |
| `bootstrap_pretrained_run.sh` | — | wires the pretrained net into an RL run |

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
cp pretrain/az_pretrain.cc pretrain/pgn_atomic.h $SCRATCH/open_spiel/open_spiel/examples/
```

Add to `open_spiel/examples/CMakeLists.txt`:

```cmake
if (${OPEN_SPIEL_BUILD_WITH_LIBTORCH})
  add_executable(az_pretrain az_pretrain.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(az_pretrain ${TORCH_LIBRARIES})
endif()
```

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
