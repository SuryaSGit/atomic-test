#!/bin/bash
# Generate T3: the FROZEN own-play test set for the Phase 2 ablation.
#
# Plays the arm-A checkpoint against Fairy-Stockfish and dumps every game as UCI
# move lines, for relabelling by sf_label. The relabelled positions become the
# test set on which ALL arms are scored.
#
#   sbatch eval/t3_dump.slurm                    # 12 shards x 100 games = 1200
#   GAMES=200 sbatch eval/t3_dump.slurm          # more per shard
#
# Then concatenate (the '#' header of each shard is a comment, so cat is safe):
#   cat $SCRATCH/data/t3_ownplay.s*.games > $SCRATCH/data/t3_ownplay.games
#
# WHY THIS IS FROZEN: every arm must be evaluated on the SAME positions. If each
# arm is tested on positions it generated itself, the arms are scored on
# different distributions and the comparison is meaningless. Generate once, from
# arm A, and never regenerate.
#
# WHY AN ARRAY OF SHORT SHARDS: az_vs_sf has no resume, and --dump_games opens
# the file with truncate. On a preemptible partition a single long job would
# restart from zero AND destroy its partial dump. One file per shard bounds the
# loss to that shard, and each shard is sized to finish inside a typical
# scavenger window (measured on job 9664549: attempts of 20s, 11m, 19m, 2h52m).
#SBATCH --job-name=t3_dump
#SBATCH --partition=scavenger
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-12
#SBATCH --requeue
#SBATCH --output=/u/%u/scratch/logs/%x-%A_%a.out
#SBATCH --error=/u/%u/scratch/logs/%x-%A_%a.out

set -uo pipefail
module purge && module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_pretrained}"
SF="${SF:-$SCRATCH/Fairy-Stockfish/src/stockfish}"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$SCRATCH/data" "$SCRATCH/logs"

SHARD="${SLURM_ARRAY_TASK_ID:-1}"

# ---- FIXED PARAMETERS: these define the distribution T3 represents. ----------
# Changing any of them invalidates the comparison across arms. Recorded here so
# a later shard can be generated identically.
GAMES="${GAMES:-100}"          # per shard; 12 shards -> ~1200 games, ~20k positions
SF_NODES="${SF_NODES:-10000}"  # matches the label generator; NOT the 100k default
AZ_SIMS="${AZ_SIMS:-400}"
OPENING_PLIES="${OPENING_PLIES:-4}"
# -----------------------------------------------------------------------------

# Seed differs per shard, so shards are independent samples of the SAME
# distribution rather than N copies of one match. It drives both the random
# openings and the MCTS (az_vs_sf.cc: make_bot(seed + pair*2 + side)).
#
# SEED_BASE separates whole DATASETS. This matters more than it looks: T3 was
# generated at seed 1, so any later run that also lands on seed 1 reproduces
# T3's openings exactly -- putting the test set into the training data. Always
# give a training dump its own SEED_BASE, far from T3's range.
#
#   T3 (test)          : SEED_BASE=0     -> seeds 1..N
#   DAgger (train)     : SEED_BASE=1000  -> seeds 1001..1000+N
#
# Do NOT pass SEED directly when running an array: it would pin every shard to
# the same value and produce N identical copies of one match.
SEED_BASE="${SEED_BASE:-0}"
SEED="${SEED:-$((SEED_BASE + SHARD))}"

# PREFIX names the dataset; the shard index is always appended, so shards can
# never overwrite each other. Do NOT pass OUT directly when running an array.
PREFIX="${PREFIX:-t3_ownplay}"
OUT="${OUT:-${PREFIX}.s${SHARD}.games}"

# checkpoint -1 is arm A *right now*. Once an arm is retrained in this RUN_DIR
# that stops being true, which is why armA-baseline.pt exists.
CKPT="${CKPT:--1}"

[[ -x "$BUILD/examples/az_vs_sf" ]] || { echo "az_vs_sf not built"; exit 1; }
[[ -x "$SF" ]] || { echo "Fairy-Stockfish not found at $SF"; exit 1; }

nvidia-smi
echo "[t3] shard=$SHARD games=$GAMES sf_nodes=$SF_NODES az_sims=$AZ_SIMS" \
     "seed=$SEED ckpt=$CKPT out=$OUT"

"$BUILD/examples/az_vs_sf" \
  --az_path="$RUN_DIR" \
  --az_checkpoint="$CKPT" \
  --sf_path="$SF" \
  --games="$GAMES" \
  --sf_nodes="$SF_NODES" \
  --az_sims="$AZ_SIMS" \
  --opening_plies="$OPENING_PLIES" \
  --seed="$SEED" \
  --device=/cuda:0 \
  --sf_nnue=false \
  --value_calibration=true \
  --dump_games="$SCRATCH/data/$OUT"

echo
echo "[t3] shard $SHARD complete:"
wc -l "$SCRATCH/data/$OUT"
