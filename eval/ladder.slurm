#!/bin/bash
# Strength ladder against Fairy-Stockfish 13.1, classical evaluation.
#
#   sbatch eval/ladder.slurm                       # arm A4, 4 rungs
#   RUN_DIR=.../run_armB sbatch eval/ladder.slurm  # another arm
#
# WHY THESE SETTINGS: MultiAra's published atomic result -- "about even" with
# Fairy-Stockfish 13.1 classical eval (Gehrke 2021 S5.2) -- was measured at
# 10s/game + 0.1s/move. Over ~36-ply games that is ~0.65s per move, and 13.1
# measures ~1.0M nps on this hardware, so the MultiAra-equivalent budget is
# ~650k nodes. Our own side gets 1300 simulations, its ~0.65s equivalent.
#
# Anything below 650k is a rung on the way, NOT a comparable result. Quoting a
# score against 10k nodes as if it were near MultiAra's condition overstates
# the position by a factor of ~65 in opponent search.
#SBATCH --job-name=az_ladder
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=sridhar-ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-4
#SBATCH --output=/u/%u/scratch/logs/%x-%A_%a.out
#SBATCH --error=/u/%u/scratch/logs/%x-%A_%a.out

set -uo pipefail
module purge && module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_armA4}"
# 13.1 specifically, and built with nnue=no, so classical evaluation is a
# property of the binary rather than a UCI option that can be forgotten.
SF="${SF:-$SCRATCH/fsf13/src/stockfish}"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$SCRATCH/logs"

# One rung per array task.
RUNGS=(0 10000 50000 200000 650000)
NODES="${RUNGS[${SLURM_ARRAY_TASK_ID:-4}]}"

GAMES="${GAMES:-400}"
AZ_SIMS="${AZ_SIMS:-1300}"
# Fixed and reported. Fairy-SF's default saturates (hashfull 988/1000 by 3M
# nodes) and atomic scores move with hash size -- measured 147cp mean shift
# between 64MB and 1MB on identical positions.
SF_HASH="${SF_HASH:-256}"
# EPD opening book (ianfab/books atomic.epd), the condition MultiAra was
# actually tested under. Overrides the random-ply openings. Both ladders
# must use the SAME setting or the comparison is meaningless.
BOOK="${BOOK:-}"

CKPT="${CKPT:--1}"

[[ -x "$BUILD/examples/az_vs_sf" ]] || { echo "az_vs_sf not built"; exit 1; }
[[ -x "$SF" ]] || { echo "Fairy-Stockfish 13.1 not found at $SF"; exit 1; }

nvidia-smi --query-gpu=name --format=csv,noheader
echo "[ladder] rung=${SLURM_ARRAY_TASK_ID:-4} sf_nodes=$NODES az_sims=$AZ_SIMS" \
     "games=$GAMES hash=${SF_HASH}MB book=$BOOK run_dir=$RUN_DIR sf=$SF"

"$BUILD/examples/az_vs_sf" \
  --az_path="$RUN_DIR" \
  --az_checkpoint="$CKPT" \
  --sf_path="$SF" \
  --games="$GAMES" \
  --sf_nodes="$NODES" \
  --sf_hash="$SF_HASH" \
  --az_sims="$AZ_SIMS" \
  --book="$BOOK" \
  --sf_nnue=false \
  --value_calibration=false \
  --device=/cuda:0

echo "[ladder] rung sf_nodes=$NODES complete"
