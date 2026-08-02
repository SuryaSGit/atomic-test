#!/bin/bash
# Does our search actually convert? Score vs a FIXED opponent as our own
# simulation count doubles.
#
#   sbatch eval/sims_sweep.slurm
#   RUN_DIR=.../run_armB SF_NODES=50000 sbatch eval/sims_sweep.slurm
#
# WHY 50k NODES. The opponent must sit in the middle of our range or the
# measurement is clipped: at 10k we score ~81% (near ceiling) and at 200k ~16%
# with Black at 2% (floor). Neither can show a search effect. 50k should be the
# sensitive region.
#
# WHY THIS MATTERS. The original sweep on the pre-retrain net was FLAT --
# 13/8.8/10/11.2% at 200/400/800/1600 -- which is the anomaly, not the
# expected result. A healthy MCTS gains steadily with simulations. Candidate
# explanations, in the order I would act on them:
#   1. No MCTS-solver. OpenSpiel's MCTSBot averages backups; MultiAra ships
#      MCTS_Solver=true. In atomic, forced explosion mates are everywhere and
#      averaging dilutes a proven win into a mild positive, so more simulations
#      buy little.
#   2. Policy prior too weak (policy_top1 0.35-0.41) -- search cannot find what
#      the prior never proposes.
#   3. Value noise off-distribution (value_mse 0.126 on our positions vs 0.069
#      on the teacher's) -- deeper search accumulates noise.
#   4. uct_c=2.0 untuned for this network's value scale.
#
# CPU ON PURPOSE: no --gres, so this does not count against QOSMaxGRESPerUser
# and runs alongside GPU jobs. One array task per simulation count, so a slow
# setting cannot eat the walltime of the others.
#SBATCH --job-name=az_simsweep
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-4
#SBATCH --output=/u/%u/scratch/logs/%x-%A_%a.out
#SBATCH --error=/u/%u/scratch/logs/%x-%A_%a.out

set -uo pipefail
SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_armA4}"
SF="${SF:-$SCRATCH/fsf13/src/stockfish}"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
# Torch CPU inference is OpenMP-threaded; without this it runs single-threaded
# and the high-simulation tasks will not finish inside the walltime.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$SCRATCH/logs"

SIM_LEVELS=(0 400 800 1600 3200)
SIMS="${SIM_LEVELS[${SLURM_ARRAY_TASK_ID:-2}]}"

GAMES="${GAMES:-80}"
SF_NODES="${SF_NODES:-50000}"
SF_HASH="${SF_HASH:-256}"

[[ -x "$BUILD/examples/az_vs_sf" ]] || { echo "az_vs_sf not built"; exit 1; }
[[ -x "$SF" ]] || { echo "Fairy-Stockfish not found at $SF"; exit 1; }

echo "[simsweep] az_sims=$SIMS sf_nodes=$SF_NODES games=$GAMES threads=$OMP_NUM_THREADS"
date

stdbuf -oL "$BUILD/examples/az_vs_sf" \
  --az_path="$RUN_DIR" \
  --az_checkpoint=-1 \
  --sf_path="$SF" \
  --opponent_name="Fairy-Stockfish-13.1" \
  --games="$GAMES" \
  --sf_nodes="$SF_NODES" \
  --sf_hash="$SF_HASH" \
  --az_sims="$SIMS" \
  --sf_nnue=false \
  --value_calibration=false \
  --device=/cpu:0

date
echo "[simsweep] az_sims=$SIMS complete"
