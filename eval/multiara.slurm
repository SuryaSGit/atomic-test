#!/bin/bash
# Head-to-head against MultiAra's published atomic model.
#
#   sbatch eval/multiara.slurm
#   RUN_DIR=.../run_armB SIMS=1600 sbatch eval/multiara.slurm
#
# WHAT THIS IS. MultiAra is the only published engine for atomic (Gehrke 2021).
# Its RL model for atomic is model-...-0025, iteration 25 -- the final one, since
# "training stopped for Atomic after update 25". We play it directly rather than
# triangulating through Fairy-Stockfish, because a proxy cannot settle a claim
# about MultiAra no matter how carefully the opponent is matched.
#
# WHAT THIS IS NOT. This is NOT MultiAra's published condition. Gehrke ran
# 10s/game + 0.1s/move on a GPU with 3 threads. We match SIMULATIONS instead,
# which is hardware-independent and fully specified but is a different question:
# it compares the networks plus their searches at equal simulation counts. Do
# not describe any result here as beating or matching the thesis numbers.
#
# MANDATORY SETTING. Search_Type defaults to mcgs; Gehrke S4.4.5 abandoned MCGS
# for atomic because it "regularly generated NaN ... which is why we decided to
# train Atomic with MCTS". A match run under mcgs measures a mode the authors
# rejected. Any result without Search_Type=mcts in the log is void.
#SBATCH --job-name=az_multiara
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=sridhar-ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/u/%u/scratch/logs/%x-%j.out
#SBATCH --error=/u/%u/scratch/logs/%x-%j.out

set -uo pipefail
module purge && module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_armA4}"
MA="${MA:-$SCRATCH/multiara/multiara-uci}"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$SCRATCH/logs"

GAMES="${GAMES:-400}"
# Equal simulations for both sides. The whole point of the protocol.
SIMS="${SIMS:-800}"
CKPT="${CKPT:--1}"
# Threads affects MultiAra's speed and, through virtual loss, introduces some
# nondeterminism. Recorded rather than tuned.
MA_THREADS="${MA_THREADS:-4}"

# Reuse_Tree defaults to TRUE in the binary but Gehrke's own config sets it
# False (Appendix C.1). Our harness sends a full `position fen` before every
# move and `ucinewgame` between games, so a reused tree can go stale across
# those jumps. Symptom observed: MultiAra competitive for ~8 pairs, then losing
# 24 straight -- an engine degrading mid-match, not one being outplayed.
#
# The transitivity check makes this decisive. MultiAra is published as "about
# even" with Fairy-SF 13.1 classical at ~2M nodes; we score 15.8% against that
# engine at only 200k nodes. We should therefore LOSE to MultiAra heavily. A
# measured 86% in our favour is not a result, it is a misconfiguration.
MA_OPTS="${MA_OPTS:-Search_Type=mcts,Allow_Early_Stopping=false,Centi_Epsilon_Greedy=0,Reuse_Tree=false,Threads=${MA_THREADS}}"

[[ -x "$BUILD/examples/az_vs_sf" ]] || { echo "az_vs_sf not built"; exit 1; }
[[ -x "$MA" ]] || { echo "MultiAra wrapper not found at $MA"; exit 1; }

nvidia-smi --query-gpu=name --format=csv,noheader
echo "[multiara] run_dir=$RUN_DIR sims=$SIMS games=$GAMES threads=$MA_THREADS"

"$BUILD/examples/az_vs_sf" \
  --az_path="$RUN_DIR" \
  --az_checkpoint="$CKPT" \
  --sf_path="$MA" \
  --opponent_name="MultiAra-RL-atomic-u25" \
  --sf_opt="$MA_OPTS" \
  --games="$GAMES" \
  --sf_nodes="$SIMS" \
  --az_sims="$SIMS" \
  --device=/cuda:0 \
  --value_calibration=false

echo "[multiara] done"
