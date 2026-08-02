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
# Label for the logs. MUST be set when MA is overridden -- the released RL
# model and the supervised model are different networks and only the SL one
# works, so a mislabelled log is worse than useless.
MA_NAME="${MA_NAME:-MultiAra-RL-atomic-u25}"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$SCRATCH/logs"

GAMES="${GAMES:-400}"
# Equal simulations for both sides. The whole point of the protocol.
SIMS="${SIMS:-800}"
CKPT="${CKPT:--1}"

# Random plies per opening. THIS IS A REAL CONFOUND, not a detail. MultiAra's
# own tournaments used opening BOOKS (<=5 plies of common lines from
# ianfab/books); we use uniformly random legal moves, which in atomic produces
# far wilder positions -- hence the 5- and 7-ply games in every run. Our
# training data is built from random openings (mean 4 plies in the base set,
# mean 10 in hard/) and MultiAra's is not, so the default setting tests us in
# our home distribution and them outside theirs. OPENING_PLIES=2 is the control.
OPENING_PLIES="${OPENING_PLIES:-4}"
# EPD book (ianfab/books atomic.epd). Overrides OPENING_PLIES. This is the
# setting MultiAra was actually tested under; random openings favour whichever
# engine trained on them, and ours did.
BOOK="${BOOK:-}"
# Threads affects MultiAra's speed and, through virtual loss, introduces some
# nondeterminism. Recorded rather than tuned.
MA_THREADS="${MA_THREADS:-4}"

# Reuse_Tree is left at the binary default (true). It was suspected of causing
# mid-match degradation -- MultiAra looked competitive for ~8 pairs then lost 24
# straight -- and Gehrke's own config sets it False (Appendix C.1). TESTED AND
# DISPROVED: with reuse off the score was 83.6% (n=58) against 86.3% (n=40) with
# it on, statistically identical, and the losing streak did not reappear. It was
# variance. Reuse off also costs ~8x in speed, since MultiAra rebuilds its
# search every move, so it stays on.
#
# The transitivity argument that motivated the check was itself overstated: it
# assumed strength is one-dimensional, and in atomic at these search levels it
# is not. Deep alpha-beta finds forced explosion sequences neither network sees,
# so NN-vs-NN and NN-vs-searcher measure different things. Losing to Fairy-SF at
# 200k while beating MultiAra at matched simulations is not a contradiction.
MA_OPTS="${MA_OPTS:-Search_Type=mcts,Allow_Early_Stopping=false,Centi_Epsilon_Greedy=0,Threads=${MA_THREADS}}"

[[ -x "$BUILD/examples/az_vs_sf" ]] || { echo "az_vs_sf not built"; exit 1; }
[[ -x "$MA" ]] || { echo "MultiAra wrapper not found at $MA"; exit 1; }

nvidia-smi --query-gpu=name --format=csv,noheader
# Report only the opening source actually in effect: printing both
# opening_plies and book reads as though plies were used when a book
# overrides them.
if [[ -n "$BOOK" ]]; then OPENINGS="openings=book:$BOOK"; else OPENINGS="openings=$OPENING_PLIES-random-plies"; fi
echo "[multiara] run_dir=$RUN_DIR sims=$SIMS games=$GAMES threads=$MA_THREADS $OPENINGS opponent=$MA_NAME ma=$MA"

"$BUILD/examples/az_vs_sf" \
  --az_path="$RUN_DIR" \
  --az_checkpoint="$CKPT" \
  --sf_path="$MA" \
  --opponent_name="$MA_NAME" \
  --sf_opt="$MA_OPTS" \
  --games="$GAMES" \
  --opening_plies="$OPENING_PLIES" \
  --book="$BOOK" \
  --sf_nodes="$SIMS" \
  --az_sims="$SIMS" \
  --device=/cuda:0 \
  --value_calibration=false

echo "[multiara] done"
