#!/bin/bash
# Place MultiAra on the SAME ladder as our network.
#
#   sbatch eval/multiara_vs_sf.slurm
#   MA_NODES=6400 GAMES=200 sbatch eval/multiara_vs_sf.slurm
#
# WHY. Our two strength measurements cannot currently be reconciled:
#   ours vs MultiAra @ matched sims ....... 75-89%   (n~128)
#   ours vs Fairy-SF 13.1 @ 200k nodes .... 18.1%    (n=400)
# and MultiAra is published as "about even" with Fairy-SF 13.1 classical
# (Gehrke 2021 S5.2). Those three cannot all hold under Elo transitivity. This
# measures the missing edge of the triangle directly, in our own units, on our
# own hardware, through the same harness -- so the answer is evidence rather
# than inference.
#
# The rungs match ladder.slurm exactly (10k / 50k / 200k, Hash=256, classical
# eval, paired openings), and MA_NODES defaults to 1300 -- the same per-move
# budget our net was given in that ladder. Read the two side by side.
#
# CPU ONLY: MultiAra's MKL build needs no GPU, and node-matched play is
# hardware-independent, so this costs nothing against QOSMaxGRESPerUser.
#
# CAVEAT that survives any result: MultiAra's own tournaments used opening
# BOOKS (<=5 plies, ianfab/books) and a 10s+0.1s time control on a GPU with 3
# threads. We use 4 uniformly random plies and fixed nodes. A score here is not
# their published condition.
#SBATCH --job-name=ma_vs_sf
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=1-3
#SBATCH --output=/u/%u/scratch/logs/%x-%A_%a.out
#SBATCH --error=/u/%u/scratch/logs/%x-%A_%a.out

set -uo pipefail
SCRATCH="/u/${USER}/scratch"
BIN="${BIN:-$SCRATCH/open_spiel/build/games/uci_vs_uci}"
MA="${MA:-$SCRATCH/multiara/multiara-uci}"
SF="${SF:-$SCRATCH/fsf13/src/stockfish}"
mkdir -p "$SCRATCH/logs"

RUNGS=(0 10000 50000 200000)
SF_NODES="${RUNGS[${SLURM_ARRAY_TASK_ID:-1}]}"

# 1300 matches what our net received in ladder.slurm, so the two ladders are
# directly comparable. Search_Type=mcts is MANDATORY -- the binary defaults to
# mcgs, which Gehrke S4.4.5 abandoned for atomic because it "regularly
# generated NaN".
MA_NODES="${MA_NODES:-1300}"
MA_OPTS="${MA_OPTS:-Search_Type=mcts,Allow_Early_Stopping=false,Centi_Epsilon_Greedy=0,Threads=${SLURM_CPUS_PER_TASK:-8}}"
SF_HASH="${SF_HASH:-256}"
GAMES="${GAMES:-100}"

[[ -x "$BIN" ]] || { echo "uci_vs_uci not built at $BIN"; exit 1; }
[[ -x "$MA" ]] || { echo "MultiAra wrapper not found at $MA"; exit 1; }
[[ -x "$SF" ]] || { echo "Fairy-Stockfish 13.1 not found at $SF"; exit 1; }

echo "[ma_vs_sf] ma_nodes=$MA_NODES sf_nodes=$SF_NODES games=$GAMES hash=${SF_HASH}MB"
date

stdbuf -oL "$BIN" \
  --e1_path="$MA" \
  --e1_name="MultiAra-RL-u25" \
  --e1_nodes="$MA_NODES" \
  --e1_opt="$MA_OPTS" \
  --e2_path="$SF" \
  --e2_name="Fairy-SF-13.1" \
  --e2_nodes="$SF_NODES" \
  --e2_opt="Hash=${SF_HASH}" \
  --games="$GAMES"

date
echo "[ma_vs_sf] sf_nodes=$SF_NODES complete"
