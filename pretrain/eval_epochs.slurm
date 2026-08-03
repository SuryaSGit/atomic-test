#!/bin/bash
# Score every per-epoch checkpoint, on both the teacher's distribution and the
# frozen own-play set, and print one table.
#
#   sbatch pretrain/eval_epochs.slurm
#   RUN_DIR=.../run_armB EPOCHS=4 sbatch pretrain/eval_epochs.slurm
#
# WHY BOTH SETS. The teacher holdout says how well the net fits Stockfish; T3
# says how well it does on positions it actually reaches in play, which is the
# thing MCTS consumes. Measured on arm A4 the two disagree -- EDGE 0.197 on the
# teacher holdout against 0.168 on T3 -- and the T3 figure is the one that
# tracks playing strength.
#
# WHY NOT JUST TAKE THE LAST EPOCH. Under the one-cycle schedule the value head
# oscillates hard through the high-LR phase: measured on run_recipe, `pred mean`
# swung from -0.505 to +0.438 within 3,000 steps while policy_top1 barely moved.
# Epochs are not monotone, and a checkpoint whose value head happens to be
# biased will search badly no matter how good its policy looks.
#
# GPU, because CPU is not viable here: a 256x20 resnet is ~1.5 GFLOPs per
# position, so 60k positions is ~90 TFLOPs and takes 15-25 min per validation on
# 12 cores. A first attempt at CPU died at the 2h wall having scored two epochs.
#SBATCH --job-name=eval_epochs
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/u/%u/scratch/logs/%x-%j.out
#SBATCH --error=/u/%u/scratch/logs/%x-%j.out

set -uo pipefail
SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_recipe}"
BIN="$BUILD/examples/az_pretrain"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "$SCRATCH/logs"

EPOCHS="${EPOCHS:-7}"
CP_SCALE="${CP_SCALE:-1580}"
# 20k is ample for RANKING checkpoints -- the differences that matter are
# several points of policy_top1, not three decimal places.
VAL_SAMPLES="${VAL_SAMPLES:-20000}"
DEVICE="${DEVICE:-/cuda:0}"
# The training log already prints teacher-holdout numbers at every epoch
# boundary, so re-computing them here doubles the work for nothing. Set
# WITH_TEACHER=1 to score both.
WITH_TEACHER="${WITH_TEACHER:-0}"
TEACHER_VAL="${TEACHER_VAL:-$SCRATCH/data/atomic.7.tsv}"
OWNPLAY_VAL="${OWNPLAY_VAL:-$SCRATCH/data/t3_ownplay.tsv}"

[[ -x "$BIN" ]] || { echo "az_pretrain not built at $BIN"; exit 1; }

# Pulls the three numbers that matter out of a PrintVal block.
score() {  # score <checkpoint> <val_tsv>
  "$BIN" --path="$RUN_DIR" --epochs=0 --init_checkpoint="$1" \
         --val_sf_tsv="$2" --sf_cp_scale="$CP_SCALE" \
         --val_samples="$VAL_SAMPLES" --device="$DEVICE" 2>/dev/null \
  | awk '/policy_top1/ && /VALIDATE-ONLY/ {for(i=1;i<=NF;i++) if($i=="policy_top1") p=$(i+1)}
         /EDGE/        {for(i=1;i<=NF;i++) if($i=="EDGE") e=$(i+1)}
         /pred/        {for(i=1;i<=NF;i++) {if($i=="mean") m=$(i+1); if($i=="sd") s=$(i+1)}}
         END {printf "%-8s %-8s %-9s %-6s", (p==""?"-":p), (e==""?"-":e), (m==""?"-":m), (s==""?"-":s)}'
}

echo "run_dir : $RUN_DIR"
echo "teacher : $TEACHER_VAL"
echo "ownplay : $OWNPLAY_VAL"
echo
printf "%-6s | %-35s | %-35s\n" "epoch" "TEACHER holdout" "OWN-PLAY (T3)"
printf "%-6s | %-8s %-8s %-9s %-6s | %-8s %-8s %-9s %-6s\n" \
       "" "top1" "EDGE" "predmean" "sd" "top1" "EDGE" "predmean" "sd"
printf -- "-%.0s" {1..80}; echo

for N in $(seq 1 "$EPOCHS"); do
  [[ -f "$RUN_DIR/checkpoint-$N.pt" ]] || continue
  printf "%-6s | " "$N"
  if [[ "$WITH_TEACHER" == "1" ]]; then
    [[ -s "$TEACHER_VAL" ]] && score "$N" "$TEACHER_VAL" || printf "%-33s" "(missing)"
  else
    printf "%-33s" "(see training log)"
  fi
  printf " | "
  [[ -s "$OWNPLAY_VAL" ]] && score "$N" "$OWNPLAY_VAL" || printf "%-33s" "(missing)"
  echo
done

echo
echo "Reference, arms A4 / B at 128x10:"
echo "  teacher holdout : top1 0.406   EDGE 0.194"
echo "  own-play (T3)   : top1 0.350   EDGE 0.168"
echo
echo "Pick on the OWN-PLAY column, and prefer predmean near 0 -- a biased value"
echo "head searches badly however good its policy looks. Then promote it:"
echo "  cp $RUN_DIR/checkpoint-<N>.pt $RUN_DIR/checkpoint--1.pt"
echo "  cp $RUN_DIR/checkpoint-<N>-optimizer.pt $RUN_DIR/checkpoint--1-optimizer.pt"
echo "and confirm by PLAY before trusting it:"
echo "  RUN_DIR=$RUN_DIR CKPT=<N> BOOK=$SCRATCH/books/atomic.epd GAMES=200 \\"
echo "    sbatch --array=2-2 $SCRATCH/atomic_az/eval/ladder.slurm"
