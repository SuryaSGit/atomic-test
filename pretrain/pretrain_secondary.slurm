#!/bin/bash
# Supervised pretraining on the SHORT queue, so it does not compete with the
# long-running RL job on scavenger. Writes checkpoint--1.pt into RUN_DIR.
#
#   sbatch pretrain/pretrain_secondary.slurm
#   EPOCHS=1 RUN_DIR=.../run_pretrained_test sbatch pretrain/pretrain_secondary.slurm
#
# First time only:  mkdir -p /u/$USER/scratch/logs
#SBATCH --job-name=atomic_pretrain
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
# Any GPU, and a small footprint, so this backfills quickly. Naming a specific
# model (gpu:H100:1) left it pending behind reserved nodes. Data loading is
# single-threaded, so extra cores buy nothing.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Sized for a 1-epoch validation run. A full 7-epoch pass on ~18M positions
# needs longer:  sbatch --time=04:00:00 pretrain/pretrain_secondary.slurm
#SBATCH --time=01:00:00
#SBATCH --output=/u/%u/scratch/logs/%x-%j.out
#SBATCH --error=/u/%u/scratch/logs/%x-%j.out

set -uo pipefail
module purge && module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
DATA="${DATA:-$SCRATCH/data}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_pretrained}"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$RUN_DIR" "$SCRATCH/logs"

# Held-out shard for validation; everything else trains. Shards are independent
# (different RNG seeds), so this is a clean split -- unlike a random split of
# positions, which would leak because positions within a game are correlated.
HOLDOUT="${HOLDOUT:-7}"
EPOCHS="${EPOCHS:-7}"          # MultiAra: 7 generalised better than 30
# Chaining, for when the epochs do not fit inside the partition walltime:
#   job 1: EPOCHS=2
#   job 2: EPOCHS=2 INIT_CHECKPOINT=-1 EPOCH_OFFSET=2, submitted with
#          sbatch --dependency=afterok:<job1>
# INIT_CHECKPOINT=-1 resumes from checkpoint--1 AND its Adam state, so the
# continuation is seamless. EPOCH_OFFSET keeps per-epoch checkpoints unique.
INIT_CHECKPOINT="${INIT_CHECKPOINT:--2}"   # -2 = fresh weights
EPOCH_OFFSET="${EPOCH_OFFSET:-0}"
# Positions between learn steps = batch_size/reuse. Raising reuse buys more
# gradient steps from the same parsed data -- useful because the single-threaded
# reader takes a large share of wall time -- at the cost of seeing each position
# more often.
REUSE="${REUSE:-4}"

# Calibrated on 579k labelled positions from this dataset by fitting
# tanh(cp/K) to the observed outcome. Atomic swings far exceed chess: the
# chess default of 400 saturates (tanh(1000/400)=0.99 where the real win rate
# at +1000cp is 0.56). Re-run the calibration if the generator config changes.
CP_SCALE="${CP_SCALE:-1620}"
# Weighted toward the search score because --explore_prob injects random moves,
# which makes the game-RESULT component noisier than the per-position scores.
LAMBDA="${LAMBDA:-0.8}"
POLICY_TEMP="${POLICY_TEMP:-100}"

TRAIN=""
for f in "$DATA"/atomic.*.tsv; do
  [[ -e "$f" ]] || { echo "no data in $DATA"; exit 1; }
  [[ "$f" == *".${HOLDOUT}.tsv" ]] && continue
  TRAIN="${TRAIN:+$TRAIN,}$f"
done
VAL="$DATA/atomic.${HOLDOUT}.tsv"
[[ -s "$VAL" ]] || { echo "holdout shard $VAL missing"; exit 1; }

nvidia-smi
echo "[pretrain] train shards: $TRAIN"
echo "[pretrain] holdout     : $VAL"

# Fail fast on a corrupt dataset rather than after burning GPU time. The
# checker replays every game and compares against the stored result.
if [[ -x "$BUILD/games/sf_data_check" ]]; then
  echo "[pretrain] validating dataset..."
  "$BUILD/games/sf_data_check" "$DATA"/atomic.*.tsv || {
    echo "[pretrain] dataset failed validation; aborting"; exit 1; }
fi

"${BUILD}/examples/az_pretrain" \
  --path="$RUN_DIR" \
  --graph_def=vpnet.pb \
  --device=/cuda:0 \
  --sf_tsv="$TRAIN" \
  --val_sf_tsv="$VAL" \
  --sf_lambda="$LAMBDA" \
  --sf_cp_scale="$CP_SCALE" \
  --sf_policy_temp="$POLICY_TEMP" \
  --epochs="$EPOCHS" \
  --init_checkpoint="$INIT_CHECKPOINT" \
  --epoch_offset="$EPOCH_OFFSET" \
  --batch_size=1024 \
  --buffer_size=262144 \
  --reuse="$REUSE" \
  --log_every=100 \
  --val_every=1000 \
  --save_every=1000 \
  --nn_model=resnet --nn_width=128 --nn_depth=10 \
  2>&1 | tee -a "$RUN_DIR/pretrain.log"

echo
echo "checkpoints in $RUN_DIR:"
ls -la "$RUN_DIR"/checkpoint-*.pt 2>/dev/null || echo "  none written"
echo
echo "Pick the epoch with the best policy_top1, then:"
echo "  cp $RUN_DIR/checkpoint-<N>.pt $RUN_DIR/checkpoint--1.pt"
echo "  cp $RUN_DIR/checkpoint-<N>-optimizer.pt $RUN_DIR/checkpoint--1-optimizer.pt"
echo "and hand the run to RL via pretrain/bootstrap_pretrained_run.sh."
