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
# Games read PER FILE per epoch (-1 = all). This is how arms are held to equal
# samples seen. Arm A trains on 7 base shards; arm B on 14 (base + hard). At
# max_games=-1 arm B would see ~1.8x the data per epoch, so a same-epochs
# comparison would confound data VOLUME with data COMPOSITION -- and
# composition is the entire claim. Halve it for the 14-file arms.
MAX_GAMES="${MAX_GAMES:--1}"
# Architecture. 128x10 (~6M params) was sized for SELF-PLAY throughput, where
# the actor data rate is the scarce resource. Supervised distillation has no
# such constraint -- the data is already on disk -- and the policy head is far
# from its ceiling (policy_top1 ~0.41) even where the value head is pinned at
# it. These flags are IGNORED if RUN_DIR already contains a vpnet.pb, so a
# different size needs a fresh RUN_DIR.
NN_MODEL="${NN_MODEL:-resnet}"
NN_WIDTH="${NN_WIDTH:-128}"
NN_DEPTH="${NN_DEPTH:-10}"

# MultiAra's supervised recipe (Gehrke 2021 S4.3.2), for comparison with ours:
#   loss      0.99 policy / 0.01 value    ours: 1.0 / 1.0
#   optimizer NAG + one-cycle, 0.35 peak  ours: Adam, fixed 2e-4
#   epochs    7 (30 overfit)              ours: 4
#   batch     512                         ours: 1024
# The learning rate is the sharpest departure -- three orders of magnitude at
# peak, with no annealing on our side. LR_TOTAL_STEPS is required for onecycle
# and equals total_positions * reuse / batch_size.
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-1.0}"
VALUE_LOSS_WEIGHT="${VALUE_LOSS_WEIGHT:-1.0}"
LR_SCHEDULE="${LR_SCHEDULE:-none}"
MAX_LR="${MAX_LR:-0.05}"
MIN_LR="${MIN_LR:-0.00001}"
LR_TOTAL_STEPS="${LR_TOTAL_STEPS:-0}"

# Fitted over all 18,967,523 labelled positions by minimising weighted SSE
# between tanh(cp/K) and the observed outcome per centipawn bucket. Atomic
# swings far exceed chess: the chess default of 400 saturates (tanh(1000/400)
# = 0.99, where the real win rate at +1000cp is 0.56) and scores 8x the error
# (SSE/n 0.131 vs 0.015). Re-fit if the generator config changes.
CP_SCALE="${CP_SCALE:-1580}"
# Weighted toward the search score because --explore_prob injects random moves,
# which makes the game-RESULT component noisier than the per-position scores.
LAMBDA="${LAMBDA:-0.8}"
POLICY_TEMP="${POLICY_TEMP:-100}"

# TRAIN_TSV / VAL_TSV override the glob entirely. The glob only ever sees ONE
# directory, so the ablation arms -- which mix $DATA with $DATA/hard and the
# relabelled own-play set -- cannot be expressed through it. Pass explicit
# comma-separated lists instead:
#
#   arm A: TRAIN_TSV="$(ls -m $DATA/atomic.[0-6].tsv | tr -d ' \n')"
#   arm B: arm A + $DATA/hard/atomic.[0-6].tsv
#   arm C: arm B + the relabelled dagger_train tsv
#   all  : VAL_TSV=$DATA/t3_ownplay.tsv   (the frozen own-play test set)
#
# Whatever you pass, the holdout must NOT appear in the training list. Nothing
# here can check that for you -- an explicit list is exactly as careful as the
# person who wrote it.
if [[ -n "${TRAIN_TSV:-}" ]]; then
  TRAIN="$TRAIN_TSV"
else
  TRAIN=""
  for f in "$DATA"/atomic.*.tsv; do
    [[ -e "$f" ]] || { echo "no data in $DATA"; exit 1; }
    [[ "$f" == *".${HOLDOUT}.tsv" ]] && continue
    TRAIN="${TRAIN:+$TRAIN,}$f"
  done
fi
VAL="${VAL_TSV:-$DATA/atomic.${HOLDOUT}.tsv}"
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
  --max_games="$MAX_GAMES" \
  --init_checkpoint="$INIT_CHECKPOINT" \
  --epoch_offset="$EPOCH_OFFSET" \
  --batch_size=1024 \
  --buffer_size=262144 \
  --reuse="$REUSE" \
  --log_every=100 \
  --val_every=1000 \
  --save_every=1000 \
  --policy_loss_weight="$POLICY_LOSS_WEIGHT" \
  --value_loss_weight="$VALUE_LOSS_WEIGHT" \
  --lr_schedule="$LR_SCHEDULE" \
  --max_lr="$MAX_LR" --min_lr="$MIN_LR" \
  --lr_total_steps="$LR_TOTAL_STEPS" \
  --nn_model="$NN_MODEL" --nn_width="$NN_WIDTH" --nn_depth="$NN_DEPTH" \
  2>&1 | tee -a "$RUN_DIR/pretrain.log"

echo
echo "checkpoints in $RUN_DIR:"
ls -la "$RUN_DIR"/checkpoint-*.pt 2>/dev/null || echo "  none written"
echo
echo "Pick the epoch with the best policy_top1, then:"
echo "  cp $RUN_DIR/checkpoint-<N>.pt $RUN_DIR/checkpoint--1.pt"
echo "  cp $RUN_DIR/checkpoint-<N>-optimizer.pt $RUN_DIR/checkpoint--1-optimizer.pt"
echo "and hand the run to RL via pretrain/bootstrap_pretrained_run.sh."
