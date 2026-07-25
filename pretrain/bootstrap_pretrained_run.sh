#!/usr/bin/env bash
# Bootstrap an RL run from a supervised-pretrained net.
#
# WHY THIS IS FIDDLY: alpha_zero_torch_example destroys pretrained weights on a
# FRESH start. Read alpha_zero.cc:585-595 -- when `resuming` is false it does
#     SaveCheckpoint(0)   // writes RANDOM init
#     LoadCheckpoint(0)   // loads it back
# so any checkpoint we placed beforehand is overwritten. It also rewrites
# vpnet.pb whenever config.graph_def is empty (i.e. whenever flags are used).
#
# The RESUME path does neither: it skips the save, loads checkpoint--1, and
# reports "Using existing model" because config.json carries graph_def. So the
# only safe way to inject a pretrained net is to make the RL job resume:
#
#   Phase 1  run the RL binary briefly with the real flags -> vpnet.pb + config.json
#   Phase 2  pretrain against THAT vpnet.pb                -> checkpoint--1.pt
#   Phase 3  seed learner.jsonl with a step-0 record       -> resume is valid
#   Phase 4  launch with the positional config.json        -> loads our weights
#
# Architecture lives in exactly one place (phase 1's flags), so vpnet.pb and the
# checkpoints cannot drift apart.
set -euo pipefail

SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_pretrained}"
AZ_BIN="${BUILD}/examples/alpha_zero_torch_example"
PRETRAIN_BIN="${BUILD}/examples/az_pretrain"

# Comma-separated DECOMPRESSED PGN files. e.g.
#   TRAIN_PGN=$SCRATCH/data/atomic_train.pgn VAL_PGN=$SCRATCH/data/atomic_val.pgn
TRAIN_PGN="${TRAIN_PGN:?set TRAIN_PGN to comma-separated decompressed PGN files}"
VAL_PGN="${VAL_PGN:-}"
EPOCHS="${EPOCHS:-7}"
MIN_ELO="${MIN_ELO:-1900}"
DEVICE="${DEVICE:-/cuda:0}"
NPROC="$(nproc 2>/dev/null || echo 8)"

# Architecture (must match between phases -- phase 1 owns it).
NN_MODEL=resnet
NN_WIDTH=128
NN_DEPTH=10
LEARNING_RATE=0.0002
WEIGHT_DECAY=0.0001

for bin in "$AZ_BIN" "$PRETRAIN_BIN"; do
  [[ -x "$bin" ]] || { echo "missing binary: $bin" >&2; exit 1; }
done
mkdir -p "$RUN_DIR"

if [[ -f "${RUN_DIR}/learner.jsonl" ]]; then
  echo "[bootstrap] ${RUN_DIR} already bootstrapped; refusing to clobber it."
  echo "[bootstrap] Use a different RUN_DIR, or delete it to start over."
  exit 1
fi

# ---------------------------------------------------------------------------
echo "[bootstrap] phase 1/4: generating vpnet.pb + config.json"
# ---------------------------------------------------------------------------
"${AZ_BIN}" \
  --game=atomic_chess \
  --path="${RUN_DIR}" \
  --nn_model="${NN_MODEL}" \
  --nn_width="${NN_WIDTH}" \
  --nn_depth="${NN_DEPTH}" \
  --devices="${DEVICE}" \
  --learning_rate="${LEARNING_RATE}" \
  --weight_decay="${WEIGHT_DECAY}" \
  --train_batch_size=2048 \
  --inference_batch_size=64 \
  --inference_threads=2 \
  --inference_cache=262144 \
  --replay_buffer_size=262144 \
  --replay_buffer_reuse=4 \
  --max_simulations=300 \
  --uct_c=2.0 \
  --policy_alpha=0.3 \
  --policy_epsilon=0.25 \
  --temperature=1.0 \
  --temperature_drop=6 \
  --cutoff_probability=0.8 \
  --cutoff_value=0.95 \
  --actors="$(( NPROC > 4 ? NPROC - 2 : 2 ))" \
  --evaluators=1 \
  --eval_levels=4 \
  --checkpoint_freq=25 \
  --evaluation_window=30 \
  --max_steps=0 \
  > "${RUN_DIR}/bootstrap_phase1.log" 2>&1 &
AZ_PID=$!

for _ in $(seq 1 120); do
  if [[ -f "${RUN_DIR}/config.json" && -f "${RUN_DIR}/vpnet.pb" ]]; then break; fi
  kill -0 "$AZ_PID" 2>/dev/null || break
  sleep 1
done
kill "$AZ_PID" 2>/dev/null || true
wait "$AZ_PID" 2>/dev/null || true

for f in config.json vpnet.pb; do
  [[ -s "${RUN_DIR}/${f}" ]] || {
    echo "[bootstrap] phase 1 failed to produce ${f}; see bootstrap_phase1.log" >&2
    exit 1
  }
done
# Random-init leftover from the fresh start; the resume path ignores it.
rm -f "${RUN_DIR}/checkpoint-0.pt" "${RUN_DIR}/checkpoint-0-optimizer.pt"
echo "[bootstrap] phase 1 OK"

# ---------------------------------------------------------------------------
echo "[bootstrap] phase 2/4: supervised pretraining (${EPOCHS} epochs)"
# ---------------------------------------------------------------------------
VAL_ARG=()
[[ -n "$VAL_PGN" ]] && VAL_ARG=(--val_pgn="${VAL_PGN}")
"${PRETRAIN_BIN}" \
  --path="${RUN_DIR}" \
  --graph_def=vpnet.pb \
  --device="${DEVICE}" \
  --pgn="${TRAIN_PGN}" \
  "${VAL_ARG[@]}" \
  --epochs="${EPOCHS}" \
  --min_elo="${MIN_ELO}" \
  --batch_size=1024 \
  --buffer_size=262144 \
  --reuse=4 \
  2>&1 | tee "${RUN_DIR}/pretrain.log"

[[ -s "${RUN_DIR}/checkpoint--1.pt" ]] || {
  echo "[bootstrap] pretraining did not write checkpoint--1.pt" >&2
  exit 1
}
echo "[bootstrap] phase 2 OK: $(ls -la "${RUN_DIR}/checkpoint--1.pt")"

# ---------------------------------------------------------------------------
echo "[bootstrap] phase 3/4: seeding learner.jsonl"
# ---------------------------------------------------------------------------
# StartInfoFromLearnerJson (alpha_zero.cc:67-93) reads the last non-empty line
# for time_rel / step / total_trajectories. step=0 gives start_step=1, which
# keeps the `start_step > 1` guard false so it does NOT try to LoadBuffer a
# replay_buffer.data that does not exist yet.
printf '{"time_abs": 0.0, "time_rel": 0.0, "step": 0, "total_trajectories": 0}\n' \
  > "${RUN_DIR}/learner.jsonl"
echo "[bootstrap] phase 3 OK"

# ---------------------------------------------------------------------------
echo "[bootstrap] phase 4/4: ready to train"
# ---------------------------------------------------------------------------
cat <<EOF

Bootstrapped: ${RUN_DIR}

Launch (or sbatch a job whose RUN_DIR points here) with:
    ${AZ_BIN} ${RUN_DIR}/config.json

It must print:
    Using existing model: ${RUN_DIR}/vpnet.pb
    Loading model from step -1
"Loading model from step 0" or "Overwriting existing model" means it started
fresh and threw the pretrained weights away -- stop and re-check phases 1-3.
EOF
