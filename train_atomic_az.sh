#!/usr/bin/env bash
# Launch OpenSpiel C++ (LibTorch) AlphaZero self-play training on atomic_chess.
# Resume-aware: if RUN_DIR already has a config.json, we RESUME from the latest
# checkpoint (the torch-AZ example resumes when given a positional config.json;
# it starts fresh when given --flags). This makes the job survive scavenger
# preemption + --requeue: the requeued attempt continues where it left off.
set -uo pipefail

# On ICC everything lives on scratch ($HOME is tiny). RUN_DIR/OPEN_SPIEL_BUILD
# are normally exported by the .slurm wrapper; these are just fallbacks.
SCRATCH="/u/${USER}/scratch"
OPEN_SPIEL_BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_main}"
AZ_BIN="${OPEN_SPIEL_BUILD}/examples/alpha_zero_torch_example"
NPROC="$(nproc 2>/dev/null || echo 8)"
CONFIG="${RUN_DIR}/config.json"

mkdir -p "${RUN_DIR}"

if [[ -f "${CONFIG}" ]]; then
  echo "[train] RESUMING from ${CONFIG} (latest checkpoint)"
  "${AZ_BIN}" "${CONFIG}" 2>&1 | tee -a "${RUN_DIR}/train.log"
else
  echo "[train] FRESH START in ${RUN_DIR}"
  # atomic_chess has chess's action space (~4674) and a 16x8x8 observation, so
  # use a chess-scale ResNet. Actors do CPU self-play with GPU-batched inference
  # via the evaluators, so actors ~= available CPU cores, 1-2 GPU evaluators.
  "${AZ_BIN}" \
    --game=atomic_chess \
    --path="${RUN_DIR}" \
    --nn_model=resnet \
    --nn_width=256 \
    --nn_depth=20 \
    --devices=/cuda:0 \
    --learning_rate=0.0002 \
    --weight_decay=0.0001 \
    --train_batch_size=2048 \
    --inference_batch_size=64 \
    --inference_threads=2 \
    --inference_cache=262144 \
    --replay_buffer_size=1048576 \
    --replay_buffer_reuse=4 \
    --max_simulations=300 \
    --uct_c=2.0 \
    --policy_alpha=0.3 \
    --policy_epsilon=0.25 \
    --temperature=1.0 \
    --temperature_drop=20 \
    --cutoff_probability=0.8 \
    --cutoff_value=0.95 \
    --actors="$(( NPROC > 4 ? NPROC - 2 : 2 ))" \
    --evaluators=2 \
    --eval_levels=7 \
    --checkpoint_freq=50 \
    --evaluation_window=100 \
    --max_steps=0 \
    2>&1 | tee -a "${RUN_DIR}/train.log"
fi
