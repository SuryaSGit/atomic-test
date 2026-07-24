#!/usr/bin/env bash
# Launch OpenSpiel C++ (LibTorch) AlphaZero self-play training on atomic_chess.
# Run on a GPU node. Assumes OpenSpiel was built with
# OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON (see README) and that
# `alpha_zero_torch_example` is on PATH or in ${OPEN_SPIEL_BUILD}.
set -euo pipefail

# On ICC everything lives on scratch ($HOME is tiny). RUN_DIR/OPEN_SPIEL_BUILD
# are normally exported by the .slurm wrapper; these are just fallbacks.
SCRATCH="/u/${USER}/scratch"
OPEN_SPIEL_BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN_DIR="${RUN_DIR:-$SCRATCH/atomic_az/run_$(date +%Y%m%d_%H%M%S)}"
AZ_BIN="${OPEN_SPIEL_BUILD}/examples/alpha_zero_torch_example"
NPROC="$(nproc 2>/dev/null || echo 8)"

mkdir -p "${RUN_DIR}"
echo "Writing run to ${RUN_DIR}"

# atomic_chess has chess's action space (~4674) and a 16x8x8 observation, so use
# a chess-scale ResNet. Actors do CPU self-play with GPU-batched inference via
# the evaluators, so set actors ~= available CPU cores and keep 1-2 evaluators
# bound to the GPU.
"${AZ_BIN}" \
  --game=atomic_chess \
  --path="${RUN_DIR}" \
  --nn_model=resnet \
  --nn_width=256 \
  --nn_depth=20 \
  --devices=/gpu:0 \
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
  2>&1 | tee "${RUN_DIR}/train.log"
