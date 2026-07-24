#!/usr/bin/env bash
# CPU-only pipeline smoke test -- no GPU allocation needed. Proves
# self-play -> train -> checkpoint -> play works on atomic_chess.
#
# Ways to run it:
#   (1) Interactive CPU node (recommended on ICC):
#         srun --partition=secondary --account=campusclusterusers \
#              --cpus-per-task=8 --mem=16G --time=00:20:00 --pty bash
#         bash /u/$USER/scratch/atomic_az/smoke_cpu.sh
#   (2) Quick check on a login node (tiny + short only -- be polite):
#         bash /u/$USER/scratch/atomic_az/smoke_cpu.sh
#   (3) Locally on your Mac IF you build OpenSpiel with CPU libtorch first
#       (OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON with an arm64 CPU libtorch), then
#       set BUILD to your local build dir.
set -uo pipefail

SCRATCH="/u/${USER}/scratch"
BUILD="${OPEN_SPIEL_BUILD:-$SCRATCH/open_spiel/build}"
RUN="${RUN_DIR:-$SCRATCH/atomic_az/smoke_cpu}"
export LD_LIBRARY_PATH="$(dirname "${BUILD}")/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN}"

echo ">>> tiny CPU training (a few steps) on atomic_chess ..."
"${BUILD}/examples/alpha_zero_torch_example" \
  --game=atomic_chess --path="${RUN}" \
  --nn_model=resnet --nn_width=32 --nn_depth=2 \
  --devices=/cpu:0 --actors=4 --evaluators=1 \
  --max_simulations=25 --train_batch_size=128 \
  --inference_threads=1 --inference_batch_size=1 \
  --replay_buffer_size=4096 --checkpoint_freq=5 \
  --max_steps=15

echo ">>> does the checkpoint load and play? (AZ vs random, on CPU) ..."
"${BUILD}/examples/alpha_zero_torch_game_example" \
  --game=atomic_chess --az_path="${RUN}" --az_checkpoint=-1 \
  --player1=az --player2=random --max_simulations=25

echo ">>> SMOKE(CPU) OK if training logged steps and a game finished with returns."
