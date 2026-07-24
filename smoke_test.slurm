#!/bin/bash
# Cheap end-to-end pipeline validation on the fast queue: train a TINY net for a
# few minutes, then confirm the checkpoint loads and plays. Run this BEFORE
# committing H200 hours. Submit:  sbatch smoke_test.slurm
# First time only:  mkdir -p /u/$USER/scratch/logs
#SBATCH --job-name=atomic_az_smoke
#SBATCH --partition=secondary
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/u/%u/scratch/logs/%x-%j.out
#SBATCH --error=/u/%u/scratch/logs/%x-%j.out

set -uo pipefail
module purge && module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
SCRATCH="/u/${USER}/scratch"
BUILD="${SCRATCH}/open_spiel/build"
RUN="${SCRATCH}/atomic_az/smoke"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN}" "${SCRATCH}/logs"
nvidia-smi

# Tiny + short: just prove self-play -> train -> checkpoint works on atomic.
# Small train_batch_size so the FIRST learn step logs to stdout within seconds
# (AZ is silent on stdout until the buffer reaches train_batch_size states;
# per-game self-play progress goes to log-actor-*.txt in the run dir).
"${BUILD}/examples/alpha_zero_torch_example" \
  --game=atomic_chess --path="${RUN}" \
  --nn_model=resnet --nn_width=64 --nn_depth=4 \
  --devices=/cuda:0 --actors=6 --evaluators=1 \
  --max_simulations=50 --train_batch_size=32 \
  --replay_buffer_size=2048 --checkpoint_freq=5 \
  --max_steps=20

echo "=== checkpoint plays? (AZ vs random, 2 games) ==="
"${BUILD}/examples/alpha_zero_torch_game_example" \
  --game=atomic_chess --az_path="${RUN}" --az_checkpoint=-1 \
  --player1=az --player2=random --max_simulations=50

echo "SMOKE OK if you see completed games with returns above."
