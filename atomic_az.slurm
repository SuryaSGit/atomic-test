#!/bin/bash
# Main AlphaZero training run on the Illinois Campus Cluster.
# Uses IllinoisComputes-GPU (H200, 72h, NO preemption) -- best fit for a long
# AZ self-play run. Submit with:  sbatch atomic_az.slurm
# NOTE: create the log dir once first:  mkdir -p /u/$USER/scratch/logs
#SBATCH --job-name=atomic_az
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=sridhar-ic
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/u/%u/scratch/logs/%x-%j.out
#SBATCH --error=/u/%u/scratch/logs/%x-%j.out

set -uo pipefail
module purge && module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

SCRATCH="/u/${USER}/scratch"
export OPEN_SPIEL_BUILD="${SCRATCH}/open_spiel/build"
# STABLE run dir so a chained/re-submitted job continues the same run.
export RUN_DIR="${SCRATCH}/atomic_az/run_main"
# LibTorch shared libs must be findable at runtime.
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_DIR}" "${SCRATCH}/logs"

nvidia-smi
srun bash "${SCRATCH}/atomic_az/train_atomic_az.sh"

# For runs longer than 72h, chain jobs (each resumes RUN_DIR if AZ supports it):
#   sbatch --dependency=afterany:<jobid> atomic_az.slurm
#
# Alternative partitions (swap the 3 header lines):
#   scavenger  (any GPU incl H200, 24h, PREEMPTIBLE -> add "#SBATCH --requeue"):
#     #SBATCH --partition=scavenger
#     #SBATCH --account=campusclusterusers
#     #SBATCH --gres=gpu:H200:1
#     #SBATCH --requeue
#   secondary  (<=80GB GPU, 4h, fast queue -- use gpu:H100:1, good for eval/smoke):
#     #SBATCH --partition=secondary
#     #SBATCH --account=campusclusterusers
#     #SBATCH --gres=gpu:H100:1
