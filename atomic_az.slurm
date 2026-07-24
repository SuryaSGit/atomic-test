#!/bin/bash
# AlphaZero training on the scavenger partition (any GPU incl H200, 24h cap,
# PREEMPTIBLE). --requeue re-runs this script when preempted; train_atomic_az.sh
# is resume-aware (continues from the latest checkpoint in the stable RUN_DIR),
# so progress survives preemption. Submit: sbatch atomic_az.slurm
# One-time first:  mkdir -p /u/$USER/scratch/logs
#SBATCH --job-name=atomic_az
#SBATCH --partition=scavenger
#SBATCH --account=campusclusterusers
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/u/%u/scratch/logs/%x-%j.out
#SBATCH --error=/u/%u/scratch/logs/%x-%j.out

set -uo pipefail
echo "[slurm] job ${SLURM_JOB_ID} start/restart at $(date), restarts=${SLURM_RESTART_COUNT:-0}"
module purge && module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

SCRATCH="/u/${USER}/scratch"
export OPEN_SPIEL_BUILD="${SCRATCH}/open_spiel/build"
# STABLE run dir -> a requeued (preempted) job resumes the SAME run.
export RUN_DIR="${SCRATCH}/atomic_az/run_main"
export LD_LIBRARY_PATH="${SCRATCH}/open_spiel/open_spiel/libtorch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${RUN_DIR}" "${SCRATCH}/logs"

nvidia-smi
srun bash "${SCRATCH}/atomic_az/train_atomic_az.sh"

# Notes:
# - Preemption: SLURM requeues this job; the resume-aware train script picks up
#   the latest checkpoint. Keep checkpoint_freq modest so little is lost.
# - 24h cap: if not preempted, chain the next 24h with
#     sbatch --dependency=afterany:<jobid> atomic_az.slurm   (also resumes RUN_DIR)
# - Prefer a specific GPU if H200 queue is long: --gres=gpu:A100:1 (still scavenger).
# - No-preemption alternative (72h): swap the 3 header lines to
#     #SBATCH --partition=IllinoisComputes-GPU
#     #SBATCH --account=sridhar-ic
#     #SBATCH --gres=gpu:H200:1   (and drop --requeue)
