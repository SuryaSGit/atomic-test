#!/usr/bin/env bash
# One-shot OpenSpiel + LibTorch build for ICC (idempotent). Run on a login node
# or an interactive shell from inside the OpenSpiel checkout's parent, e.g.:
#   bash /u/$USER/scratch/atomic_az/cluster_setup.sh
#
# Handles the two ICC/OpenSpiel gotchas:
#  1) OpenSpiel reads OPEN_SPIEL_BUILD_WITH_* from the ENVIRONMENT (not -D).
#  2) The top CMakeLists does add_subdirectory(libnop)/(libtorch) but those
#     wrapper CMakeLists.txt are neither committed nor written by install.sh,
#     so we create no-op ones (the parent CMakeLists does the real work).
set -uo pipefail

SCRATCH="/u/${USER}/scratch"
OS_DIR="${OPEN_SPIEL_DIR:-$SCRATCH/open_spiel}"
LIBTORCH_URL="${OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL:-https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip}"

module load cuda/12.8 || echo "[setup] (module load cuda/12.8 not available; continuing)"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON
export OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL="$LIBTORCH_URL"

cd "$OS_DIR"

# 1) Provision deps. install.sh exits nonzero on non-Debian (OS packages come
#    from modules here) AFTER cloning/downloading -- that's expected/benign.
./install.sh || echo "[setup] install.sh returned nonzero (expected on ICC: skip OS-package step)"

# 2) Sanity: libtorch present?
if [[ ! -f open_spiel/libtorch/libtorch/lib/libtorch.so ]]; then
  echo "[setup] ERROR: libtorch not found at open_spiel/libtorch/libtorch/lib."
  echo "        Check the download URL / re-run install.sh. URL=$LIBTORCH_URL"
  exit 1
fi

# 3) No-op wrapper CMakeLists that add_subdirectory() requires.
[[ -f open_spiel/libnop/CMakeLists.txt ]]  || printf '# libnop is header-only; include dir set by the parent CMakeLists.\n'   > open_spiel/libnop/CMakeLists.txt
[[ -f open_spiel/libtorch/CMakeLists.txt ]] || printf '# libtorch provided via find_package(Torch) in the parent CMakeLists.\n' > open_spiel/libtorch/CMakeLists.txt

# 4) Configure (fresh) + build only the AZ-torch binaries.
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../open_spiel
make -j"$(nproc)" alpha_zero_torch_example alpha_zero_torch_game_example

echo "[setup] DONE."
ls -l examples/alpha_zero_torch_example examples/alpha_zero_torch_game_example
