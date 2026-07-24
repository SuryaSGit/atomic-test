# AlphaZero vs Stockfish on Atomic Chess

Goal: train an AlphaZero-style agent on the OpenSpiel `atomic_chess` environment
(GPU cluster) and measure it against **Fairy-Stockfish** (the reference atomic
engine). This folder has the training scripts, a SLURM job, and two evaluation
harnesses.

## Reality check (read first)

- **Standard Stockfish cannot play atomic.** The opponent is **Fairy-Stockfish**
  driven over UCI with `setoption name UCI_Variant value atomic`. It is *very*
  strong at atomic.
- **Beating full-strength Fairy-Stockfish is a stretch goal**, even with GPU
  training. AlphaZero reached superhuman variant play on large TPU fleets; a
  single-GPU run should target the *strength ladder* below, not an instant win.
- The OpenSpiel C++ LibTorch AlphaZero is a **community contribution** and isn't
  covered by core CI (see `open_spiel/algorithms/alpha_zero_torch/README.md`).
  Expect to debug the build once.

## Measured baselines (local, CPU, this machine)

Established with `eval/sf_bridge.cc` (a plain random-rollout MCTS vs Fairy-Stockfish):

| Opponent                         | Our bot            | Score  |
|----------------------------------|--------------------|--------|
| Fairy-Stockfish, full, 50 ms     | MCTS 300 sims      | ~0%    |
| Fairy-Stockfish, **skill 0**, 30 ms | MCTS 800 sims  | ~17% (1W/6) |

So the **strength ladder** to track during training:
random → beat skill-0 SF → climb skill levels / movetime → approach full strength.

## ICC (Illinois Campus Cluster) essentials

- **Everything on scratch** — `$HOME` is tiny. Put the repo, libtorch, venvs,
  and all outputs under `/u/$USER/scratch/`. These scripts assume
  `SCRATCH=/u/$USER/scratch` and `open_spiel` cloned to `$SCRATCH/open_spiel`.
- Login node (`ssh <netid>@cc-login1.campuscluster.illinois.edu`, VPN off-campus)
  has **no GPU** — build & submit there; run on nodes.
- Partition choice for this workload:
  - **`scavenger` / `campusclusterusers` — any GPU incl H200, 24h, preemptible**
    → the main training run (`atomic_az.slurm`). Safe because `--requeue` +
    resume-aware training continue from the latest checkpoint after preemption.
  - `IllinoisComputes-GPU` / `sridhar-ic` — H200, 72h, no preemption →
    no-preemption alternative (swap the 3 header lines, drop `--requeue`).
  - `secondary` / `campusclusterusers` — ≤80GB GPU, 4h, fast → smoke test & eval
    (`smoke_test.slurm`).
- One-time: `mkdir -p /u/$USER/scratch/logs` (SLURM won't create log dirs).

## 1. Build OpenSpiel with CUDA LibTorch (on a login node)

```bash
cd /u/$USER/scratch
git clone https://github.com/SuryaSGit/open_spiel.git
cd open_spiel && git checkout add-atomic-chess
module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

# CRITICAL: OpenSpiel reads these from the ENVIRONMENT, not from -D flags
# (its CMake does `set(VAR $ENV{VAR})`, which overrides any -DVAR=ON). Export
# them before BOTH install.sh and cmake, or the AZ target is silently skipped.
export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON
export OPEN_SPIEL_BUILD_WITH_LIBTORCH_DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip"

./install.sh                        # provisions open_spiel/open_spiel/libtorch/libtorch + libnop
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ../open_spiel   # MUST print "OPEN_SPIEL_BUILD_WITH_LIBTORCH: ON" and find Torch
make -j"$(nproc)" alpha_zero_torch_example alpha_zero_torch_game_example
```

**Shortcut:** `bash cluster_setup.sh` does all of the above (export env, run
install.sh, create the no-op wrappers below, configure, build).

Two ICC gotchas this handles:
- `install.sh` **exits nonzero** on ICC (it assumes Debian/`apt-get`); that's
  benign — deps are already fetched and OS packages come from `module`s.
- The top CMakeLists does `add_subdirectory(libnop)` / `add_subdirectory(libtorch)`,
  but those wrapper `CMakeLists.txt` are **not committed anywhere and install.sh
  never writes them**. Create no-op ones (the parent CMakeLists does the real
  include/`find_package` work, and nothing depends on targets from these dirs):
  ```bash
  printf '# header-only; include dir set by parent\n'      > open_spiel/libnop/CMakeLists.txt
  printf '# libtorch via find_package(Torch) in parent\n'  > open_spiel/libtorch/CMakeLists.txt
  ```

If cmake does *not* print `OPEN_SPIEL_BUILD_WITH_LIBTORCH: ON`, the env var
wasn't exported in that shell — fix and re-run in a **fresh** `build/` dir. If
`find_package(Torch)` errors, libtorch isn't at
`open_spiel/open_spiel/libtorch/libtorch/` — re-run `./install.sh` with the env
vars exported. Match the libtorch build to `cuda/12.8` (cu121/cu124 cxx11-ABI).
The `atomic_chess` game is already on the `add-atomic-chess` branch.

Also build Fairy-Stockfish once (for evaluation):
```bash
cd /u/$USER/scratch && git clone https://github.com/fairy-stockfish/Fairy-Stockfish.git
cd Fairy-Stockfish/src && make -j build ARCH=x86-64-bmi2   # binary: ./stockfish
```

## 2. Smoke test first (fast queue, ~minutes)

```bash
sbatch /u/$USER/scratch/atomic_az/smoke_test.slurm   # trains a tiny net, plays it
```
Confirm it trains, checkpoints, and the checkpoint plays vs random before
spending H200 hours.

## 3. Train (main run — scavenger, preemptible)

```bash
sbatch /u/$USER/scratch/atomic_az/atomic_az.slurm
```
Runs on **scavenger** (any GPU incl H200, 24h, preemptible). Preemption is safe:
`--requeue` re-runs the job and `train_atomic_az.sh` **resumes from the latest
checkpoint** in the stable `RUN_DIR=$SCRATCH/atomic_az/run_main` — the torch-AZ
example resumes when handed a positional `config.json`, and starts fresh only
when given `--flags`. Keep `checkpoint_freq` modest so little is lost per
preemption.

Chess-scale hyperparameters (ResNet 256×20, 300 sims/move, replay 2^20,
`--devices=/gpu:0`, `actors≈#cores`). Checkpoints (`checkpoint-<step>`) land in
`RUN_DIR`. After a clean (non-preempted) 24h, chain the next window with
`sbatch --dependency=afterany:<jobid> atomic_az.slurm` (also resumes).
No-preemption alternative: `IllinoisComputes-GPU`/`sridhar-ic` (72h) per the
`.slurm` header comments, dropping `--requeue`. Watch progress:

```bash
squeue --me
tail -f /u/$USER/scratch/logs/atomic_az-<jobid>.out    # value/policy loss + eval window
```

## 4. Evaluate

**Sanity (no Stockfish needed)** — is the checkpoint actually playing?
Use the stock example to pit AZ vs random/mcts:
```bash
./examples/alpha_zero_torch_game_example --game=atomic_chess \
  --az_path=$RUN_DIR --az_checkpoint=-1 --player1=az --player2=mcts
```

**Vs Fairy-Stockfish** — `eval/az_vs_sf.cc`. It needs LibTorch, so
build it *inside* the OpenSpiel tree:
```bash
cp eval/az_vs_sf.cc open_spiel/examples/
# add to open_spiel/examples/CMakeLists.txt, guarded like the other torch examples:
#   if (${OPEN_SPIEL_BUILD_WITH_LIBTORCH})
#     add_executable(az_vs_sf az_vs_sf.cc ${OPEN_SPIEL_OBJECTS})
#     target_link_libraries(az_vs_sf ${TORCH_LIBRARIES})
#   endif()
make -j az_vs_sf
./examples/az_vs_sf --az_path=/u/$USER/scratch/atomic_az/run_main --az_checkpoint=-1 \
    --sf_path=/u/$USER/scratch/Fairy-Stockfish/src/stockfish \
    --games=40 --sf_skill=0 --sf_movetime=100 --device=/gpu:0
```
Run eval on the **secondary** queue (short, 1 GPU). Climb the ladder as the
agent improves: `--sf_skill` 0→20, then raise `--sf_movetime`. Milestones:
beat random → beat skill-0 SF (our MCTS baseline ~17%) → higher skills → full.

`eval/sf_bridge.cc` is the **tested** MCTS-vs-Fairy-Stockfish harness (no LibTorch
needed) — useful for calibrating Skill Level / movetime on any machine.

## Notes / gotchas

- **Move translation is exact:** both harnesses feed Fairy-Stockfish our board's
  FEN and parse its UCI reply with our engine's `ParseMove`, so there's no
  second atomic implementation to disagree with. (Full games already played
  cleanly between our engine and Fairy-Stockfish — good cross-validation that
  our rules match the reference.)
- Fairy-Stockfish atomic == lichess ruleset (`atomic`, not `nocheckatomic`),
  which matches this implementation.
- If the torch build hits the known pybind11/torch conflict, see the workaround
  linked in the torch-AZ README.
- CPU-only fallback works for a smoke test (`--devices=/cpu:0`, tiny net) just to
  prove the pipeline end-to-end before committing GPU hours.
