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

# Resume needs BOTH config.json and a non-empty learner.jsonl. config.json is
# written at startup but learner.jsonl only after the first learn step
# completes, so a preemption in between would leave us "resuming" with no
# learner record -- StartInfoFromLearnerJson reads the last line and dies in
# json::FromString("").value() (alpha_zero.cc:67-93). Requiring both makes the
# requeued job start fresh instead of crash-looping.
if [[ -f "${CONFIG}" && -s "${RUN_DIR}/learner.jsonl" ]]; then
  echo "[train] RESUMING from ${CONFIG} (latest checkpoint)"
  "${AZ_BIN}" "${CONFIG}" 2>&1 | tee -a "${RUN_DIR}/train.log"
else
  if [[ -f "${CONFIG}" ]]; then
    echo "[train] config.json exists but learner.jsonl is empty/missing:"
    echo "[train] no learn step ever completed -> starting FRESH (not resuming)."
  fi
  echo "[train] FRESH START in ${RUN_DIR}"
  # Sized for ATOMIC, not chess. Measured mean game length is ~15 plies (vs
  # 80-150 for chess), which drives every choice below.
  #
  #   temperature_drop=6   MUST stay below the mean game length. PlayGame samples
  #       moves proportional to visit counts until history.size() >=
  #       temperature_drop (alpha_zero.cc:135-139); with the old value of 20 the
  #       greedy phase never happened, so every self-play game was sampled end to
  #       end, regularly playing moves the search knew lost to an explosion mate
  #       and teaching the value head that sound positions are lost.
  #   replay_buffer_size=262144  The learner blocks until it collects
  #       replay_buffer_size/replay_buffer_reuse states per step. At ~15
  #       states/game the old 2^20 buffer meant ~17,000 games per single learn
  #       step; 2^18 gives ~4,300. It is also re-serialised to disk every step
  #       (~4.7KB/state), so this cuts several GB of scratch I/O per step too.
  #   nn_width/depth=128/10  ~6M params instead of ~24M. Net size throttles the
  #       self-play data rate, which is the scarce resource here. MultiAra used
  #       13 residual blocks for the same variants. Scale up after a plateau.
  #   eval_levels=4  Opponent sims are max_simulations*10^(level/2), so 4 levels
  #       = 300/948/3000/9486. Levels 5-6 (95k/300k rollout sims) took 3-13 min
  #       per game and ~85% of evaluator CPU for no useful signal -- and the real
  #       yardstick is Fairy-Stockfish, not a rollout bot.
  #   actors  Thread budget: actors + 1 evaluator + 1 learner + 2 inference
  #       threads = NPROC. Actors run MCTS on CPU with only batched inference on
  #       the GPU, so cores are the binding constraint on data rate, not the GPU.
  #
  # WARNING: nn_width, nn_depth and replay_buffer_size can only be changed in a
  # FRESH RUN_DIR. The first two would desynchronise vpnet.pb from the existing
  # checkpoints; the third makes LoadBuffer fatally error on a max-size mismatch
  # (serializable_circular_buffer.h:53-57).
  "${AZ_BIN}" \
    --game=atomic_chess \
    --path="${RUN_DIR}" \
    --nn_model=resnet \
    --nn_width=128 \
    --nn_depth=10 \
    --devices=/cuda:0 \
    --learning_rate=0.0002 \
    --weight_decay=0.0001 \
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
    --actors="$(( NPROC > 6 ? NPROC - 4 : 2 ))" \
    --evaluators=1 \
    --eval_levels=4 \
    --checkpoint_freq=25 \
    --evaluation_window=30 \
    --max_steps=0 \
    2>&1 | tee -a "${RUN_DIR}/train.log"
fi
