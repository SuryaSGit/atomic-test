#!/usr/bin/env bash
# Verify a live training run against what we intended. Run this ~20 min after
# launch; safe to re-run any time. Read-only.
#
#   bash verify_run.sh [RUN_DIR]
set -uo pipefail

RUN_DIR="${1:-${RUN_DIR:-/u/${USER}/scratch/atomic_az/run_v2}}"
pass=0; fail=0; warn=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
note() { echo "  ....  $*"; warn=$((warn+1)); }

# JSON access. Cluster login nodes often lack jq, and silently skipping the
# config check would disable the most important verification in this script --
# so fall back to python3 and refuse to continue if neither exists.
# Override with VERIFY_JSON=python3 (or jq) if auto-detection picks a broken one.
if [[ -n "${VERIFY_JSON:-}" ]]; then
  JSON="$VERIFY_JSON"
elif command -v jq >/dev/null 2>&1; then
  JSON=jq
elif command -v python3 >/dev/null 2>&1; then
  JSON=python3
else
  echo "Need either jq or python3 to read config.json / learner.jsonl." >&2
  echo "Try: module load jq   (or)   module load python" >&2
  exit 1
fi

# cfg_get <dotted.key> <file> -> value, or empty on miss.
cfg_get() {
  if [[ $JSON == jq ]]; then
    jq -r ".$1 // empty" "$2" 2>/dev/null
  else
    python3 -c '
import json,sys
try: d=json.load(open(sys.argv[2]))
except Exception: sys.exit(0)
for k in sys.argv[1].split("."):
    if not isinstance(d,dict) or k not in d: sys.exit(0)
    d=d[k]
print(json.dumps(d) if isinstance(d,(list,dict)) else d)' "$1" "$2" 2>/dev/null
  fi
}

# learner_tail <n> <file> -> one compact summary line per step.
learner_tail() {
  if [[ $JSON == jq ]]; then
    jq -c '{step, t: (.time_rel|floor), states: .total_states,
            sps: (.states_per_s|floor), glen: .game_length.avg,
            outcomes: .outcomes.counts, eval: .eval.results,
            evalN: .eval.count, loss: .loss.sum, pol: .loss.policy,
            val: .loss.value}' "$2" 2>/dev/null | tail -"$1"
  else
    python3 -c '
import json,sys
rows=[]
for line in open(sys.argv[2]):
    line=line.strip()
    if not line: continue
    try: d=json.loads(line)
    except Exception: continue
    g=lambda *ks: (lambda c: c)(__import__("functools").reduce(
        lambda a,k: a.get(k) if isinstance(a,dict) else None, ks, d))
    rows.append(json.dumps({"step":d.get("step"),"t":int(d.get("time_rel",0)),
        "states":d.get("total_states"),"sps":int(d.get("states_per_s",0)),
        "glen":g("game_length","avg"),"outcomes":g("outcomes","counts"),
        "eval":g("eval","results"),"evalN":g("eval","count"),
        "loss":g("loss","sum"),"pol":g("loss","policy"),"val":g("loss","value")}))
print("\n".join(rows[-int(sys.argv[1]):]))' "$1" "$2" 2>/dev/null
  fi
}

echo "=== verifying ${RUN_DIR} ==="
if [[ ! -d "$RUN_DIR" ]]; then
  echo "no such run dir."
  parent=$(dirname "$RUN_DIR")
  if [[ -d "$parent" ]]; then
    echo
    echo "Run directories that DO exist under ${parent}:"
    found=0
    for d in "$parent"/*/; do
      [[ -d "$d" ]] || continue
      found=1
      printf '  %-28s %8s  %s\n' "$(basename "$d")" \
        "$(du -sh "$d" 2>/dev/null | cut -f1)" \
        "$([[ -f "$d/config.json" ]] && echo 'has config.json' || echo 'no config.json')"
    done
    [[ $found -eq 1 ]] || echo "  (none)"
    echo
    echo "If the job is writing to a different directory than expected, the"
    echo "cluster copy of atomic_az.slurm is probably older than your local one"
    echo "(RUN_DIR was changed to run_v2). Check with:"
    echo "  git -C \$(dirname \$(dirname \"$RUN_DIR\")) log --oneline -1"
    echo "  grep -n 'RUN_DIR=' \$(dirname \$(dirname \"$RUN_DIR\"))/atomic_az.slurm"
    echo "Or point this script at the directory in use:  bash verify_run.sh <dir>"
  else
    echo "Parent ${parent} does not exist either -- has the job started? squeue --me"
  fi
  exit 1
fi

# --- 1. Did it start fresh, in the right place? --------------------------------
echo
echo "[1] start mode"
if [[ -f "$RUN_DIR/train.log" ]]; then
  if grep -q "FRESH START" "$RUN_DIR/train.log"; then
    ok "started FRESH (new hyperparameters are in effect)"
  elif grep -q "RESUMING" "$RUN_DIR/train.log"; then
    bad "RESUMED -> config.json was replayed and script flag edits were IGNORED"
  fi
  if grep -q "Using existing model" "$RUN_DIR/train.log"; then
    note "reused an existing vpnet.pb (expected only for a pretrained run)"
  fi
else
  note "no train.log yet"
fi

# --- 2. Does config.json match intent? ----------------------------------------
echo
echo "[2] config.json vs intended values"
CFG="$RUN_DIR/config.json"
if [[ -s "$CFG" ]]; then
  check() { # key expected
    local got; got=$(cfg_get "$1" "$CFG")
    if [[ -z "$got" ]]; then bad "$1 missing from config.json"
    elif [[ "$got" == "$2" ]]; then ok "$1 = $got"
    else bad "$1 = $got (want $2)"; fi
  }
  check temperature_drop 6
  check replay_buffer_size 262144
  check replay_buffer_reuse 4
  check nn_width 128
  check nn_depth 10
  check eval_levels 4
  check evaluators 1
  check evaluation_window 30
  check checkpoint_freq 25
  check max_simulations 300
  echo "  ....  actors = $(cfg_get actors "$CFG"), devices = $(cfg_get devices "$CFG")"

  # Derived: states the learner must collect before each learn step.
  rbs=$(cfg_get replay_buffer_size "$CFG")
  rbr=$(cfg_get replay_buffer_reuse "$CFG")
  if [[ -n "$rbs" && -n "$rbr" && "$rbr" != 0 ]]; then
    echo "  ....  learn_rate = $(( rbs / rbr )) states/step (262144 before the fix)"
  fi
else
  note "no config.json yet"
fi

# --- 3. Has a learn step completed, and how fast? -----------------------------
echo
echo "[3] learner progress"
LJ="$RUN_DIR/learner.jsonl"
if [[ -s "$LJ" ]]; then
  TMP="${TMPDIR:-/tmp}/verify_run.$$"
  trap 'rm -f "${TMPDIR:-/tmp}"/verify_run.$$.*' EXIT
  steps=$(grep -c . "$LJ" | tr -d ' ')
  head -1 "$LJ" > "$TMP.first"
  first_t=$(cfg_get time_rel "$TMP.first")
  ok "$steps learn step(s) logged; first at t=${first_t:-?}s"
  learner_tail 5 "$LJ" | sed 's/^/  /'
  tail -1 "$LJ" > "$TMP.last"
  glen=$(cfg_get game_length.avg "$TMP.last")
  if [[ -z "$glen" ]]; then
    note "game_length.avg not present in the last record"
  elif awk "BEGIN{exit !($glen > 5 && $glen < 60)}"; then
    ok "mean self-play game length ${glen} plies (plausible for atomic)"
  else
    bad "mean game length ${glen} plies looks wrong"
  fi
else
  note "no learn step yet. Expected within ~20 min of launch; if it is much"
  note "longer, replay_buffer_size did not take effect (check section [2])."
fi

# --- 3b. Throughput diagnostics: is the GPU actually being fed? ----------------
echo
echo "[3b] throughput (baseline from run_main: 9.7 states/s, 14 actors, 24M net)"
if [[ -s "$LJ" ]]; then
  TMP2="${TMPDIR:-/tmp}/verify_run.$$"
  tail -1 "$LJ" > "$TMP2.thr"
  sps=$(cfg_get states_per_s "$TMP2.thr")
  spsa=$(cfg_get states_per_s_actor "$TMP2.thr")
  bavg=$(cfg_get batch_size.avg "$TMP2.thr")
  hit=$(cfg_get cache.hit_rate "$TMP2.thr")
  want_batch=$(cfg_get inference_batch_size "$CFG")

  if [[ -n "$sps" ]]; then
    echo "  ....  states_per_s = ${sps} (per actor: ${spsa:-?})"
    if awk "BEGIN{exit !($sps > 9.7)}"; then
      ok "faster than the old config's 9.7 states/s"
    else
      bad "no faster than the old config -- investigate before running for hours"
    fi
  fi

  # The key tuning signal. inference_batch_size is clamped to actors+evaluators
  # (alpha_zero.cc:545-547), and a mean batch far below it means actors are
  # spending their time blocked rather than queuing work -> add more actors.
  if [[ -n "$bavg" && -n "$want_batch" ]]; then
    echo "  ....  mean inference batch = ${bavg} / ${want_batch} requested"
    if awk "BEGIN{exit !($bavg < $want_batch * 0.5)}"; then
      note "batches under half full -> raise --actors (they are latency-bound,"
      note "not CPU-bound; 2x cores is the current setting)"
    else
      ok "inference batches filling well"
    fi
  fi

  if [[ -n "$hit" ]]; then
    printf '  ....  inference cache hit rate = %s\n' "$hit"
    if awk "BEGIN{exit !($hit < 0.2)}"; then
      note "low hit rate; --inference_cache may not be helping much"
    else
      ok "cache is absorbing a useful share of evaluations"
    fi
  fi
else
  note "no learner record yet"
fi

# --- 4. Are the actors actually generating games? -----------------------------
echo
echo "[4] self-play actors"
shopt -s nullglob
actors=("$RUN_DIR"/log-actor-*.txt)
if (( ${#actors[@]} )); then
  games=$(grep -h -c "^.*Game " "${actors[@]}" 2>/dev/null | paste -sd+ - | bc 2>/dev/null)
  ok "${#actors[@]} actor logs, ~${games:-?} games recorded"
else
  note "no actor logs yet"
fi

# --- 5. Is the eval ladder capped at 4 levels? --------------------------------
echo
echo "[5] eval ladder"
evals=("$RUN_DIR"/log-evaluator-*.txt)
if (( ${#evals[@]} )); then
  simset=$(grep -ho "Running MCTS with [0-9]* simulations" "${evals[@]}" \
           | awk '{print $4}' | sort -n -u | paste -sd, -)
  echo "  ....  opponent sim counts seen: ${simset}"
  if grep -qE "with (30000|94868|300000) simulations" "${evals[@]}"; then
    bad "levels 5-6 still running -> eval_levels=4 did not take effect"
  else
    ok "capped at the 4 intended levels (300/948/3000/9486)"
  fi
  wins=$(grep -ho "AZ: *[-0-9.]*" "${evals[@]}" | awk '{print $2}' \
         | awk '{n++; s+=$1} END{if(n) printf "%d games, mean return %.2f", n, s/n}')
  echo "  ....  ${wins:-no finished eval games yet}"
else
  note "no evaluator logs yet"
fi

# --- 6. Checkpoints and disk --------------------------------------------------
echo
echo "[6] checkpoints / disk"
if [[ -s "$RUN_DIR/checkpoint--1.pt" ]]; then
  sz=$(du -h "$RUN_DIR/checkpoint--1.pt" | cut -f1)
  ok "checkpoint--1.pt present (${sz}); ~75MB expected at 128x10, ~97MB at 256x20"
else
  note "no checkpoint--1.pt yet (written at the end of the first learn step)"
fi
echo "  ....  run dir total: $(du -sh "$RUN_DIR" | cut -f1)"
[[ -s "$RUN_DIR/replay_buffer.data" ]] && \
  echo "  ....  replay_buffer.data: $(du -h "$RUN_DIR/replay_buffer.data" | cut -f1) (rewritten every step)"

# --- 7. GPU actually in use? --------------------------------------------------
echo
echo "[7] GPU"
if command -v nvidia-smi >/dev/null; then
  if nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
    ok "a process is resident on the GPU"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  ....  /'
  else
    note "no compute process on the GPU (run this on the compute node, not login)"
  fi
else
  note "nvidia-smi unavailable (login node)"
fi

echo
echo "=== ${pass} pass, ${fail} fail, ${warn} pending/info ==="
[[ $fail -eq 0 ]] || echo "Fix the FAILs before letting this run for hours."
exit 0
