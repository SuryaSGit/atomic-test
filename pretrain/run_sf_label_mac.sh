#!/usr/bin/env bash
# Build and run the Fairy-Stockfish distillation labeller on macOS.
#
# CPU only -- no LibTorch, no GPU -- so it runs on a laptop in parallel with
# whatever the cluster is doing. Measured on an M4 (4P + 6E cores):
#   8 shards @ 10k nodes, MultiPV 4  ->  ~635 positions/s, 49 bytes/position
#   ~2.3M positions/hour   ~18M overnight (8h, ~0.9GB)   ~55M/day (~2.7GB)
# For reference, MultiAra's entire atomic dataset was 36M samples.
#
#   bash pretrain/run_sf_label_mac.sh            # ~8h, 8 shards
#   HOURS=2 SHARDS=4 bash pretrain/run_sf_label_mac.sh
set -euo pipefail

AZ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="${OPEN_SPIEL:-$HOME/open_spiel}"
OUT_DIR="${OUT_DIR:-$HOME/atomic_sf_data}"
SHARDS="${SHARDS:-8}"
HOURS="${HOURS:-8}"
NODES="${NODES:-10000}"
MULTIPV="${MULTIPV:-4}"
SF="${SF:-/opt/homebrew/bin/fairy-stockfish}"
BIN="${OUT_DIR}/sf_label"
CHECK="${OUT_DIR}/sf_data_check"

# Throughput constants measured on an M4; only used to size the run.
POS_PER_SEC_PER_SHARD=80
POS_PER_GAME=32

[[ -x "$SF" ]] || { echo "Fairy-Stockfish not found at $SF (brew install fairy-stockfish)"; exit 1; }
[[ -d "$SP/build/games" ]] || { echo "No OpenSpiel build at $SP/build (set OPEN_SPIEL=)"; exit 1; }
mkdir -p "$OUT_DIR"

# --- build both tools by reusing the atomic_chess_test link line ---------------
build() {  # build <source.cc> <output>
  local src="$1" out="$2" obj="${OUT_DIR}/$(basename "$1" .cc).o"
  local flags="-O2 -w -std=gnu++20 -I$SP/open_spiel/abseil-cpp \
    -I$SP/open_spiel/json/include -I$SP -I$SP/open_spiel"
  # shellcheck disable=SC2086
  clang++ $flags -c "$src" -o "$obj"
  local link
  link=$(cat "$SP/build/games/CMakeFiles/atomic_chess_test.dir/link.txt")
  link=${link/CMakeFiles\/atomic_chess_test.dir\/atomic_chess\/atomic_chess_test.cc.o/$obj}
  link=${link/-o atomic_chess_test/-o $out}
  ( cd "$SP/build/games" && eval "$link" )
  echo "  built $out"
}

echo "[1/3] building"
build "$AZ_DIR/pretrain/sf_label.cc" "$BIN"
build "$AZ_DIR/pretrain/sf_data_check.cc" "$CHECK"

# --- size the run -------------------------------------------------------------
SECS=$(( HOURS * 3600 ))
GAMES_PER_SHARD=$(( SECS * POS_PER_SEC_PER_SHARD / POS_PER_GAME ))
EST_POS=$(( GAMES_PER_SHARD * POS_PER_GAME * SHARDS ))
EST_MB=$(( EST_POS * 49 / 1000000 ))

cat <<EOF

[2/3] plan
  shards          : ${SHARDS}
  hours (target)  : ${HOURS}
  games per shard : ${GAMES_PER_SHARD}
  est. positions  : ${EST_POS}
  est. size       : ~${EST_MB} MB
  nodes/position  : ${NODES}   multipv: ${MULTIPV}
  output          : ${OUT_DIR}/atomic.<shard>.tsv

Laptop notes: stay plugged in; this pins ${SHARDS} cores and will run hot.
caffeinate keeps the machine awake. Ctrl-C is safe -- partial .tsv files are
still valid, every completed game is a complete line.

EOF
read -r -p "Start? [y/N] " reply
[[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted"; exit 0; }

# --- run ----------------------------------------------------------------------
echo "[3/3] labelling"
START=$(date +%s)
for i in $(seq 0 $(( SHARDS - 1 ))); do
  caffeinate -i "$BIN" \
    --sf_path="$SF" \
    --out="${OUT_DIR}/atomic" \
    --shard="$i" \
    --num_shards="$SHARDS" \
    --games="$GAMES_PER_SHARD" \
    --label_nodes="$NODES" \
    --multipv="$MULTIPV" \
    --hash_mb=64 \
    --random_plies=8 \
    --explore_prob=0.05 \
    --progress_every=500 \
    > "${OUT_DIR}/shard-$i.log" 2>&1 &
done
echo "  ${SHARDS} shards running. Watch with:"
echo "    tail -f ${OUT_DIR}/shard-0.log"
wait

ELAPSED=$(( $(date +%s) - START ))
POS=$(grep -hv '^#' "${OUT_DIR}"/atomic.*.tsv | tr ' ' '\n' | grep -c '|' || true)
echo
echo "done in ${ELAPSED}s: ${POS} labelled positions"
du -sh "$OUT_DIR"

echo
echo "validating (replays every game through our engine)..."
"$CHECK" "${OUT_DIR}"/atomic.*.tsv
