# Supporting harnesses

Standalone C++ drivers used to validate and analyze the OpenSpiel `atomic_chess`
environment (the RL environment this project trains on). They are not needed for
training; they document *why we trust the environment* and calibrate opponents.

| file | what it shows | result found |
|------|---------------|--------------|
| `bot_check.cc` | generic OpenSpiel bots (random, stateful-random, MCTS) run on atomic | all play legal games to terminal |
| `bot_strength.cc` | does search add strength? MCTS vs random, and more-sims vs fewer | MCTS ~99% vs random; 400-sim ~97% vs 20-sim |
| `theory_check.cc` | verifies published opening theory: after `1.Nf3`, which Black replies lose by force | 17/20 forced losses; only `d6/e5/f6` hold |
| `forced_win.cc` | alpha-beta proves forced wins on constructed positions | mate-in-1 / back-rank mate proven (+1) |
| `start_search.cc` | can alpha-beta force a win from the start? (+ the `Ng5`/`Nxf7` trap) | value 0 to depth 8 → no shallow forced win |

## Building

They link against a normal (CPU) OpenSpiel build — no LibTorch needed. The quick
way is to reuse an existing test target's link line, swapping in the harness
object. From `open_spiel/build/games`:

```bash
FLAGS="-O2 -w -std=gnu++20 -I<repo>/open_spiel/abseil-cpp \
  -I<repo>/open_spiel/json/include -I<repo>/open_spiel/.. -I<repo>/open_spiel"
clang++ $FLAGS -c harnesses/bot_strength.cc -o /tmp/h.o
LINK=$(cat games/CMakeFiles/atomic_chess_test.dir/link.txt)
LINK=${LINK/CMakeFiles\/atomic_chess_test.dir\/atomic_chess\/atomic_chess_test.cc.o//tmp/h.o}
LINK=${LINK/-o atomic_chess_test/-o /tmp/bot_strength}
eval "$LINK" && /tmp/bot_strength
```

`bot_strength.cc`, `forced_win.cc`, `start_search.cc`, `theory_check.cc` need
the `algorithms` objects (mcts, minimax, evaluate_bots) — the atomic test link
line already includes them, so the recipe above works as-is.
