// Reader for the sf_label distillation dataset: replays each game and emits a
// Sample per labelled position. LibTorch-free so it can be tested against a
// plain CPU OpenSpiel build (see sf_data_test.cc).
//
// Input format, one line per game (see sf_label.cc):
//   <result_p0> \t <ply> <ply> ...
//   ply = "e2e4"                      unlabelled random opening ply
//       = "e2e4|25|e2e4:25,d2d4:20"   played | stm_cp | multipv list
//
// TWO SIGN CONVENTIONS MEET HERE, so be careful:
//   * stm_cp and the MultiPV scores are SIDE-TO-MOVE relative (UCI native).
//   * Sample::value must be PLAYER-0 relative, and player 0 is Black.
// The conversion is done once, in ValueTarget(), and asserted in the tests.

#ifndef ATOMIC_AZ_PRETRAIN_SF_DATA_H_
#define ATOMIC_AZ_PRETRAIN_SF_DATA_H_

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

#include "sample.h"

namespace atomic_az {

struct SfOptions {
  // value = lambda * tanh(cp / cp_scale) + (1 - lambda) * game_result
  // lambda=1 trusts SF's evaluation entirely, 0 uses only the outcome. The
  // Fairy-Stockfish NNUE pipeline exposes the same knob; 0.7 is a common
  // starting point. Blending matters because pure search scores inherit the
  // teacher's biases while pure outcomes are noisy.
  double lambda = 0.7;
  // Centipawns mapped to |value| ~ 0.76. Atomic swings are larger than chess,
  // so calibrate with `sf_data_check --calibrate` rather than assuming 400.
  double cp_scale = 400.0;
  // Softmax temperature over MultiPV scores, in centipawns. 0 => one-hot on
  // SF's best move. Soft targets carry more information per position.
  double policy_temp = 0.0;
  // Drop positions whose |cp| exceeds this (0 = keep all). Mate-scored
  // positions are trivially won; keeping them teaches mates, dropping them
  // focuses capacity on unresolved play.
  int max_abs_cp = 0;
};

struct SfStats {
  int64_t lines = 0, games_ok = 0, games_bad = 0;
  int64_t samples = 0, skipped_cp = 0, opening_plies = 0;

  std::string ToString() const {
    return "lines=" + std::to_string(lines) + " games_ok=" +
           std::to_string(games_ok) + " games_bad=" + std::to_string(games_bad) +
           " samples=" + std::to_string(samples) + " opening_plies=" +
           std::to_string(opening_plies) + " skipped_cp=" +
           std::to_string(skipped_cp);
  }
};

namespace sf_internal {

inline std::vector<std::string> Split(const std::string& s, char d) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t p = s.find(d, start);
    if (p == std::string::npos) {
      if (start < s.size()) out.push_back(s.substr(start));
      break;
    }
    if (p > start) out.push_back(s.substr(start, p - start));
    start = p + 1;
  }
  return out;
}

// Parses a UCI move AND verifies the resulting action is actually legal here.
// The membership check is not optional: ChessBoard::ParseMove can hand back a
// Move for something the atomic rules reject, and ApplyAction on a non-legal
// action segfaults rather than raising. A corrupt dataset line must be
// rejected, never crash the trainer.
inline open_spiel::Action UciToAction(
    const open_spiel::atomic_chess::AtomicChessState& s,
    const std::vector<open_spiel::Action>& legal, const std::string& uci) {
  auto m = s.Board().ParseMove(uci, /*chess960=*/false);
  if (!m.has_value()) return open_spiel::kInvalidAction;
  const open_spiel::Action a =
      open_spiel::chess::MoveToAction(*m, s.BoardSize());
  for (open_spiel::Action l : legal) {
    if (l == a) return a;
  }
  return open_spiel::kInvalidAction;
}

}  // namespace sf_internal

// Maps a side-to-move centipawn score plus the game result onto the
// player-0-relative value the network is trained to predict.
inline double ValueTarget(int stm_cp, double result_p0,
                          open_spiel::Player mover, const SfOptions& opt) {
  const double stm_value = std::tanh(static_cast<double>(stm_cp) /
                                     opt.cp_scale);
  // stm_value is from the mover's point of view; flip it when the mover is not
  // player 0.
  const double search_p0 = (mover == 0) ? stm_value : -stm_value;
  return opt.lambda * search_p0 + (1.0 - opt.lambda) * result_p0;
}

// Softmax over MultiPV centipawn scores, or one-hot when policy_temp <= 0.
inline open_spiel::ActionsAndProbs PolicyTarget(
    const std::vector<std::pair<open_spiel::Action, int>>& moves,
    const SfOptions& opt) {
  open_spiel::ActionsAndProbs out;
  if (moves.empty()) return out;
  if (opt.policy_temp <= 0.0) {
    return OneHot(moves[0].first);  // sf_label writes best-first
  }
  double best = static_cast<double>(moves[0].second);
  double sum = 0.0;
  std::vector<double> w;
  w.reserve(moves.size());
  for (const auto& [a, cp] : moves) {
    // Subtract the max for numerical stability.
    const double e = std::exp((static_cast<double>(cp) - best) /
                              opt.policy_temp);
    w.push_back(e);
    sum += e;
  }
  for (size_t i = 0; i < moves.size(); ++i) {
    out.emplace_back(moves[i].first, w[i] / sum);
  }
  return out;
}

// Replays one dataset line. Emits nothing if any move fails to parse, so a
// malformed game never contributes positions labelled with a result it did not
// reach.
inline bool SfLineToSamples(const open_spiel::Game& game,
                            const std::string& line, const SfOptions& opt,
                            const std::function<void(Sample&&)>& on_sample,
                            SfStats* stats) {
  const size_t tab = line.find('\t');
  if (tab == std::string::npos) return false;
  const double result_p0 = std::atof(line.substr(0, tab).c_str());
  const std::vector<std::string> plies =
      sf_internal::Split(line.substr(tab + 1), ' ');
  if (plies.empty()) return false;

  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<Sample> pending;
  int64_t skipped = 0, openings = 0;

  for (const std::string& tok : plies) {
    if (state->IsTerminal()) return false;  // moves past the end of the game
    const auto& as =
        open_spiel::down_cast<open_spiel::atomic_chess::AtomicChessState&>(
            *state);
    const std::vector<open_spiel::Action> legal = state->LegalActions();
    const std::vector<std::string> parts = sf_internal::Split(tok, '|');

    if (parts.size() == 1) {  // unlabelled opening ply: advance only
      const open_spiel::Action a =
          sf_internal::UciToAction(as, legal, parts[0]);
      if (a == open_spiel::kInvalidAction) return false;
      ++openings;
      state->ApplyAction(a);
      continue;
    }
    if (parts.size() != 3) return false;

    const open_spiel::Action played =
        sf_internal::UciToAction(as, legal, parts[0]);
    if (played == open_spiel::kInvalidAction) return false;
    const int stm_cp = std::atoi(parts[1].c_str());

    if (opt.max_abs_cp > 0 && std::abs(stm_cp) > opt.max_abs_cp) {
      ++skipped;
      state->ApplyAction(played);
      continue;
    }

    // MultiPV list -> legal actions with their scores, best first.
    std::vector<std::pair<open_spiel::Action, int>> scored;
    for (const std::string& alt : sf_internal::Split(parts[2], ',')) {
      const size_t colon = alt.rfind(':');
      if (colon == std::string::npos) continue;
      const open_spiel::Action a =
          sf_internal::UciToAction(as, legal, alt.substr(0, colon));
      if (a == open_spiel::kInvalidAction) continue;
      scored.emplace_back(a, std::atoi(alt.substr(colon + 1).c_str()));
    }
    if (scored.empty()) return false;

    Sample s;
    s.legal_actions = legal;
    s.observation = state->ObservationTensor();
    s.policy = PolicyTarget(scored, opt);
    // The label describes THIS position, so the mover is the current player --
    // not the player who happens to have moved last.
    s.value = ValueTarget(stm_cp, result_p0, state->CurrentPlayer(), opt);
    pending.push_back(std::move(s));

    state->ApplyAction(played);  // may differ from scored[0] when exploring
  }

  if (!state->IsTerminal()) return false;
  if (stats) {
    stats->skipped_cp += skipped;
    stats->opening_plies += openings;
    stats->samples += pending.size();
  }
  for (Sample& s : pending) on_sample(std::move(s));
  return true;
}

// Streams a whole .tsv file.
inline SfStats ReadSfTsv(std::istream& in, const open_spiel::Game& game,
                         const SfOptions& opt,
                         const std::function<void(Sample&&)>& on_sample,
                         int64_t max_games = -1) {
  SfStats stats;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    ++stats.lines;
    if (SfLineToSamples(game, line, opt, on_sample, &stats)) {
      ++stats.games_ok;
    } else {
      ++stats.games_bad;
    }
    if (max_games >= 0 && stats.games_ok >= max_games) break;
  }
  return stats;
}

}  // namespace atomic_az

#endif  // ATOMIC_AZ_PRETRAIN_SF_DATA_H_
