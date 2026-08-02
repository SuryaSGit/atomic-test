// Shared match-running and scoring logic for the evaluation harnesses.
//
// Used by az_vs_sf.cc (needs LibTorch) and sf_bridge.cc (does not), so the
// scoring, pairing and reporting can be exercised on any machine by running
// sf_bridge -- the part that is easy to get subtly wrong is tested even where
// the neural net cannot be loaded.
//
// The choices here exist because the obvious implementation produces numbers
// that look reasonable and mean nothing:
//   * per-colour scores, because atomic is strongly first-move-favoured
//   * colour-swapped opening pairs, so opening luck cancels
//   * unfinished games excluded, never counted as draws
//   * ucinewgame between games
//   * confidence intervals on every figure

#ifndef ATOMIC_AZ_EVAL_MATCH_H_
#define ATOMIC_AZ_EVAL_MATCH_H_

#include <cmath>
#include <functional>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

namespace atomic_az {

struct Tally {
  int win = 0, draw = 0, loss = 0;
  int n() const { return win + draw + loss; }
  double score() const { return n() == 0 ? 0.0 : (win + 0.5 * draw) / n(); }

  // Standard error over the per-game scores {1, 0.5, 0}. The plain binomial
  // formula is wrong once draws exist.
  double stderr_() const {
    const int k = n();
    if (k < 2) return 0.0;
    const double m = score();
    const double ss = win * (1 - m) * (1 - m) + draw * (0.5 - m) * (0.5 - m) +
                      loss * m * m;
    return std::sqrt(ss / (k - 1) / k);
  }

  std::string Report() const {
    if (n() == 0) return "no games";
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%5.1f%% +/- %4.1f  (%dW %dD %dL, n=%d)",
                  100.0 * score(), 100.0 * 1.96 * stderr_(), win, draw, loss,
                  n());
    return std::string(buf);
  }
};

struct MatchResult {
  Tally as_white, as_black;
  int unfinished = 0, sf_failures = 0;

  Tally overall() const {
    Tally t;
    t.win = as_white.win + as_black.win;
    t.draw = as_white.draw + as_black.draw;
    t.loss = as_white.loss + as_black.loss;
    return t;
  }
};

// Called at each position where OUR bot is to move, with the state and the
// player id we are playing. Lets a caller measure the network on the positions
// it actually reaches -- which is not the distribution it was trained on.
using PositionHook =
    std::function<void(const open_spiel::State&, open_spiel::Player)>;

// Called once per finished game with the final result for player 0, so a hook
// can score its recorded positions against the outcome.
using ResultHook = std::function<void(double result_p0, bool finished)>;

struct MatchConfig {
  int pairs = 20;          // each pair = the same opening played both colours
  int opening_plies = 4;   // random plies for opening diversity
  int max_plies = 400;
  int seed = 1;
  bool verbose = false;
  std::string go_cmd = "go nodes 100000";
  // Optional: dump each game's move sequence (UCI, space separated) so the
  // positions can later be relabelled by Stockfish for an iteration round.
  std::string dump_games_path;
};

// Asks Fairy-Stockfish for a move and translates it through OUR board, so we
// never apply something the atomic rules reject. kInvalidAction on failure.
inline open_spiel::Action StockfishMove(
    open_spiel::uci::UCIBot* sf,
    const open_spiel::atomic_chess::AtomicChessState& s,
    const std::string& go_cmd) {
  sf->Position(s.Board().ToFEN());
  sf->Write(go_cmd);
  std::string best;
  while (true) {
    const std::string line = sf->ReadLine();
    if (line.rfind("bestmove", 0) == 0) {
      std::istringstream iss(line);
      std::string tok;
      iss >> tok >> best;
      break;
    }
  }
  if (best.empty() || best == "(none)" || best == "0000")
    return open_spiel::kInvalidAction;
  auto m = s.Board().ParseMove(best, /*chess960=*/false);
  if (!m.has_value()) return open_spiel::kInvalidAction;
  const open_spiel::Action a =
      open_spiel::chess::MoveToAction(*m, s.BoardSize());
  // ParseMove can return a Move the atomic rules reject; ApplyAction on a
  // non-legal action segfaults rather than raising.
  for (open_spiel::Action l : s.LegalActions()) {
    if (l == a) return a;
  }
  return open_spiel::kInvalidAction;
}

// Runs the match. `make_bot(seed)` supplies our side; the opponent is `sf`.
inline MatchResult RunMatch(
    const open_spiel::Game& game, open_spiel::uci::UCIBot* sf,
    const std::function<std::unique_ptr<open_spiel::Bot>(int)>& make_bot,
    const MatchConfig& cfg, const PositionHook& on_our_position = nullptr,
    const ResultHook& on_result = nullptr) {
  std::unique_ptr<std::ofstream> dump;
  if (!cfg.dump_games_path.empty()) {
    dump = std::make_unique<std::ofstream>(cfg.dump_games_path);
    *dump << "# games played by our bot; UCI moves, one game per line\n";
  }
  const open_spiel::Player kWhite = game.NewInitialState()->CurrentPlayer();
  MatchResult res;
  std::mt19937 rng(cfg.seed);

  for (int p = 0; p < cfg.pairs; ++p) {
    // One opening per pair, shared by both colour assignments.
    std::vector<open_spiel::Action> opening;
    {
      auto s = game.NewInitialState();
      for (int i = 0; i < cfg.opening_plies && !s->IsTerminal(); ++i) {
        auto legal = s->LegalActions();
        const open_spiel::Action a = legal[
            std::uniform_int_distribution<size_t>(0, legal.size() - 1)(rng)];
        opening.push_back(a);
        s->ApplyAction(a);
      }
    }

    for (int side = 0; side < 2; ++side) {
      const open_spiel::Player our_seat = (side == 0) ? kWhite : 1 - kWhite;
      sf->Restart();  // ucinewgame: no hash or history across games
      auto bot = make_bot(cfg.seed + p * 2 + side);

      auto s = game.NewInitialState();
      for (open_spiel::Action a : opening) s->ApplyAction(a);

      int ply = static_cast<int>(opening.size());
      bool broke = false;
      std::vector<std::string> history;
      std::vector<std::string> uci_moves;
      if (dump) {
        // Replay the opening so the dumped line starts from the initial
        // position, matching the sf_label dataset format.
        auto tmp = game.NewInitialState();
        for (open_spiel::Action a : opening) {
          const auto& tas = open_spiel::down_cast<
              open_spiel::atomic_chess::AtomicChessState&>(*tmp);
          uci_moves.push_back(
              open_spiel::chess::ActionToMove(a, tas.Board()).ToLAN());
          tmp->ApplyAction(a);
        }
      }
      while (!s->IsTerminal() && ply < cfg.max_plies) {
        const auto& as = open_spiel::down_cast<
            open_spiel::atomic_chess::AtomicChessState&>(*s);
        const open_spiel::Player cur = s->CurrentPlayer();
        open_spiel::Action a;
        if (cur == our_seat) {
          if (on_our_position) on_our_position(*s, our_seat);
          a = bot->Step(*s);
          // Our side is normally an MCTS bot that cannot fail, but it may also
          // be a UCI engine (uci_vs_uci). ApplyAction on kInvalidAction
          // segfaults rather than raising, so check here too.
          if (a == open_spiel::kInvalidAction) {
            std::cerr << "  !! our bot returned no usable move at "
                      << as.Board().ToFEN() << std::endl;
            res.sf_failures += 1;
            broke = true;
            break;
          }
        } else {
          a = StockfishMove(sf, as, cfg.go_cmd);
          if (a == open_spiel::kInvalidAction) {
            std::cerr << "  !! no usable SF move at " << as.Board().ToFEN()
                      << std::endl;
            res.sf_failures += 1;
            broke = true;
            break;
          }
        }
        if (cfg.verbose) history.push_back(s->ActionToString(cur, a));
        if (dump) {
          uci_moves.push_back(
              open_spiel::chess::ActionToMove(a, as.Board()).ToLAN());
        }
        s->ApplyAction(a);
        ++ply;
      }

      const bool finished = !broke && s->IsTerminal();
      if (on_result) {
        on_result(finished ? s->Returns()[0] : 0.0, finished);
      }
      if (dump && finished) {
        *dump << s->Returns()[0] << "\t";
        for (size_t i = 0; i < uci_moves.size(); ++i) {
          if (i) *dump << " ";
          *dump << uci_moves[i];
        }
        *dump << "\n";
      }

      std::string label;
      if (broke || !s->IsTerminal()) {
        res.unfinished += 1;
        label = "UNFINISHED";
      } else {
        const double r = s->Returns()[our_seat];
        Tally& t = (our_seat == kWhite) ? res.as_white : res.as_black;
        if (r > 0)       { t.win++;  label = "WIN "; }
        else if (r == 0) { t.draw++; label = "draw"; }
        else             { t.loss++; label = "loss"; }
      }
      std::cout << "pair " << p << " us as "
                << (our_seat == kWhite ? "White" : "Black") << ": " << label
                << "  plies=" << ply << std::endl;
      if (cfg.verbose && !history.empty()) {
        std::cout << "    ";
        for (const auto& m : history) std::cout << m << " ";
        std::cout << std::endl;
      }
    }
  }
  return res;
}

inline void PrintResult(const MatchResult& res, const std::string& header) {
  const Tally all = res.overall();
  std::cout << "\n================ " << header << " ================\n"
            << "as White : " << res.as_white.Report() << "\n"
            << "as Black : " << res.as_black.Report() << "\n"
            << "overall  : " << all.Report() << std::endl;
  if (res.unfinished > 0) {
    std::cout << "unfinished (EXCLUDED from the score): " << res.unfinished;
    if (res.sf_failures > 0)
      std::cout << ", of which SF move failures: " << res.sf_failures;
    std::cout << std::endl;
  }
  if (res.as_white.n() > 0 && res.as_black.n() > 0) {
    const double gap = res.as_white.score() - res.as_black.score();
    // 0.15, not 0.25: runs scoring 23.3/0.0 and 62.1/41.0 both stayed silent
    // under the old threshold, and the first of those had a colour that never
    // won a single game.
    if (std::fabs(gap) > 0.15) {
      std::cout << "NOTE: colour gap of " << (100.0 * gap)
                << "pp -- the overall figure averages two very different "
                   "results and is not a single strength." << std::endl;
    }
    // A shut-out is worth flagging at any gap: it usually means one colour is
    // broken rather than weak.
    if ((res.as_white.win == 0 && res.as_white.n() >= 20) ||
        (res.as_black.win == 0 && res.as_black.n() >= 20)) {
      std::cout << "NOTE: one colour did not win a single game -- treat the "
                   "aggregate as meaningless until that is explained."
                << std::endl;
    }
  }
  if (all.n() < 100 && all.n() > 0) {
    std::cout << "NOTE: n=" << all.n()
              << " is small; the interval is wide. Several hundred games are "
                 "needed to separate nearby strengths." << std::endl;
  }
}

}  // namespace atomic_az

#endif  // ATOMIC_AZ_EVAL_MATCH_H_
