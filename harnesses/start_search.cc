// Can alpha-beta prove a forced mate from the ATOMIC START POSITION?
// Expectation: no -- the tree is astronomically large, non-terminal leaves
// evaluate to 0, and there is no known shallow forced win. We iterative-deepen
// (with timing) to show the value stays 0 while the cost explodes, and repeat
// after a few "theory hint" opening moves.
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

using namespace open_spiel;

static std::function<double(const State&)> kZeroEval =
    [](const State&) { return 0.0; };

std::string San(const State& state, Action a) {
  const auto& s = down_cast<const atomic_chess::AtomicChessState&>(state);
  return chess::ActionToMove(a, s.Board()).ToSAN(s.Board());
}

void IterativeDeepen(const std::string& label, std::unique_ptr<State> root,
                     int max_depth, double budget_sec) {
  auto game = root->GetGame();
  std::cout << "\n=== " << label << " ===\n"
            << down_cast<atomic_chess::AtomicChessState&>(*root).Board().ToFEN()
            << "\n  (side to move: player " << root->CurrentPlayer()
            << ", legal moves: " << root->LegalActions().size() << ")\n";
  for (int d = 1; d <= max_depth; ++d) {
    auto t0 = std::chrono::steady_clock::now();
    auto [value, action] = algorithms::AlphaBetaSearch(
        *game, root.get(), kZeroEval, d, kInvalidPlayer);
    double secs = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
    std::cout << "  depth " << d << ": value=" << value
              << "  best=" << San(*root, action) << "  (" << secs << "s)\n";
    if (value >= 1.0) {
      std::cout << "  -> FORCED WIN for the side to move proven at depth " << d
                << "\n";
      return;
    }
    if (secs > budget_sec) {
      std::cout << "  -> stopping: depth " << d << " took " << secs
                << "s; deeper is intractable for full-width search.\n";
      return;
    }
  }
  std::cout << "  -> no forced win found within depth " << max_depth
            << " (value stayed 0: neither side has a forced mate this shallow)."
            << "\n";
}

std::unique_ptr<State> AfterMoves(const std::shared_ptr<const Game>& game,
                                  const std::vector<std::string>& sans) {
  std::unique_ptr<State> s = game->NewInitialState();
  for (const std::string& mv : sans) {
    auto& as = down_cast<atomic_chess::AtomicChessState&>(*s);
    auto m = as.Board().ParseSANMove(mv);
    SPIEL_CHECK_TRUE(m.has_value());
    s->ApplyAction(chess::MoveToAction(*m, as.BoardSize()));
  }
  return s;
}

int main() {
  auto game = LoadGame("atomic_chess");
  const Player white = game->NewInitialState()->CurrentPlayer();

  // (1) Pure start position -- push as deep as stays tractable.
  IterativeDeepen("atomic start position", game->NewInitialState(),
                  /*max_depth=*/11, /*budget_sec=*/40.0);

  // (2) The classic atomic motif: knight to g5 eyeing Nxf7 (capturing f7
  // explodes the enemy king). Is it a FORCED win, or just a trap?
  // After 1.Nf3 d5 2.Ng5 it is Black to move. Search from White's perspective:
  // value +1 would mean every Black reply loses (forced); 0 means Black defends.
  {
    std::unique_ptr<State> s = AfterMoves(game, {"Nf3", "d5", "Ng5"});
    std::cout << "\n=== after 1.Nf3 d5 2.Ng5 (Black to move) ===\n"
              << down_cast<atomic_chess::AtomicChessState&>(*s).Board().ToFEN()
              << "\n";
    // Enumerate Black replies; count how many lose to an immediate Nxf7 mate.
    int total = 0, lose_now = 0;
    for (Action a : s->LegalActions()) {
      ++total;
      auto child = s->Clone();
      child->ApplyAction(a);
      auto [v, mv] = algorithms::AlphaBetaSearch(*game, child.get(), kZeroEval,
                                                 1, white);
      if (v >= 1.0) ++lose_now;  // White has a mate in 1 (Nxf7) against it.
    }
    std::cout << "  Black replies: " << total << "; of these " << lose_now
              << " lose immediately to a White mate-in-1 (the Nxf7 trap).\n";
    // Now the game-theoretic verdict for White with best defence by Black:
    for (int d = 2; d <= 6; ++d) {
      auto [v, mv] = algorithms::AlphaBetaSearch(*game, s.get(), kZeroEval, d,
                                                 white);
      std::cout << "  best-play value for White at depth " << d << " = " << v
                << "\n";
      if (v >= 1.0) break;
    }
    std::cout << "  (value 0 => Black has a defence => NOT a forced win.)\n";
  }

  // (3) Trap sprung: if Black blunders with 2...e6, White mates at once.
  IterativeDeepen("after 1.Nf3 d5 2.Ng5 e6?? (White to move)",
                  AfterMoves(game, {"Nf3", "d5", "Ng5", "e6"}),
                  /*max_depth=*/2, /*budget_sec=*/25.0);

  return 0;
}
