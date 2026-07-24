// Can search PROVE a forced win in atomic_chess? AlphaBetaSearch returns the
// game-theoretic value under optimal play; with a 0-valued eval on non-terminal
// leaves, a returned value of +1 is a proof of a forced win (mate) within the
// search depth -- every opponent reply is accounted for.
#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

using namespace open_spiel;

// Non-terminal leaves are worth 0, so only genuine terminal wins yield +1.
static std::function<double(const State&)> kZeroEval =
    [](const State&) { return 0.0; };

std::string San(const State& state, Action a) {
  const auto& s = down_cast<const atomic_chess::AtomicChessState&>(state);
  chess::Move m = chess::ActionToMove(a, s.Board());
  return m.ToSAN(s.Board());
}

// Prints the value at increasing depths, then plays out the proven winning
// line move by move.
void Analyze(const std::string& label, const std::string& fen, int max_depth) {
  auto game = LoadGame("atomic_chess");
  std::cout << "\n=== " << label << " ===\n" << fen << "\n";

  int solved_depth = -1;
  for (int d = 1; d <= max_depth; ++d) {
    std::unique_ptr<State> s = game->NewInitialState(fen);
    auto [value, action] = algorithms::AlphaBetaSearch(
        *game, s.get(), kZeroEval, d, kInvalidPlayer);
    std::cout << "  depth " << d << ": value=" << value
              << "  best=" << San(*s, action) << "\n";
    if (value >= 1.0 && solved_depth < 0) solved_depth = d;
  }

  if (solved_depth < 0) {
    std::cout << "  -> no forced win found within depth " << max_depth << "\n";
    return;
  }
  std::cout << "  -> FORCED WIN proven at depth " << solved_depth << ".\n";

  // Play out the line: mover follows the proof; opponent plays its
  // alpha-beta-optimal (best defensive) reply. All lines must still lose.
  std::unique_ptr<State> s = game->NewInitialState(fen);
  Player winner = s->CurrentPlayer();
  std::cout << "  line:";
  int depth = solved_depth;
  while (!s->IsTerminal() && depth >= 1) {
    auto [value, action] =
        algorithms::AlphaBetaSearch(*game, s.get(), kZeroEval, depth,
                                    /*maximizing_player=*/winner);
    std::cout << " " << San(*s, action);
    s->ApplyAction(action);
    --depth;
  }
  std::cout << "\n  terminal: " << s->IsTerminal()
            << "  return[mover]=" << s->Returns()[winner] << "\n";
}

int main() {
  // (1) Mate in 1: Rxe7 blast removes the black king on e8.
  Analyze("mate in 1 (immediate king explosion)",
          "4k3/4r3/8/8/8/8/8/K3R3 w - - 0 1", 3);

  // (2) Genuine multi-move forced win (mate in 3): a two-rook mate. There is no
  // mate in 1 (Rb8+/Ra8+ both let the king step to the 7th rank). White plays
  // the quiet 1.Rb7 (sealing the 7th rank); the lone black king must move along
  // the 8th rank, and 2.Ra8 is checkmate. Pure rook mate, so it does not rely
  // on king proximity (where atomic's adjacency rule would change things).
  Analyze("forced mate in 3 (two-rook mate)",
          "3k4/8/R7/8/8/8/8/1R5K w - - 0 1", 5);

  return 0;
}
