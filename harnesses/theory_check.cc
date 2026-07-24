// Verifies published atomic-chess opening theory on our implementation:
//   "After 1.Nf3, 1...f6 is essentially forced; the threat 2.Ne5 (then
//    Nxd7/Nxf7 exploding the black king) refutes most replies; 1...e5 and
//    1...c5 lose by force."
// We evaluate every Black reply to 1.Nf3 with alpha-beta (value from White's
// perspective; +1 == proven forced win for White) and print the verdict.
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

std::unique_ptr<State> AfterSAN(const std::shared_ptr<const Game>& game,
                                const std::vector<std::string>& sans) {
  std::unique_ptr<State> s = game->NewInitialState();
  for (const auto& mv : sans) {
    auto& as = down_cast<atomic_chess::AtomicChessState&>(*s);
    auto m = as.Board().ParseSANMove(mv.c_str());
    SPIEL_CHECK_TRUE(m.has_value());
    s->ApplyAction(chess::MoveToAction(*m, as.BoardSize()));
  }
  return s;
}

void PrintLine(const std::shared_ptr<const Game>& game,
               std::unique_ptr<State> s, Player winner, int depth) {
  std::cout << "      line:";
  while (!s->IsTerminal() && depth >= 1) {
    auto [v, a] = algorithms::AlphaBetaSearch(*game, s.get(), kZeroEval, depth,
                                              winner);
    std::cout << " " << San(*s, a);
    s->ApplyAction(a);
    --depth;
  }
  std::cout << "   (terminal=" << s->IsTerminal()
            << ", White return=" << s->Returns()[winner] << ")\n";
}

int main() {
  auto game = LoadGame("atomic_chess");
  const Player white = game->NewInitialState()->CurrentPlayer();
  const int kDepth = 6;

  std::unique_ptr<State> after_nf3 = AfterSAN(game, {"Nf3"});
  std::cout << "After 1.Nf3 (Black to move). Evaluating all "
            << after_nf3->LegalActions().size()
            << " Black replies with alpha-beta depth " << kDepth
            << " (value from White's view; +1 = forced White win):\n\n";

  std::vector<std::string> holds, loses;
  for (Action a : after_nf3->LegalActions()) {
    std::string bmove = San(*after_nf3, a);
    auto child = after_nf3->Clone();
    child->ApplyAction(a);
    auto [value, wmove] =
        algorithms::AlphaBetaSearch(*game, child.get(), kZeroEval, kDepth,
                                    white);
    if (value >= 1.0) {
      loses.push_back(bmove);
      std::cout << "  1..." << bmove << "  -> White WINS by force (refutation: "
                << San(*child, wmove) << ")\n";
    } else {
      holds.push_back(bmove);
    }
  }

  std::cout << "\nBlack replies that HOLD (no forced White win within depth "
            << kDepth << "): ";
  for (auto& h : holds) std::cout << h << " ";
  std::cout << "\n  (" << loses.size() << " of "
            << (holds.size() + loses.size())
            << " Black replies are proven forced losses.)\n";

  // Show the forced winning lines for the two headline "losing" defences.
  for (const char* bad : {"c5", "e5"}) {
    std::cout << "\n=== refuting 1.Nf3 " << bad << " ===\n";
    auto pos = AfterSAN(game, {"Nf3", bad});
    auto [v, w] =
        algorithms::AlphaBetaSearch(*game, pos.get(), kZeroEval, kDepth, white);
    std::cout << "    White value = " << v << "\n";
    if (v >= 1.0) PrintLine(game, AfterSAN(game, {"Nf3", bad}), white, kDepth);
  }

  return 0;
}
