// Eval bridge: OpenSpiel atomic_chess bot vs Fairy-Stockfish (atomic variant).
//
// Both sides use OUR engine's rules: we feed Fairy-Stockfish the FEN produced
// by our atomic board and parse its UCI reply with our own ParseMove, so there
// is no risk of a rules mismatch between two atomic implementations. Fairy-SF
// is driven through UCIBot's public pipe helpers (Position/Write/ReadLine);
// we avoid UCIBot::Step because it hard-casts the state to chess::ChessState.
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

using namespace open_spiel;

// Ask Fairy-Stockfish for a move in the current atomic position and translate
// it into our action id. Returns kInvalidAction if SF reports no move.
Action StockfishMove(uci::UCIBot* sf, const atomic_chess::AtomicChessState& s,
                     int movetime_ms) {
  sf->Position(s.Board().ToFEN());
  sf->Write("go movetime " + std::to_string(movetime_ms));
  std::string best;
  while (true) {
    std::string line = sf->ReadLine();
    if (line.rfind("bestmove", 0) == 0) {
      std::istringstream iss(line);
      std::string tok;
      iss >> tok >> best;  // "bestmove <uci>"
      break;
    }
  }
  if (best.empty() || best == "(none)") return kInvalidAction;
  absl::optional<chess::Move> m = s.Board().ParseMove(best, /*natural=*/false);
  if (!m.has_value()) {
    std::cerr << "  !! could not parse SF move '" << best << "' in "
              << s.Board().ToFEN() << std::endl;
    return kInvalidAction;
  }
  return chess::MoveToAction(*m, s.BoardSize());
}

std::unique_ptr<Bot> MctsBot(const Game& game, int sims, int seed) {
  auto eval = std::make_shared<algorithms::RandomRolloutEvaluator>(1, seed);
  return std::make_unique<algorithms::MCTSBot>(
      game, eval, 2.0, sims, /*max_memory_mb=*/256, /*solve=*/true, seed,
      /*verbose=*/false);
}

int main(int argc, char** argv) {
  std::string sf_path = argc > 1 ? argv[1] : "/opt/homebrew/bin/fairy-stockfish";
  int games = argc > 2 ? std::stoi(argv[2]) : 2;
  int sf_movetime = argc > 3 ? std::stoi(argv[3]) : 100;   // ms per SF move.
  int mcts_sims = argc > 4 ? std::stoi(argv[4]) : 400;
  int sf_skill = argc > 5 ? std::stoi(argv[5]) : 20;       // 0..20 (20=full).

  auto game = LoadGame("atomic_chess");
  const Player white = game->NewInitialState()->CurrentPlayer();

  uci::Options opts = {{"UCI_Variant", "atomic"},
                       {"Skill Level", std::to_string(sf_skill)}};
  uci::UCIBot sf(sf_path, sf_movetime, /*ponder=*/false, opts,
                 uci::SearchLimitType::kMoveTime);

  int mcts_pts2 = 0;  // MCTS score *2 (win=2, draw=1, loss=0).
  for (int g = 0; g < games; ++g) {
    int mcts_seat = g % 2;  // alternate colors.
    auto mcts = MctsBot(*game, mcts_sims, 1000 + g);
    std::unique_ptr<State> state = game->NewInitialState();
    int ply = 0;
    while (!state->IsTerminal() && ply < 600) {
      const auto& as = down_cast<atomic_chess::AtomicChessState&>(*state);
      Player cur = state->CurrentPlayer();
      Action a;
      if (cur == mcts_seat) {
        a = mcts->Step(*state);
      } else {
        a = StockfishMove(&sf, as, sf_movetime);
        if (a == kInvalidAction) break;
      }
      state->ApplyAction(a);
      ++ply;
    }
    double mr = state->IsTerminal() ? state->Returns()[mcts_seat] : 0.0;
    mcts_pts2 += (mr > 0 ? 2 : (mr == 0 ? 1 : 0));
    std::cout << "game " << g << " (MCTS as " << (mcts_seat == white ? "White" : "Black")
              << "): plies=" << ply << " terminal=" << state->IsTerminal()
              << " MCTS_return=" << mr << std::endl;
  }
  std::cout << "\nMCTS(" << mcts_sims << " sims) vs Fairy-Stockfish(atomic, "
            << sf_movetime << "ms): MCTS score = " << (50.0 * mcts_pts2 / games)
            << "% over " << games << " games" << std::endl;
  return 0;
}
