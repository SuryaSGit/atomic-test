// Evaluate a trained AlphaZero (LibTorch) atomic-chess checkpoint against
// Fairy-Stockfish (atomic variant), alternating colours.
//
// STATUS: written against the OpenSpiel torch-AZ API (see
// examples/alpha_zero_torch_game_example.cc for the checkpoint-loading path and
// sf_bridge.cc for the UCI driver), but it can ONLY be compiled where OpenSpiel
// was built with OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON (i.e. on the GPU cluster) --
// it was NOT compiled locally. Build it as an example target guarded by
// ${OPEN_SPIEL_BUILD_WITH_LIBTORCH} (see README).
//
// Usage:
//   az_vs_sf --az_path=/path/to/run --az_checkpoint=-1 \
//            --sf_path=/usr/bin/fairy-stockfish --games=40 \
//            --sf_movetime=200 --sf_skill=20 --az_sims=400 --device=/gpu:0
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

ABSL_FLAG(std::string, az_path, "", "Directory of the AZ run (has vpnet.pb).");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb", "Graph def filename.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint step (-1 = most recent).");
ABSL_FLAG(std::string, device, "/gpu:0", "Torch device for inference.");
ABSL_FLAG(int, az_sims, 400, "MCTS simulations for the AZ bot.");
ABSL_FLAG(std::string, sf_path, "/usr/bin/fairy-stockfish", "Fairy-SF binary.");
ABSL_FLAG(int, sf_movetime, 200, "Fairy-Stockfish ms per move.");
ABSL_FLAG(int, sf_skill, 20, "Fairy-Stockfish Skill Level (0..20).");
ABSL_FLAG(int, games, 40, "Number of games (colours alternate).");

using namespace open_spiel;
namespace az = open_spiel::algorithms::torch_az;

Action StockfishMove(uci::UCIBot* sf, const atomic_chess::AtomicChessState& s,
                     int movetime_ms) {
  sf->Position(s.Board().ToFEN());
  sf->Write("go movetime " + std::to_string(movetime_ms));
  std::string best;
  while (true) {
    std::string line = sf->ReadLine();
    if (line.rfind("bestmove", 0) == 0) {
      std::istringstream iss(line);
      std::string t;
      iss >> t >> best;
      break;
    }
  }
  if (best.empty() || best == "(none)") return kInvalidAction;
  auto m = s.Board().ParseMove(best, /*natural=*/false);
  if (!m.has_value()) return kInvalidAction;
  return chess::MoveToAction(*m, s.BoardSize());
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto game = LoadGame("atomic_chess");
  const Player white = game->NewInitialState()->CurrentPlayer();

  // --- Load the trained AlphaZero net as an MCTS evaluator. ---
  az::DeviceManager dm;
  dm.AddDevice(az::VPNetModel(*game, absl::GetFlag(FLAGS_az_path),
                              absl::GetFlag(FLAGS_az_graph_def),
                              absl::GetFlag(FLAGS_device)));
  dm.Get(0, 0)->LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
  auto az_eval = std::make_shared<az::VPNetEvaluator>(
      &dm, /*batch_size=*/16, /*threads=*/1, /*cache_size=*/1 << 16,
      /*cache_shards=*/1);

  // --- Fairy-Stockfish over UCI in the atomic variant. ---
  uci::Options opts = {{"UCI_Variant", "atomic"},
                       {"Skill Level", std::to_string(absl::GetFlag(FLAGS_sf_skill))}};
  int mt = absl::GetFlag(FLAGS_sf_movetime);
  uci::UCIBot sf(absl::GetFlag(FLAGS_sf_path), mt, false, opts,
                 uci::SearchLimitType::kMoveTime);

  int games = absl::GetFlag(FLAGS_games);
  double az_pts = 0;
  for (int g = 0; g < games; ++g) {
    int az_seat = g % 2;
    auto az_bot = std::make_unique<algorithms::MCTSBot>(
        *game, az_eval, 2.0, absl::GetFlag(FLAGS_az_sims), 1024, true, 1000 + g,
        false, algorithms::ChildSelectionPolicy::PUCT, 0, 0,
        /*dont_return_chance_node=*/true);
    std::unique_ptr<State> s = game->NewInitialState();
    int ply = 0;
    while (!s->IsTerminal() && ply < 600) {
      const auto& as = down_cast<atomic_chess::AtomicChessState&>(*s);
      Action a = (s->CurrentPlayer() == az_seat)
                     ? az_bot->Step(*s)
                     : StockfishMove(&sf, as, mt);
      if (a == kInvalidAction) break;
      s->ApplyAction(a);
      ++ply;
    }
    double r = s->IsTerminal() ? s->Returns()[az_seat] : 0.0;
    az_pts += (r > 0 ? 1.0 : (r == 0 ? 0.5 : 0.0));
    std::cout << "game " << g << " AZ as "
              << (az_seat == white ? "White" : "Black") << ": return=" << r
              << " plies=" << ply << std::endl;
  }
  std::cout << "\nAZ(sims=" << absl::GetFlag(FLAGS_az_sims)
            << ") vs Fairy-Stockfish(atomic, skill="
            << absl::GetFlag(FLAGS_sf_skill) << ", " << mt
            << "ms): AZ score = " << (100.0 * az_pts / games) << "% over "
            << games << " games" << std::endl;
  return 0;
}
