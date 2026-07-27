// Measure a trained AlphaZero atomic-chess checkpoint against Fairy-Stockfish.
//
// Design notes, because the naive version of this program gives numbers that
// look fine and mean nothing:
//
//  * PER-COLOUR SCORES. harnesses/theory_check.cc found that after 1.Nf3, 17 of
//    20 Black replies lose BY FORCE. If atomic is that White-favoured, an
//    aggregate 50% could be 100% as White and 0% as Black -- indistinguishable
//    from parity. Always read the split.
//  * PAIRED OPENINGS. Each opening is played twice, colours swapped, so a lucky
//    opening cannot flatter either side. Cuts variance substantially.
//  * UNFINISHED GAMES ARE NOT DRAWS. A game that hits the ply cap, or where SF
//    returns an unparseable move, is excluded from the score and reported
//    separately. Scoring them 0.5 silently inflates results.
//  * ucinewgame BETWEEN GAMES, so Stockfish does not carry hash and history
//    from one game into the next.
//  * NODES, NOT MILLISECONDS, by default. Movetime results depend on machine
//    load and are not reproducible across runs or hardware.
//  * CONFIDENCE INTERVALS. At n=40 the standard error on a score is ~8pp; the
//    README's "~17% (1W/6)" baseline has a 95% CI of roughly 0-50%. Any claim
//    without an interval is noise.
//
// Build inside the OpenSpiel tree (needs LibTorch); see eval/README or the
// top-level README. Runs on CPU with --device=/cpu:0, which is slower but needs
// no GPU allocation -- enough for a coarse first read.
//
// Usage:
//   az_vs_sf --az_path=$RUN_DIR --az_checkpoint=-1 --device=/cpu:0 \
//            --sf_path=/path/to/fairy-stockfish --games=40 \
//            --sf_nodes=100000 --az_sims=400

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/algorithms/alpha_zero_torch/device_manager.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpevaluator.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

#include "match.h"

ABSL_FLAG(std::string, az_path, "", "Directory of the AZ run (has vpnet.pb).");
ABSL_FLAG(std::string, az_graph_def, "vpnet.pb", "Graph def filename.");
ABSL_FLAG(int, az_checkpoint, -1, "Checkpoint step (-1 = most recent).");
ABSL_FLAG(std::string, device, "/cuda:0",
          "Torch device. /cpu:0 works and needs no GPU allocation.");
ABSL_FLAG(int, az_sims, 400,
          "MCTS simulations per move for the AZ bot. Evaluating with more "
          "simulations than training used is normal and usually worth Elo.");
ABSL_FLAG(double, az_uct_c, 2.0, "PUCT exploration constant.");

ABSL_FLAG(std::string, sf_path, "/opt/homebrew/bin/fairy-stockfish",
          "Fairy-Stockfish binary.");
ABSL_FLAG(int, sf_nodes, 100000,
          "Nodes per Stockfish move. Deterministic and hardware-independent, "
          "unlike movetime. Set 0 to use --sf_movetime instead.");
ABSL_FLAG(int, sf_movetime, 0, "Milliseconds per SF move; only if sf_nodes=0.");
ABSL_FLAG(int, sf_skill, -1,
          "Stockfish Skill Level 0..20. -1 leaves it unset. Prefer --sf_elo, "
          "which is a smooth dial rather than 21 blunder-injecting steps.");
ABSL_FLAG(int, sf_elo, 0,
          "If > 0, sets UCI_LimitStrength + UCI_Elo (~1350-2850).");
ABSL_FLAG(bool, sf_nnue, false,
          "Leave Stockfish's NNUE on. Default false: parity with the CLASSICAL "
          "evaluation is the milestone (MultiAra reached it; it lost to NNUE "
          "by ~300 Elo).");

ABSL_FLAG(int, games, 40, "Total games. Rounded down to an even number of "
                          "colour-swapped pairs.");
ABSL_FLAG(int, opening_plies, 4,
          "Random plies used to diversify each opening. Both games of a pair "
          "start from the same opening, with colours swapped.");
ABSL_FLAG(int, max_plies, 400, "Ply cap; games hitting it are excluded.");
ABSL_FLAG(int, seed, 1, "RNG seed for openings and MCTS.");
ABSL_FLAG(bool, verbose, false, "Print each game's moves.");

using namespace open_spiel;
namespace az = open_spiel::algorithms::torch_az;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_az_path).empty()) {
    std::cerr << "--az_path is required." << std::endl;
    return 1;
  }

  auto game = LoadGame("atomic_chess");
  // Derive rather than assume: OpenSpiel chess maps White -> player 1.
  const Player kWhite = game->NewInitialState()->CurrentPlayer();

  az::DeviceManager dm;
  dm.AddDevice(az::VPNetModel(*game, absl::GetFlag(FLAGS_az_path),
                              absl::GetFlag(FLAGS_az_graph_def),
                              absl::GetFlag(FLAGS_device)));
  dm.Get(0, 0)->LoadCheckpoint(absl::GetFlag(FLAGS_az_checkpoint));
  // One game at a time means one inference request in flight, so batch 1:
  // a larger batch would just wait for requests that cannot arrive.
  auto az_eval = std::make_shared<az::VPNetEvaluator>(
      &dm, /*batch_size=*/1, /*threads=*/1, /*cache_size=*/1 << 16,
      /*cache_shards=*/1);

  // uci::Options is a std::map<std::string, std::string>.
  uci::Options opts;
  opts["UCI_Variant"] = "atomic";
  opts["Use NNUE"] = absl::GetFlag(FLAGS_sf_nnue) ? "true" : "false";
  if (absl::GetFlag(FLAGS_sf_elo) > 0) {
    opts["UCI_LimitStrength"] = "true";
    opts["UCI_Elo"] = std::to_string(absl::GetFlag(FLAGS_sf_elo));
  } else if (absl::GetFlag(FLAGS_sf_skill) >= 0) {
    opts["Skill Level"] = std::to_string(absl::GetFlag(FLAGS_sf_skill));
  }

  const int nodes = absl::GetFlag(FLAGS_sf_nodes);
  const int movetime = absl::GetFlag(FLAGS_sf_movetime);
  const std::string go_cmd = nodes > 0
                                 ? absl::StrCat("go nodes ", nodes)
                                 : absl::StrCat("go movetime ", movetime);
  uci::UCIBot sf(absl::GetFlag(FLAGS_sf_path),
                 nodes > 0 ? nodes : movetime, /*ponder=*/false, opts,
                 nodes > 0 ? uci::SearchLimitType::kNodes
                           : uci::SearchLimitType::kMoveTime);

  atomic_az::MatchConfig cfg;
  cfg.pairs = std::max(1, absl::GetFlag(FLAGS_games) / 2);
  cfg.opening_plies = absl::GetFlag(FLAGS_opening_plies);
  cfg.max_plies = absl::GetFlag(FLAGS_max_plies);
  cfg.seed = absl::GetFlag(FLAGS_seed);
  cfg.verbose = absl::GetFlag(FLAGS_verbose);
  cfg.go_cmd = go_cmd;

  auto make_bot = [&](int seed) -> std::unique_ptr<Bot> {
    return std::make_unique<algorithms::MCTSBot>(
        *game, az_eval, absl::GetFlag(FLAGS_az_uct_c),
        absl::GetFlag(FLAGS_az_sims), /*max_memory_mb=*/1024,
        /*solve=*/true, seed, /*verbose=*/false,
        algorithms::ChildSelectionPolicy::PUCT,
        /*dirichlet_alpha=*/0, /*dirichlet_epsilon=*/0,
        /*dont_return_chance_node=*/true);
  };

  std::cout << "AZ(" << absl::GetFlag(FLAGS_az_sims) << " sims) vs "
            << "Fairy-Stockfish(atomic, " << go_cmd
            << ", NNUE=" << (absl::GetFlag(FLAGS_sf_nnue) ? "on" : "off");
  if (absl::GetFlag(FLAGS_sf_elo) > 0)
    std::cout << ", UCI_Elo=" << absl::GetFlag(FLAGS_sf_elo);
  else if (absl::GetFlag(FLAGS_sf_skill) >= 0)
    std::cout << ", skill=" << absl::GetFlag(FLAGS_sf_skill);
  std::cout << ")\n" << cfg.pairs << " colour-swapped pairs\n" << std::endl;

  atomic_az::MatchResult res = atomic_az::RunMatch(*game, &sf, make_bot, cfg);
  atomic_az::PrintResult(res, "AZ vs Fairy-Stockfish");
  return 0;
}
