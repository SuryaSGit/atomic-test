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
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
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
ABSL_FLAG(std::string, opponent_name, "Fairy-Stockfish",
          "Label for the opponent in the printed header. These logs are the "
          "record of what was played; a hardcoded name makes a MultiAra match "
          "read as a Stockfish one.");
ABSL_FLAG(std::string, sf_opt, "",
          "Extra UCI options, 'Name=Value' comma separated, applied after the "
          "defaults. Needed for non-Stockfish opponents. For MultiAra at "
          "atomic, Search_Type=mcts is MANDATORY: the binary defaults to mcgs, "
          "and Gehrke 2021 S4.4.5 abandoned MCGS for atomic because it "
          "'regularly generated NaN' -- 'which is why we decided to train "
          "Atomic with MCTS'. Also worth pinning Allow_Early_Stopping=false "
          "for a strict node match and Centi_Epsilon_Greedy=0 to remove search "
          "exploration meant for self-play.");
ABSL_FLAG(int, sf_hash, 256,
          "Fairy-Stockfish hash, MB. Sized for the node budget: the default "
          "16MB saturates well before 650k nodes/move. Fixed and reported "
          "because atomic scores are hash-sensitive.");
ABSL_FLAG(bool, sf_nnue, false,
          "Leave Stockfish's NNUE on. Default false: parity with the CLASSICAL "
          "evaluation is the milestone (MultiAra reached it; it lost to NNUE "
          "by ~300 Elo).");

ABSL_FLAG(int, games, 40, "Total games. Rounded down to an even number of "
                          "colour-swapped pairs.");
ABSL_FLAG(int, opening_plies, 4,
          "Random plies used to diversify each opening. Both games of a pair "
          "start from the same opening, with colours swapped.");
ABSL_FLAG(std::string, book, "",
          "EPD opening book (e.g. ianfab/books atomic.epd). Overrides "
          "--opening_plies. Random openings and book positions give very "
          "different scores -- measured 80% vs 63.5% for the same match at 4 "
          "vs 2 random plies -- so the opening source belongs in any reported "
          "result. MultiAra's published tournaments used these books.");
ABSL_FLAG(int, max_plies, 400, "Ply cap; games hitting it are excluded.");
ABSL_FLAG(int, seed, 1, "RNG seed for openings and MCTS.");
ABSL_FLAG(bool, verbose, false, "Print each game's moves.");
ABSL_FLAG(bool, value_calibration, true,
          "Measure the value head on the positions WE reach, and compare with "
          "its accuracy on held-out training positions. A large gap means "
          "distribution shift: competent on the teacher's positions, unreliable "
          "on its own. That distinguishes 'net too small' from 'net trained on "
          "the wrong distribution', which need different fixes.");
ABSL_FLAG(std::string, dump_games, "",
          "Write our games as UCI move lines, for relabelling by sf_label in "
          "an iteration round.");

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
  // Hash must be set explicitly and RECORDED. Fairy-SF's default saturates
  // (hashfull 988/1000 by 3M nodes), and atomic evaluations are strongly
  // hash-dependent -- measured: changing 64MB -> 1MB moved 35 of 77 scores,
  // mean |delta| 147cp, on identical positions at identical node counts. Two
  // runs at different hash sizes are not comparable.
  opts["Hash"] = std::to_string(absl::GetFlag(FLAGS_sf_hash));
  // Applied last so they can override any default above. An engine that does
  // not know an option reports it and continues ("Given option Hash does not
  // exist"), so passing Stockfish-specific names to MultiAra is harmless.
  for (absl::string_view kv :
       absl::StrSplit(absl::GetFlag(FLAGS_sf_opt), ',', absl::SkipEmpty())) {
    const size_t eq = kv.find('=');
    if (eq == absl::string_view::npos) {
      std::cerr << "--sf_opt entry '" << kv << "' is not Name=Value\n";
      return 1;
    }
    opts[std::string(kv.substr(0, eq))] = std::string(kv.substr(eq + 1));
  }
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
  cfg.book_path = absl::GetFlag(FLAGS_book);
  cfg.verbose = absl::GetFlag(FLAGS_verbose);
  cfg.go_cmd = go_cmd;
  cfg.dump_games_path = absl::GetFlag(FLAGS_dump_games);

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
            << absl::GetFlag(FLAGS_opponent_name) << "(atomic, " << go_cmd
            << ", NNUE=" << (absl::GetFlag(FLAGS_sf_nnue) ? "on" : "off");
  if (absl::GetFlag(FLAGS_sf_elo) > 0)
    std::cout << ", UCI_Elo=" << absl::GetFlag(FLAGS_sf_elo);
  else if (absl::GetFlag(FLAGS_sf_skill) >= 0)
    std::cout << ", skill=" << absl::GetFlag(FLAGS_sf_skill);
  std::cout << ")\n" << cfg.pairs << " colour-swapped pairs, openings="
            << (cfg.book_path.empty()
                    ? absl::StrCat(absl::GetFlag(FLAGS_opening_plies),
                                   " random plies")
                    : absl::StrCat("book ", cfg.book_path))
            << "\n" << std::endl;

  // --- value-head calibration on our own positions -------------------------
  // The raw network value, without search, at every position we had to move in.
  // Buffered per game because the label (who actually won) only exists at the
  // end.
  std::vector<double> game_values;      // player-0-relative, current game
  std::vector<double> all_values;       // over finished games
  std::vector<double> all_results;
  const bool calib = absl::GetFlag(FLAGS_value_calibration);

  atomic_az::PositionHook pos_hook = nullptr;
  atomic_az::ResultHook res_hook = nullptr;
  if (calib) {
    pos_hook = [&](const State& st, Player) {
      // Evaluate() returns {v, -v} for players {0, 1}, so index 0 is the
      // player-0-relative value -- the same convention the training targets use.
      game_values.push_back(az_eval->Evaluate(st)[0]);
    };
    res_hook = [&](double result_p0, bool finished) {
      if (finished && result_p0 != 0.0) {
        for (double v : game_values) {
          all_values.push_back(v);
          all_results.push_back(result_p0);
        }
      }
      game_values.clear();
    };
  }

  atomic_az::MatchResult res =
      atomic_az::RunMatch(*game, &sf, make_bot, cfg, pos_hook, res_hook);
  atomic_az::PrintResult(res, absl::StrCat("AZ vs ", absl::GetFlag(FLAGS_opponent_name)));

  if (calib && !all_values.empty()) {
    int64_t sign_ok = 0, res_pos = 0, pred_pos = 0;
    double sse = 0, pred_sum = 0, pred_sumsq = 0;
    for (size_t i = 0; i < all_values.size(); ++i) {
      sign_ok += ((all_values[i] >= 0) == (all_results[i] >= 0)) ? 1 : 0;
      const double d = all_values[i] - all_results[i];
      sse += d * d;
      if (all_results[i] > 0) res_pos += 1;
      if (all_values[i] >= 0) pred_pos += 1;
      pred_sum += all_values[i];
      pred_sumsq += all_values[i] * all_values[i];
    }
    const double n = static_cast<double>(all_values.size());
    const double acc = sign_ok / n;
    // The majority-class baseline on THESE positions. Comparing raw accuracy
    // across test sets is invalid when their base rates differ, and ours have
    // ranged from 0.54 to 0.65. Measured on this project, the same degradation
    // read as 0.618 (base 0.617) and 0.707 (base 0.648) -- identical in edge,
    // 9pp apart in accuracy, and a fixed-threshold verdict called the second
    // one "capacity, not shift". Report the edge.
    const double p = res_pos / n;
    const double base = std::max(p, 1.0 - p);
    const double edge = acc - base;
    const double mean = pred_sum / n;
    const double sd = std::sqrt(std::max(0.0, pred_sumsq / n - mean * mean));
    // Reference edge on the teacher's distribution: 0.711 accuracy against a
    // 0.540 base rate on the held-out SF-vs-SF shard.
    const double kRefEdge = 0.711 - 0.540;
    std::cout << "\n========== VALUE HEAD ON OUR OWN POSITIONS ==========\n"
              << "positions        : " << all_values.size()
              << "  (decisive games only)\n"
              << "result_sign_acc  : " << acc << "\n"
              << "majority baseline: " << base << "\n"
              << "EDGE over base   : " << edge << "\n"
              << "mse vs outcome   : " << (sse / n) << "\n"
              << "prediction mean  : " << mean << "  sd " << sd
              << "  frac>=0 " << (pred_pos / n) << "\n"
              << "reference edge   : " << kRefEdge
              << " on held-out SF-vs-SF positions (0.711 acc vs 0.540 base)\n";
    const double retained = kRefEdge > 0 ? edge / kRefEdge : 0.0;
    std::cout << "retained         : " << (100.0 * retained)
              << "% of the teacher-distribution edge\n";
    if (edge < 0.02) {
      std::cout << "READ: at or below the majority-class baseline. The value "
                   "head is not using the board on these positions at all.\n";
    } else if (retained < 0.60) {
      std::cout << "READ: most of the edge is lost off the teacher's "
                   "distribution. That is shift; relabelling our own games "
                   "addresses it, more capacity would not.\n";
    } else {
      std::cout << "READ: the edge largely survives on our own positions, so "
                   "shift is not the binding constraint. Look at capacity and "
                   "policy quality.\n";
    }
    if (sd < 0.10) {
      std::cout << "READ: predictions are nearly constant (sd " << sd
                << "); a fixed output scores the baseline on a skewed set.\n";
    }
  }
  return 0;
}
