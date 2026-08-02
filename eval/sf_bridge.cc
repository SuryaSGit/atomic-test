// Calibrate Fairy-Stockfish strength using plain random-rollout MCTS.
//
// Needs no LibTorch and no GPU, so it runs anywhere. Two jobs:
//
//  1. Establish the rungs for the evaluation ladder. Without this, the target
//     strengths in PLAN.md Phase 4 are invented numbers -- this sweeps node
//     counts (or Elo settings) and reports what a known-strength bot scores
//     against each, so later AlphaZero results have a scale to sit on.
//  2. Exercise the shared match logic in match.h -- pairing, per-colour
//     scoring, unfinished-game handling, confidence intervals -- on a machine
//     where az_vs_sf cannot be built.
//
// Both sides use OUR engine's rules: Fairy-Stockfish is fed the FEN produced by
// our atomic board and its reply is parsed with our own ParseMove, so there is
// no second atomic implementation to disagree with.
//
// Usage:
//   sf_bridge --sf_path=/opt/homebrew/bin/fairy-stockfish --games=20 \
//             --mcts_sims=800 --sweep=1000,10000,100000

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/spiel.h"

#include "match.h"

ABSL_FLAG(std::string, sf_path, "/opt/homebrew/bin/fairy-stockfish",
          "Fairy-Stockfish binary.");
ABSL_FLAG(int, games, 20, "Games per rung (rounded down to colour-swapped "
                          "pairs).");
ABSL_FLAG(int, mcts_sims, 800, "Random-rollout MCTS simulations per move.");
ABSL_FLAG(double, uct_c, 2.0, "UCT exploration constant.");
ABSL_FLAG(std::string, sweep, "",
          "Comma-separated Stockfish node counts to sweep, e.g. "
          "1000,10000,100000. Empty = a single rung at --sf_nodes.");
ABSL_FLAG(std::string, elo_sweep, "",
          "Comma-separated UCI_Elo values to sweep instead of node counts.");
ABSL_FLAG(int, sf_nodes, 10000, "Nodes per SF move for a single rung.");
ABSL_FLAG(int, sf_hash, 256,
          "Fairy-Stockfish hash, MB. Fixed and reported: atomic scores are "
          "hash-sensitive (see az_vs_sf.cc).");
ABSL_FLAG(bool, sf_nnue, false,
          "Leave NNUE on. Default false: the classical evaluation is the "
          "milestone opponent.");
ABSL_FLAG(int, opening_plies, 4, "Random plies per opening.");
ABSL_FLAG(int, max_plies, 400, "Ply cap; games hitting it are excluded.");
ABSL_FLAG(int, seed, 1, "RNG seed.");
ABSL_FLAG(bool, verbose, false, "Print moves.");

using namespace open_spiel;

namespace {

std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  for (absl::string_view p : absl::StrSplit(s, ',', absl::SkipEmpty())) {
    out.emplace_back(p);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto game = LoadGame("atomic_chess");

  const int sims = absl::GetFlag(FLAGS_mcts_sims);
  auto make_bot = [&](int seed) -> std::unique_ptr<Bot> {
    auto eval = std::make_shared<algorithms::RandomRolloutEvaluator>(1, seed);
    return std::make_unique<algorithms::MCTSBot>(
        *game, eval, absl::GetFlag(FLAGS_uct_c), sims,
        /*max_memory_mb=*/256, /*solve=*/true, seed, /*verbose=*/false);
  };

  atomic_az::MatchConfig cfg;
  cfg.pairs = std::max(1, absl::GetFlag(FLAGS_games) / 2);
  cfg.opening_plies = absl::GetFlag(FLAGS_opening_plies);
  cfg.max_plies = absl::GetFlag(FLAGS_max_plies);
  cfg.seed = absl::GetFlag(FLAGS_seed);
  cfg.verbose = absl::GetFlag(FLAGS_verbose);

  // Build the list of rungs to sweep.
  std::vector<std::pair<std::string, std::string>> rungs;  // label, extra
  const std::vector<std::string> node_sweep = SplitCsv(absl::GetFlag(FLAGS_sweep));
  const std::vector<std::string> elo_sweep =
      SplitCsv(absl::GetFlag(FLAGS_elo_sweep));
  if (!elo_sweep.empty()) {
    for (const auto& e : elo_sweep) rungs.push_back({absl::StrCat("elo=", e), e});
  } else if (!node_sweep.empty()) {
    for (const auto& n : node_sweep)
      rungs.push_back({absl::StrCat("nodes=", n), n});
  } else {
    rungs.push_back({absl::StrCat("nodes=", absl::GetFlag(FLAGS_sf_nodes)),
                     std::to_string(absl::GetFlag(FLAGS_sf_nodes))});
  }

  std::vector<std::string> summary;
  for (const auto& [label, value] : rungs) {
    uci::Options opts;
    opts["UCI_Variant"] = "atomic";
    opts["Use NNUE"] = absl::GetFlag(FLAGS_sf_nnue) ? "true" : "false";
    // See az_vs_sf.cc: fixed and recorded, because atomic scores move with it.
    opts["Hash"] = std::to_string(absl::GetFlag(FLAGS_sf_hash));
    int limit;
    if (!elo_sweep.empty()) {
      opts["UCI_LimitStrength"] = "true";
      opts["UCI_Elo"] = value;
      limit = absl::GetFlag(FLAGS_sf_nodes);
      cfg.go_cmd = absl::StrCat("go nodes ", limit);
    } else {
      limit = std::atoi(value.c_str());
      cfg.go_cmd = absl::StrCat("go nodes ", limit);
    }

    std::cout << "\n### rung " << label << " (MCTS " << sims << " sims) ###"
              << std::endl;
    uci::UCIBot sf(absl::GetFlag(FLAGS_sf_path), limit, /*ponder=*/false, opts,
                   uci::SearchLimitType::kNodes);
    atomic_az::MatchResult res =
        atomic_az::RunMatch(*game, &sf, make_bot, cfg);
    atomic_az::PrintResult(res, absl::StrCat("MCTS", sims, " vs SF ", label));
    summary.push_back(absl::StrCat(label, "  ", res.overall().Report()));
  }

  std::cout << "\n================ LADDER ================" << std::endl;
  std::cout << "MCTS(" << sims << " sims, random rollouts) scores:" << std::endl;
  for (const auto& line : summary) std::cout << "  " << line << std::endl;
  std::cout << "\nUse these as the calibrated rungs for PLAN.md Phase 4: the "
               "node count where a known bot scores ~50% is the level to aim "
               "the trained net at first." << std::endl;
  return 0;
}
