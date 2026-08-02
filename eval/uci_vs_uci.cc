// Play two UCI engines against each other at atomic, through OUR board.
//
// WHY THIS EXISTS. az_vs_sf measures our network against an engine, so every
// number is relative to us. That leaves the central question unanswerable: does
// MultiAra's published result -- "about even" with Fairy-Stockfish 13.1
// classical eval (Gehrke 2021 S5.2) -- reproduce on our hardware, in our units?
// Until MultiAra and Fairy-Stockfish are measured against EACH OTHER on the
// same ladder, our score against each of them cannot be placed on one scale,
// and the apparent contradiction (we beat MultiAra ~80%, we lose to Fairy-SF at
// 200k nodes with 18%) has no resolution beyond speculation.
//
// It also serves as a control on the whole harness: Fairy-SF vs Fairy-SF at
// equal nodes must score ~50%, and at unequal nodes must favour the deeper
// side. If it does not, the bug is here rather than in any engine.
//
// Needs no LibTorch, so it builds and runs anywhere sf_bridge does.
//
// Usage -- engine1's score is what gets reported:
//   uci_vs_uci --e1_path=.../multiara-uci --e1_nodes=800 \
//              --e1_opt="Search_Type=mcts,Threads=4" --e1_name=MultiAra \
//              --e2_path=.../fsf13/src/stockfish --e2_nodes=200000 \
//              --e2_opt="Hash=256" --e2_name=Fairy-SF-13.1 --games=200

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/spiel.h"

#include "match.h"

ABSL_FLAG(std::string, e1_path, "", "Engine 1 binary. Its score is reported.");
ABSL_FLAG(std::string, e1_name, "engine1", "Label for engine 1.");
ABSL_FLAG(int, e1_nodes, 800, "Nodes (or MCTS simulations) per move.");
ABSL_FLAG(std::string, e1_opt, "",
          "UCI options, 'Name=Value' comma separated. UCI_Variant=atomic is "
          "always set. For MultiAra at atomic, Search_Type=mcts is MANDATORY: "
          "the default mcgs is the mode Gehrke S4.4.5 abandoned for atomic "
          "because it 'regularly generated NaN'.");

ABSL_FLAG(std::string, e2_path, "", "Engine 2 binary (the opponent).");
ABSL_FLAG(std::string, e2_name, "engine2", "Label for engine 2.");
ABSL_FLAG(int, e2_nodes, 200000, "Nodes per move for engine 2.");
ABSL_FLAG(std::string, e2_opt, "",
          "UCI options for engine 2. For Fairy-Stockfish set Hash explicitly: "
          "the default saturates and atomic scores are hash-sensitive.");

ABSL_FLAG(int, games, 200, "Total games; rounded down to colour-swapped pairs.");
ABSL_FLAG(int, opening_plies, 4,
          "Random plies per opening. NOTE: MultiAra's own tournaments used "
          "opening BOOKS (<=5 plies of common lines from ianfab/books), not "
          "uniform random moves. Random atomic openings are far wilder and "
          "many are already decided -- a real difference from the published "
          "setup, and one that favours whichever engine trained on them.");
ABSL_FLAG(int, max_plies, 400, "Ply cap; games hitting it are excluded.");
ABSL_FLAG(int, seed, 1, "RNG seed for openings.");
ABSL_FLAG(bool, verbose, false, "Print each game's moves.");

using namespace open_spiel;

namespace {

// Adapts a UCI engine to the Bot interface RunMatch expects, so both sides go
// through the same legality-checked translation.
class UciAdapterBot : public Bot {
 public:
  UciAdapterBot(uci::UCIBot* engine, std::string go_cmd)
      : engine_(engine), go_cmd_(std::move(go_cmd)) {
    engine_->Restart();  // ucinewgame: RunMatch builds one bot per game
  }

  Action Step(const State& state) override {
    const auto& as =
        down_cast<const atomic_chess::AtomicChessState&>(state);
    return atomic_az::StockfishMove(engine_, as, go_cmd_);
  }

  void Restart() override { engine_->Restart(); }

 private:
  uci::UCIBot* engine_;
  std::string go_cmd_;
};

uci::Options ParseOpts(const std::string& csv) {
  uci::Options opts;
  opts["UCI_Variant"] = "atomic";
  for (absl::string_view kv : absl::StrSplit(csv, ',', absl::SkipEmpty())) {
    const size_t eq = kv.find('=');
    if (eq == absl::string_view::npos) {
      std::cerr << "option '" << kv << "' is not Name=Value" << std::endl;
      std::exit(1);
    }
    opts[std::string(kv.substr(0, eq))] = std::string(kv.substr(eq + 1));
  }
  return opts;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_e1_path).empty() ||
      absl::GetFlag(FLAGS_e2_path).empty()) {
    std::cerr << "--e1_path and --e2_path are required." << std::endl;
    return 1;
  }

  auto game = LoadGame("atomic_chess");

  const int e1_nodes = absl::GetFlag(FLAGS_e1_nodes);
  const int e2_nodes = absl::GetFlag(FLAGS_e2_nodes);

  uci::Options e1_opts = ParseOpts(absl::GetFlag(FLAGS_e1_opt));
  uci::Options e2_opts = ParseOpts(absl::GetFlag(FLAGS_e2_opt));

  uci::UCIBot e1(absl::GetFlag(FLAGS_e1_path), e1_nodes, /*ponder=*/false,
                 e1_opts, uci::SearchLimitType::kNodes);
  uci::UCIBot e2(absl::GetFlag(FLAGS_e2_path), e2_nodes, /*ponder=*/false,
                 e2_opts, uci::SearchLimitType::kNodes);

  atomic_az::MatchConfig cfg;
  cfg.pairs = std::max(1, absl::GetFlag(FLAGS_games) / 2);
  cfg.opening_plies = absl::GetFlag(FLAGS_opening_plies);
  cfg.max_plies = absl::GetFlag(FLAGS_max_plies);
  cfg.seed = absl::GetFlag(FLAGS_seed);
  cfg.verbose = absl::GetFlag(FLAGS_verbose);
  cfg.go_cmd = absl::StrCat("go nodes ", e2_nodes);

  const std::string e1_go = absl::StrCat("go nodes ", e1_nodes);
  auto make_bot = [&](int) -> std::unique_ptr<Bot> {
    return std::make_unique<UciAdapterBot>(&e1, e1_go);
  };

  std::cout << absl::GetFlag(FLAGS_e1_name) << "(" << e1_nodes << " nodes) vs "
            << absl::GetFlag(FLAGS_e2_name) << "(" << e2_nodes << " nodes)\n"
            << cfg.pairs << " colour-swapped pairs, opening_plies="
            << cfg.opening_plies << "\n"
            << std::endl;

  atomic_az::MatchResult res = atomic_az::RunMatch(*game, &e2, make_bot, cfg);
  atomic_az::PrintResult(res, absl::StrCat(absl::GetFlag(FLAGS_e1_name), " vs ",
                                           absl::GetFlag(FLAGS_e2_name)));
  return 0;
}
