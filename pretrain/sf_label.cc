// Generate a Fairy-Stockfish-labelled atomic-chess dataset for distillation.
//
// Plays SF-vs-SF games from randomised openings and, for every position,
// records SF's evaluation and its top-N moves. The output feeds az_pretrain as
// an alternative to human PGNs -- a stronger teacher, since Fairy-Stockfish at
// atomic is far above the Elo>=1900 lichess players MultiAra trained on.
//
// CPU only: no LibTorch, no GPU. Builds against a plain OpenSpiel build and is
// meant to run on a laptop or CPU nodes in parallel with GPU training.
//
// OUTPUT IS COMPACT ON PURPOSE. Storing VPNetModel::TrainInputs directly would
// cost ~4.3KB per position (the 1024-float observation) -- ~151GB for a
// 36M-position dataset. We store move sequences instead (~42 bytes/position,
// ~1.5GB) and az_pretrain replays them to rebuild observations.
//
// WHY MOVE SEQUENCES AND NOT PER-POSITION FENs: the observation tensor contains
// a repetition-count plane derived from the game history
// (atomic_chess.cc:387-392), so NewInitialState(fen) cannot reproduce the
// tensor for a position that has occurred before -- measured at 4 of 11,408
// positions in a round-trip test. Replaying from the start position is both
// exact and smaller.
//
// Format: one line per GAME, tab-separated; '#' lines are comments.
//   <result_p0>  <ply> <ply> <ply> ...
//
// where each <ply> is one of
//   e2e4                              unlabelled (random opening ply)
//   e2e4|25|e2e4:25,d2d4:20           played | stm_cp | multipv list
//
//   result_p0  Final game result for PLAYER 0, matching Sample::value. Note
//              OpenSpiel chess maps Black->0 and White->1, so a white win
//              stores -1.
//   stm_cp     SF score for the SIDE TO MOVE, centipawns. UCI's native
//              convention. Mate scores fold to +-(30000 - plies).
//   played     May differ from the first multipv move when --explore_prob
//              fires; the label still records what SF preferred.
//
// Usage (8 shards on a laptop):
//   for i in $(seq 0 7); do
//     ./sf_label --out=$DIR/atomic --shard=$i --num_shards=8 --games=2000 &
//   done; wait

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/bots/uci/uci_bot.h"
#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

ABSL_FLAG(std::string, sf_path, "/opt/homebrew/bin/fairy-stockfish",
          "Fairy-Stockfish binary.");
ABSL_FLAG(std::string, out, "atomic_sf", "Output prefix; .<shard>.tsv appended.");
ABSL_FLAG(std::string, games_in, "",
          "RELABEL MODE. Path to a .games file (az_vs_sf --dump_games format: "
          "result_p0 TAB uci uci ...). Instead of generating SF-vs-SF games, "
          "replay these and label every position with Stockfish, preserving "
          "the moves actually played and the recorded result. This is the "
          "DAgger step: it turns games OUR net played into training targets. "
          "--games, --random_plies and --explore_prob are ignored.");
ABSL_FLAG(int, games, 1000, "Games to generate in this shard.");
ABSL_FLAG(int, label_nodes, 10000, "Nodes per position. Higher = better labels.");
ABSL_FLAG(int, multipv, 4, "Top-N moves to record (1 = best move only).");
ABSL_FLAG(int, hash_mb, 64, "SF hash per process. Keep small when sharding.");
ABSL_FLAG(int, random_plies, 8,
          "Up to this many uniformly random opening plies per game, to spread "
          "the position distribution beyond SF's own preferences.");
ABSL_FLAG(double, explore_prob, 0.05,
          "Chance of playing a random legal move instead of SF's best. The "
          "label still records SF's best, so this widens coverage without "
          "corrupting targets.");
ABSL_FLAG(int, max_plies, 300, "Abandon (and discard) games longer than this.");
ABSL_FLAG(int, shard, 0, "Shard index; also seeds the RNG.");
ABSL_FLAG(int, num_shards, 1, "Total shards (for the seed offset only).");
ABSL_FLAG(int, progress_every, 100, "Log every N games to stderr.");
ABSL_FLAG(bool, use_nnue, true,
          "Leave SF's NNUE on if the build has an atomic net. A stronger "
          "teacher; set false to distil from the classical evaluation.");

using namespace open_spiel;

namespace {

constexpr int kMateBase = 30000;

struct PvEntry {
  int depth = -1;
  int cp = 0;
  std::string move;
};

// UCI/LAN text for an action, so replayed games can be parsed back with
// ParseMove. Used for the unlabelled random opening plies.
std::string ActionToUci(const atomic_chess::AtomicChessState& s, Action a) {
  return chess::ActionToMove(a, s.Board()).ToLAN();
}

int ParseIntTok(const std::vector<std::string>& t, size_t i) {
  return (i < t.size()) ? std::atoi(t[i].c_str()) : 0;
}

// Runs one fixed-node search and returns the MultiPV table, best rank first.
// Empty on failure or when SF reports no move.
std::vector<PvEntry> Search(uci::UCIBot* sf, const std::string& fen, int nodes) {
  sf->Position(fen);
  sf->Write(absl::StrCat("go nodes ", nodes));

  std::map<int, PvEntry> best;  // multipv rank -> deepest entry seen
  while (true) {
    std::string line = sf->ReadLine();
    if (line.rfind("bestmove", 0) == 0) break;
    if (line.rfind("info", 0) != 0) continue;
    if (line.find(" score ") == std::string::npos) continue;
    if (line.find("currmove") != std::string::npos) continue;

    std::vector<std::string> t;
    {
      std::istringstream iss(line);
      std::string w;
      while (iss >> w) t.push_back(w);
    }

    int rank = 1, depth = -1, cp = 0;
    bool have_score = false;
    std::string pv_move;
    for (size_t i = 0; i < t.size(); ++i) {
      if (t[i] == "multipv") {
        rank = ParseIntTok(t, i + 1);
      } else if (t[i] == "depth") {
        depth = ParseIntTok(t, i + 1);
      } else if (t[i] == "score" && i + 2 < t.size()) {
        if (t[i + 1] == "cp") {
          cp = ParseIntTok(t, i + 2);
          have_score = true;
        } else if (t[i + 1] == "mate") {
          // Fold mate distance into a saturating centipawn value so downstream
          // consumers need only one numeric type. tanh() will pin these to +-1.
          int m = ParseIntTok(t, i + 2);
          cp = (m > 0) ? (kMateBase - m) : (-kMateBase - m);
          have_score = true;
        }
      } else if (t[i] == "pv" && i + 1 < t.size()) {
        pv_move = t[i + 1];
        break;  // only the first PV move matters
      }
    }
    if (!have_score || pv_move.empty() || rank < 1) continue;
    if (depth >= best[rank].depth) best[rank] = PvEntry{depth, cp, pv_move};
  }

  std::vector<PvEntry> out;
  for (const auto& [rank, e] : best) {
    if (!e.move.empty()) out.push_back(e);
  }
  return out;
}

// Translate a UCI move through OUR board, so we never apply anything the
// atomic rules would reject. kInvalidAction on failure.
Action UciToAction(const atomic_chess::AtomicChessState& s,
                   const std::string& uci) {
  if (uci.empty() || uci == "(none)" || uci == "0000") return kInvalidAction;
  absl::optional<chess::Move> m = s.Board().ParseMove(uci, /*chess960=*/false);
  if (!m.has_value()) return kInvalidAction;
  return chess::MoveToAction(*m, s.BoardSize());
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const int shard = absl::GetFlag(FLAGS_shard);
  const int nodes = absl::GetFlag(FLAGS_label_nodes);
  const int multipv = std::max(1, absl::GetFlag(FLAGS_multipv));
  const int max_plies = absl::GetFlag(FLAGS_max_plies);
  const double explore = absl::GetFlag(FLAGS_explore_prob);

  auto game = LoadGame("atomic_chess");
  // Derive rather than assume: OpenSpiel chess maps White -> player 1.
  const Player white = game->NewInitialState()->CurrentPlayer();

  uci::Options opts = {
      {"UCI_Variant", "atomic"},
      {"MultiPV", std::to_string(multipv)},
      {"Hash", std::to_string(absl::GetFlag(FLAGS_hash_mb))},
      {"Use NNUE", absl::GetFlag(FLAGS_use_nnue) ? "true" : "false"},
  };
  uci::UCIBot sf(absl::GetFlag(FLAGS_sf_path), nodes, /*ponder=*/false, opts,
                 uci::SearchLimitType::kNodes);

  const std::string path =
      absl::StrCat(absl::GetFlag(FLAGS_out), ".", shard, ".tsv");
  std::ofstream out(path);
  if (!out) {
    std::cerr << "cannot open " << path << std::endl;
    return 1;
  }
  out << "# atomic_chess Fairy-Stockfish distillation dataset\n"
      << "# one line per game: result_p0 \\t ply ply ply ...\n"
      << "#   ply = 'e2e4' (unlabelled opening) or "
         "'played|stm_cp|uci1:cp1,uci2:cp2'\n"
      << "# stm_cp: centipawns for the SIDE TO MOVE (mate folded to +-"
      << kMateBase << ")\n"
      << "# result_p0: final result for PLAYER 0 (= Black in OpenSpiel chess),"
         " so a white win is -1\n"
      << "# replay from the start position: the observation tensor has a"
         " repetition plane that FENs cannot reproduce\n"
      << "# nodes=" << nodes << " multipv=" << multipv
      << " nnue=" << absl::GetFlag(FLAGS_use_nnue)
      << " explore_prob=" << explore << "\n";

  // ---- RELABEL MODE -------------------------------------------------------
  // Replay games we were given and label every position, rather than playing
  // new ones. The moves and the result come from the input; only the targets
  // are new. Sharding is by line, so N processes can split one file.
  const std::string games_in = absl::GetFlag(FLAGS_games_in);
  if (!games_in.empty()) {
    std::ifstream in(games_in);
    if (!in) {
      std::cerr << "cannot open --games_in " << games_in << std::endl;
      return 1;
    }
    int64_t line_no = 0, done = 0, positions = 0, bad_move = 0, bad_search = 0,
            past_end = 0;
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line[0] == '#') continue;
      // Shard by data-line index so the split is deterministic and disjoint.
      if ((line_no++ % std::max(1, absl::GetFlag(FLAGS_num_shards))) != shard)
        continue;

      const size_t tab = line.find('\t');
      if (tab == std::string::npos) { ++bad_move; continue; }
      const std::string result_field = line.substr(0, tab);

      // Accept bare 'e2e4' (a --dump_games file) and 'e2e4|cp|...' (an already
      // labelled file), so a set can be relabelled at different node counts.
      std::vector<std::string> moves;
      {
        std::istringstream iss(line.substr(tab + 1));
        std::string tok;
        while (iss >> tok) {
          const size_t bar = tok.find('|');
          moves.push_back(bar == std::string::npos ? tok : tok.substr(0, bar));
        }
      }
      if (moves.empty()) { ++bad_move; continue; }

      sf.Restart();  // ucinewgame: no hash or history carried between games
      std::unique_ptr<State> state = game->NewInitialState();
      std::vector<std::string> plies;
      bool ok = true;

      for (const std::string& mv : moves) {
        if (state->IsTerminal()) { ok = false; ++past_end; break; }
        const auto& as = down_cast<atomic_chess::AtomicChessState&>(*state);
        const Action a = UciToAction(as, mv);
        // UciToAction only parses; ParseMove can return a Move the atomic rules
        // reject, and ApplyAction on a non-legal action segfaults.
        bool legal = false;
        if (a != kInvalidAction) {
          for (Action l : state->LegalActions()) {
            if (l == a) { legal = true; break; }
          }
        }
        if (!legal) { ok = false; ++bad_move; break; }

        std::vector<PvEntry> pv = Search(&sf, as.Board().ToFEN(), nodes);
        if (pv.empty()) { ok = false; ++bad_search; break; }

        std::string mstr;
        for (const PvEntry& e : pv) {
          if (UciToAction(as, e.move) == kInvalidAction) continue;
          if (!mstr.empty()) mstr += ",";
          absl::StrAppend(&mstr, e.move, ":", e.cp);
        }
        // `played` is OUR move, which is usually NOT SF's best -- that
        // disagreement is exactly the training signal DAgger is after.
        plies.push_back(absl::StrCat(mv, "|", pv[0].cp, "|", mstr));
        state->ApplyAction(a);
      }

      if (!ok || plies.empty()) continue;
      out << result_field << '\t';
      for (size_t i = 0; i < plies.size(); ++i) {
        if (i) out << ' ';
        out << plies[i];
      }
      out << '\n';
      positions += static_cast<int64_t>(plies.size());
      ++done;

      if (absl::GetFlag(FLAGS_progress_every) > 0 &&
          done % absl::GetFlag(FLAGS_progress_every) == 0) {
        std::cerr << "[relabel " << shard << "] " << done << " games, "
                  << positions << " positions" << std::endl;
        out.flush();
      }
    }
    out.flush();
    std::cerr << "[relabel " << shard << "] DONE games=" << done
              << " positions=" << positions << " dropped(bad_move=" << bad_move
              << ", search=" << bad_search << ", past_end=" << past_end
              << ") -> " << path << std::endl;
    return 0;
  }
  // ---- end relabel mode ---------------------------------------------------

  std::mt19937 rng(1234567 + shard * 7919);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  int64_t emitted = 0, games_used = 0, dropped_unfinished = 0,
          dropped_parse = 0, dropped_early = 0;
  const int total_games = absl::GetFlag(FLAGS_games);

  for (int g = 0; g < total_games; ++g) {
    sf.Restart();  // ucinewgame: do not carry hash/history across games
    std::unique_ptr<State> state = game->NewInitialState();

    std::vector<std::string> plies;  // tokens, in order, for this game

    // Randomised opening, unlabelled -- pure position diversity.
    const int r = std::uniform_int_distribution<int>(
        0, std::max(0, absl::GetFlag(FLAGS_random_plies)))(rng);
    for (int i = 0; i < r && !state->IsTerminal(); ++i) {
      const auto& as = down_cast<atomic_chess::AtomicChessState&>(*state);
      std::vector<Action> legal = state->LegalActions();
      Action a = legal[std::uniform_int_distribution<size_t>(
          0, legal.size() - 1)(rng)];
      plies.push_back(ActionToUci(as, a));
      state->ApplyAction(a);
    }
    if (state->IsTerminal()) { ++dropped_early; continue; }

    bool ok = true;
    int ply = 0;
    while (!state->IsTerminal() && ply < max_plies) {
      const auto& as = down_cast<atomic_chess::AtomicChessState&>(*state);
      const std::string fen = as.Board().ToFEN();

      std::vector<PvEntry> pv = Search(&sf, fen, nodes);
      if (pv.empty()) { ok = false; break; }

      Action best_action = UciToAction(as, pv[0].move);
      if (best_action == kInvalidAction) {
        std::cerr << "unparseable SF move '" << pv[0].move << "' in " << fen
                  << std::endl;
        ok = false;
        break;
      }

      std::string moves;
      for (const PvEntry& e : pv) {
        if (UciToAction(as, e.move) == kInvalidAction) continue;  // skip only it
        if (!moves.empty()) moves += ",";
        absl::StrAppend(&moves, e.move, ":", e.cp);
      }

      // Occasionally deviate to widen coverage; the stored target is still
      // SF's best move for this position, so labels stay clean.
      Action play = best_action;
      if (explore > 0 && unit(rng) < explore) {
        std::vector<Action> legal = state->LegalActions();
        play = legal[std::uniform_int_distribution<size_t>(
            0, legal.size() - 1)(rng)];
      }
      plies.push_back(
          absl::StrCat(ActionToUci(as, play), "|", pv[0].cp, "|", moves));
      state->ApplyAction(play);
      ++ply;
    }

    if (!ok) { ++dropped_parse; continue; }
    if (!state->IsTerminal()) { ++dropped_unfinished; continue; }

    const double result_p0 = state->Returns()[0];
    out << result_p0;
    out << '\t';
    for (size_t i = 0; i < plies.size(); ++i) {
      if (i) out << ' ';
      out << plies[i];
    }
    out << '\n';
    emitted += ply;  // labelled positions only; opening plies carry no target
    ++games_used;

    if (absl::GetFlag(FLAGS_progress_every) > 0 &&
        (g + 1) % absl::GetFlag(FLAGS_progress_every) == 0) {
      std::cerr << "[shard " << shard << "] " << (g + 1) << "/" << total_games
                << " games, " << emitted << " positions" << std::endl;
      out.flush();
    }
  }

  out.flush();
  std::cerr << "[shard " << shard << "] DONE games_used=" << games_used
            << " positions=" << emitted
            << " dropped(unfinished=" << dropped_unfinished
            << ", parse=" << dropped_parse << ", early=" << dropped_early << ")"
            << " -> " << path << std::endl;
  // White is player 1, so a "white won" game stores result_p0 = -1.
  std::cerr << "[shard " << shard << "] (white=player " << white << ")"
            << std::endl;
  return 0;
}
