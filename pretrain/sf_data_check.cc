// Validate an sf_label dataset by replaying every game through our engine.
//
// This is the contract az_pretrain depends on: each line must replay to a
// terminal state whose Returns()[0] equals the stored result, and every
// labelled move must be a legal action at the position it labels. Run it on a
// small sample before committing hours of laptop time, and again on the full
// dataset before training.
//
//   sf_data_check data.0.tsv [data.1.tsv ...]

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/games/atomic_chess/atomic_chess.h"
#include "open_spiel/games/chess/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"

using namespace open_spiel;

namespace {

struct Totals {
  int64_t games = 0, labelled = 0, unlabelled = 0;
  int64_t bad_parse = 0, bad_result = 0, not_terminal = 0, bad_label = 0;
  int64_t label_is_played = 0, mate_scores = 0;
};

// Verifies legality, not just parseability: ParseMove can return a Move the
// atomic rules reject, and ApplyAction on a non-legal action segfaults.
Action UciToAction(const atomic_chess::AtomicChessState& s,
                   const std::vector<Action>& legal, const std::string& uci) {
  auto m = s.Board().ParseMove(uci, /*chess960=*/false);
  if (!m.has_value()) return kInvalidAction;
  const Action a = chess::MoveToAction(*m, s.BoardSize());
  for (Action l : legal) {
    if (l == a) return a;
  }
  return kInvalidAction;
}

std::vector<std::string> Split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream iss(s);
  while (std::getline(iss, cur, d)) {
    if (!cur.empty()) out.push_back(cur);
  }
  return out;
}

void CheckLine(const Game& game, const std::string& line, Totals* t,
               int lineno, const std::string& file) {
  const size_t tab = line.find('\t');
  if (tab == std::string::npos) { t->bad_parse++; return; }
  const double stored_result = std::atof(line.substr(0, tab).c_str());
  const std::vector<std::string> plies = Split(line.substr(tab + 1), ' ');

  auto state = game.NewInitialState();
  for (const std::string& tok : plies) {
    if (state->IsTerminal()) {
      std::cerr << file << ":" << lineno << " moves continue past terminal\n";
      t->bad_parse++;
      return;
    }
    const auto& as = down_cast<atomic_chess::AtomicChessState&>(*state);
    const std::vector<Action> legal = state->LegalActions();
    const std::vector<std::string> parts = Split(tok, '|');

    if (parts.size() == 1) {           // unlabelled opening ply
      Action a = UciToAction(as, legal, parts[0]);
      if (a == kInvalidAction) {
        std::cerr << file << ":" << lineno << " illegal opening move '"
                  << parts[0] << "'\n";
        t->bad_parse++;
        return;
      }
      t->unlabelled++;
      state->ApplyAction(a);
      continue;
    }
    if (parts.size() != 3) { t->bad_parse++; return; }

    // parts = played | stm_cp | uci1:cp1,uci2:cp2,...
    Action played = UciToAction(as, legal, parts[0]);
    if (played == kInvalidAction) {
      std::cerr << file << ":" << lineno << " illegal played move '" << parts[0]
                << "'\n";
      t->bad_parse++;
      return;
    }
    if (std::abs(std::atoi(parts[1].c_str())) > 20000) t->mate_scores++;

    const std::vector<std::string> alts = Split(parts[2], ',');
    if (alts.empty()) { t->bad_label++; return; }
    bool first = true;
    for (const std::string& alt : alts) {
      const size_t colon = alt.rfind(':');
      if (colon == std::string::npos) { t->bad_label++; continue; }
      const std::string uci = alt.substr(0, colon);
      // The policy target must be a legal action, or the masked softmax would
      // put zero probability on it.
      if (UciToAction(as, legal, uci) == kInvalidAction) {
        std::cerr << file << ":" << lineno << " label move '" << uci
                  << "' not legal\n";
        t->bad_label++;
      }
      if (first && uci == parts[0]) t->label_is_played++;
      first = false;
    }

    t->labelled++;
    state->ApplyAction(played);
  }

  if (!state->IsTerminal()) { t->not_terminal++; return; }
  if (state->Returns()[0] != stored_result) {
    std::cerr << file << ":" << lineno << " result mismatch: stored "
              << stored_result << " but replay gives " << state->Returns()[0]
              << "\n";
    t->bad_result++;
    return;
  }
  t->games++;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: sf_data_check <file.tsv> [...]\n";
    return 2;
  }
  auto game = LoadGame("atomic_chess");
  Totals t;

  for (int i = 1; i < argc; ++i) {
    std::ifstream in(argv[i]);
    if (!in) { std::cerr << "cannot open " << argv[i] << "\n"; return 2; }
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
      ++lineno;
      if (line.empty() || line[0] == '#') continue;
      CheckLine(*game, line, &t, lineno, argv[i]);
    }
  }

  std::cout << "games replayed OK : " << t.games << "\n"
            << "labelled positions: " << t.labelled << "\n"
            << "opening plies     : " << t.unlabelled << "\n"
            << "mate-score labels : " << t.mate_scores << "\n"
            << "best == played    : " << t.label_is_played << " / " << t.labelled
            << "  (gap = explore_prob deviations)\n"
            << "--- failures ---\n"
            << "parse/illegal move: " << t.bad_parse << "\n"
            << "illegal label move: " << t.bad_label << "\n"
            << "never terminal    : " << t.not_terminal << "\n"
            << "result mismatch   : " << t.bad_result << "\n";
  const bool good = !t.bad_parse && !t.bad_label && !t.not_terminal &&
                    !t.bad_result;
  std::cout << (good ? "DATASET OK" : "DATASET HAS ERRORS") << std::endl;
  return good ? 0 : 1;
}
