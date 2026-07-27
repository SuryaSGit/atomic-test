// Streaming PGN reader that turns human games into AlphaZero training samples.
//
// Deliberately free of any LibTorch dependency so that the parsing logic --
// where all the risk lives -- can be compiled and tested against a plain CPU
// OpenSpiel build (see pgn_atomic_test.cc). az_pretrain.cc consumes this and
// converts Sample -> VPNetModel::TrainInputs.
//
// Moves are matched by comparing the PGN's SAN token against
// State::ActionToString() over the state's own LegalActions(). We never call
// ParseSANMove: going through LegalActions() guarantees we only ever emit
// actions the game itself considers legal, which matters for atomic chess
// because its legality rules (no self-king explosion, exploding the enemy king
// wins even while in check) live in AtomicChessState::LegalActions() rather
// than in the underlying chess board.

#ifndef ATOMIC_AZ_PRETRAIN_PGN_ATOMIC_H_
#define ATOMIC_AZ_PRETRAIN_PGN_ATOMIC_H_

#include <cctype>
#include <cstdint>
#include <functional>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

#include "sample.h"

namespace atomic_az {

struct Filter {
  // Applied to both players; 0 disables. The MultiAra thesis used the top 10th
  // percentile of lichess players, which for atomic was Elo >= 1900.
  int min_elo = 0;
  // Drop games that ended almost immediately (abandoned / misclicks).
  int min_plies = 4;
  // Require [Variant "Atomic"]. Turn off for single-variant exports.
  bool require_variant = true;
  std::string variant = "atomic";
  // Time forfeits keep a result that may not reflect the position.
  bool exclude_time_forfeit = false;
};

struct Stats {
  int64_t games_seen = 0;
  int64_t games_used = 0;
  int64_t skipped_variant = 0;
  int64_t skipped_elo = 0;
  int64_t skipped_result = 0;
  int64_t skipped_termination = 0;
  int64_t skipped_short = 0;
  int64_t skipped_parse = 0;
  int64_t samples = 0;

  std::string ToString() const {
    return "games_seen=" + std::to_string(games_seen) +
           " used=" + std::to_string(games_used) +
           " samples=" + std::to_string(samples) +
           " | skipped: variant=" + std::to_string(skipped_variant) +
           " elo=" + std::to_string(skipped_elo) +
           " result=" + std::to_string(skipped_result) +
           " termination=" + std::to_string(skipped_termination) +
           " short=" + std::to_string(skipped_short) +
           " parse=" + std::to_string(skipped_parse);
  }
};

namespace internal {

inline std::string ToLower(std::string s) {
  for (char& c : s) c = std::tolower(static_cast<unsigned char>(c));
  return s;
}

inline bool IsResultToken(const std::string& t) {
  return t == "1-0" || t == "0-1" || t == "1/2-1/2" || t == "*";
}

// "12." / "12..." / "12" -> true. Never true for "1-0" (checked before this).
inline bool IsPureMoveNumber(const std::string& t) {
  bool digit = false;
  for (char c : t) {
    if (std::isdigit(static_cast<unsigned char>(c))) {
      digit = true;
    } else if (c != '.') {
      return false;
    }
  }
  return digit;
}

// Strips a "12." / "12..." prefix from tokens like "12.e4" (PGNs that omit the
// space after the move number). Returns the SAN part, possibly empty.
inline std::string StripMoveNumberPrefix(const std::string& t) {
  size_t p = 0;
  while (p < t.size() && std::isdigit(static_cast<unsigned char>(t[p]))) ++p;
  if (p == 0 || p >= t.size() || t[p] != '.') return t;
  while (p < t.size() && t[p] == '.') ++p;
  return t.substr(p);
}

// Canonical form for comparing a PGN SAN token with ActionToString output.
// Removes check/mate/annotation marks, promotion '=', "e.p.", and normalises
// zero-style castling ("0-0" -> "O-O").
inline std::string NormalizeSan(const std::string& in) {
  std::string s = in;
  for (const char* marker : {"e.p.", "ep"}) {
    size_t pos = s.find(marker);
    if (pos != std::string::npos && marker[0] == 'e' && marker[1] == '.') {
      s.erase(pos, 4);
    }
  }
  std::string out;
  out.reserve(s.size());
  bool castling_only = !s.empty();
  for (char c : s) {
    if (c != 'O' && c != '0' && c != '-') castling_only = false;
  }
  for (char c : s) {
    if (c == '+' || c == '#' || c == '!' || c == '?' || c == '=') continue;
    if (castling_only && c == '0') {
      out.push_back('O');
      continue;
    }
    out.push_back(c);
  }
  return out;
}

// Splits movetext into SAN tokens, dropping {comments}, ;comments,
// (variations), $NAGs, move numbers and the result token.
inline std::vector<std::string> TokenizeMovetext(const std::string& text) {
  std::vector<std::string> out;
  size_t i = 0;
  int paren_depth = 0;
  while (i < text.size()) {
    char c = text[i];
    if (c == '{') {
      while (i < text.size() && text[i] != '}') ++i;
      if (i < text.size()) ++i;
      continue;
    }
    if (c == ';') {
      while (i < text.size() && text[i] != '\n') ++i;
      continue;
    }
    if (c == '(') { ++paren_depth; ++i; continue; }
    if (c == ')') { if (paren_depth > 0) --paren_depth; ++i; continue; }
    if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
    size_t start = i;
    while (i < text.size()) {
      char d = text[i];
      if (std::isspace(static_cast<unsigned char>(d)) || d == '{' || d == '(' ||
          d == ')' || d == ';') {
        break;
      }
      ++i;
    }
    std::string tok = text.substr(start, i - start);
    if (paren_depth > 0 || tok.empty()) continue;
    if (tok[0] == '$') continue;
    if (IsResultToken(tok)) continue;
    if (IsPureMoveNumber(tok)) continue;
    tok = StripMoveNumberPrefix(tok);
    if (tok.empty()) continue;
    out.push_back(tok);
  }
  return out;
}

inline int ParseIntOr(const std::string& s, int fallback) {
  if (s.empty()) return fallback;
  char* end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str()) return fallback;
  return static_cast<int>(v);
}

}  // namespace internal

// A single parsed PGN game: tag pairs plus raw movetext.
struct RawGame {
  std::string Tag(const std::string& key) const {
    for (const auto& [k, v] : tags) {
      if (k == key) return v;
    }
    return "";
  }
  std::vector<std::pair<std::string, std::string>> tags;
  std::string movetext;
};

enum class GameStatus { kOk, kTooShort, kParseFailed };

// Replays one game and emits a Sample per position where a move was played.
// Emits nothing unless the whole game replays cleanly -- a half-parsed game
// would otherwise contribute positions labelled with a result that the
// truncated line never reached.
inline GameStatus GameToSamples(const open_spiel::Game& game,
                                const RawGame& raw, double p0_value,
                                int min_plies,
                                const std::function<void(Sample&&)>& on_sample,
                                int* plies_out) {
  std::vector<std::string> tokens = internal::TokenizeMovetext(raw.movetext);
  if (static_cast<int>(tokens.size()) < min_plies) {
    if (plies_out) *plies_out = tokens.size();
    return GameStatus::kTooShort;
  }

  std::unique_ptr<open_spiel::State> state = game.NewInitialState();
  std::vector<Sample> pending;
  pending.reserve(tokens.size());

  for (const std::string& tok : tokens) {
    if (state->IsTerminal()) break;  // PGN continued past a decided game.
    const open_spiel::Player mover = state->CurrentPlayer();
    const std::string want = internal::NormalizeSan(tok);
    open_spiel::Action chosen = open_spiel::kInvalidAction;
    std::vector<open_spiel::Action> legal = state->LegalActions();
    for (open_spiel::Action a : legal) {
      if (internal::NormalizeSan(state->ActionToString(mover, a)) == want) {
        chosen = a;
        break;
      }
    }
    if (chosen == open_spiel::kInvalidAction) {
      if (plies_out) *plies_out = pending.size();
      return GameStatus::kParseFailed;
    }
    pending.push_back(Sample{std::move(legal), state->ObservationTensor(),
                             OneHot(chosen), p0_value, p0_value});
    state->ApplyAction(chosen);
  }

  if (plies_out) *plies_out = pending.size();
  if (static_cast<int>(pending.size()) < min_plies) {
    return GameStatus::kTooShort;
  }
  for (Sample& s : pending) on_sample(std::move(s));
  return GameStatus::kOk;
}

// Streams `in`, calling `on_sample` for every position of every accepted game.
// `max_games` < 0 means no limit. Counting is reported in the returned Stats.
inline Stats ReadPgn(std::istream& in, const open_spiel::Game& game,
                     const Filter& filter,
                     const std::function<void(Sample&&)>& on_sample,
                     int64_t max_games = -1) {
  Stats stats;
  // White moves first; derive the player id rather than assuming 0, because
  // OpenSpiel chess maps White -> player 1.
  const open_spiel::Player white = game.NewInitialState()->CurrentPlayer();

  RawGame raw;
  bool in_moves = false;

  auto finish = [&]() {
    if (raw.tags.empty() && raw.movetext.empty()) return;
    stats.games_seen += 1;
    RawGame cur = std::move(raw);
    raw = RawGame();
    in_moves = false;

    if (filter.require_variant) {
      const std::string v = internal::ToLower(cur.Tag("Variant"));
      if (v != filter.variant) { stats.skipped_variant += 1; return; }
    }
    if (filter.exclude_time_forfeit &&
        internal::ToLower(cur.Tag("Termination")) == "time forfeit") {
      stats.skipped_termination += 1;
      return;
    }
    if (filter.min_elo > 0) {
      const int we = internal::ParseIntOr(cur.Tag("WhiteElo"), -1);
      const int be = internal::ParseIntOr(cur.Tag("BlackElo"), -1);
      if (we < filter.min_elo || be < filter.min_elo) {
        stats.skipped_elo += 1;
        return;
      }
    }

    const std::string result = cur.Tag("Result");
    double p0_value;
    if (result == "1-0") {
      p0_value = (white == 0) ? 1.0 : -1.0;
    } else if (result == "0-1") {
      p0_value = (white == 0) ? -1.0 : 1.0;
    } else if (result == "1/2-1/2") {
      p0_value = 0.0;
    } else {
      stats.skipped_result += 1;
      return;
    }

    int plies = 0;
    GameStatus status = GameToSamples(
        game, cur, p0_value, filter.min_plies,
        [&](Sample&& s) { stats.samples += 1; on_sample(std::move(s)); },
        &plies);
    switch (status) {
      case GameStatus::kOk:          stats.games_used += 1;   break;
      case GameStatus::kTooShort:    stats.skipped_short += 1; break;
      case GameStatus::kParseFailed: stats.skipped_parse += 1; break;
    }
  };

  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const bool blank = line.find_first_not_of(" \t") == std::string::npos;

    if (!blank && line[0] == '[') {
      // A tag line while reading moves means the previous game ended without a
      // trailing blank line.
      if (in_moves) finish();
      const size_t q1 = line.find('"');
      const size_t q2 = line.rfind('"');
      size_t key_end = line.find_first_of(" \t", 1);
      if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1 &&
          key_end != std::string::npos) {
        raw.tags.emplace_back(line.substr(1, key_end - 1),
                              line.substr(q1 + 1, q2 - q1 - 1));
      }
      continue;
    }

    if (blank) {
      if (in_moves) {
        finish();
        if (max_games >= 0 && stats.games_seen >= max_games) return stats;
      }
      continue;
    }

    in_moves = true;
    raw.movetext += line;
    raw.movetext += "\n";
  }
  finish();
  return stats;
}

}  // namespace atomic_az

#endif  // ATOMIC_AZ_PRETRAIN_PGN_ATOMIC_H_
