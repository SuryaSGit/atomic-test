// Tests for pgn_atomic.h. Links against a plain CPU OpenSpiel build -- no
// LibTorch -- so the parsing can be verified on any machine before committing
// GPU hours. Build per pretrain/README.md.

#include "pgn_atomic.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace atomic_az {
namespace {

using open_spiel::Action;
using open_spiel::kInvalidAction;

// Move lists below were replayed against the atomic_chess implementation and
// their outcomes confirmed, so a parse failure here means the reader is broken,
// not the data.

// White (player 1) wins by exploding the black king with Qxe7. Exercises
// lichess-style clock comments, an eval comment, "1..." numbering and a '#'
// suffix on the final move.
constexpr char kGameWhiteWins[] = R"(
[Event "Casual Atomic game"]
[Site "https://lichess.org/abcd1234"]
[Variant "Atomic"]
[WhiteElo "2100"]
[BlackElo "1950"]
[Termination "Normal"]
[Result "1-0"]

1. d4 { [%eval 0.0] [%clk 0:03:00] } 1... Na6 { [%clk 0:03:00] } 2. Qd2 { [%clk 0:02:58] } 2... c5 3. Qg5 3... f5 4. Qxe7# 1-0

)";

// Black (player 0) wins with Qxf2. Same position set, opposite result.
constexpr char kGameBlackWins[] = R"(
[Event "Rated Atomic game"]
[Variant "Atomic"]
[WhiteElo "1900"]
[BlackElo "2000"]
[Result "0-1"]

1. d4 c6 2. e4 e6 3. Be2 Qh4+ 4. Qd2 Qxf2# 0-1

)";

// Contains castling, and uses the "1.d4" no-space style plus a '?!' annotation.
constexpr char kGameWithCastling[] = R"(
[Variant "Atomic"]
[WhiteElo "1930"]
[BlackElo "1940"]
[Result "0-1"]

1.d4 Nf6 2.e4 Ng4 3.Qxg4 g5 4.Be2 h5 5.f4 Bg7 6.Bd2 O-O 7.Bf1 d6 8.Be2 Qd7 9.Bf1?! Kh8 10.Be2 Qb5 11.fxg5 Qxe2 0-1

)";

constexpr char kGameDraw[] = R"(
[Variant "Atomic"]
[WhiteElo "2000"]
[BlackElo "2000"]
[Result "1/2-1/2"]

1. d4 c6 2. e4 e6 3. Be2 Qh4 1/2-1/2

)";

constexpr char kGameWrongVariant[] = R"(
[Variant "Standard"]
[WhiteElo "2400"]
[BlackElo "2400"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

)";

constexpr char kGameLowElo[] = R"(
[Variant "Atomic"]
[WhiteElo "1200"]
[BlackElo "1150"]
[Result "1-0"]

1. d4 Na6 2. Qd2 c5 3. Qg5 f5 4. Qxe7# 1-0

)";

constexpr char kGameUnfinished[] = R"(
[Variant "Atomic"]
[WhiteElo "2000"]
[BlackElo "2000"]
[Result "*"]

1. d4 c6 2. e4 e6 *

)";

// An illegal move for the position -- must be rejected, not silently skipped.
constexpr char kGameIllegalMove[] = R"(
[Variant "Atomic"]
[WhiteElo "2000"]
[BlackElo "2000"]
[Result "1-0"]

1. d4 c6 2. Qxh8 e6 1-0

)";

struct Collected {
  std::vector<Sample> samples;
  void operator()(Sample&& s) { samples.push_back(std::move(s)); }
};

Stats Run(const open_spiel::Game& game, const std::string& pgn,
          const Filter& filter, std::vector<Sample>* out) {
  std::istringstream in(pgn);
  Collected c;
  Stats s = ReadPgn(in, game, filter,
                    [&](Sample&& x) { c.samples.push_back(std::move(x)); });
  if (out) *out = std::move(c.samples);
  return s;
}

void TestNormalization() {
  SPIEL_CHECK_EQ(internal::NormalizeSan("Qxe7#"), "Qxe7");
  SPIEL_CHECK_EQ(internal::NormalizeSan("Qh4+"), "Qh4");
  SPIEL_CHECK_EQ(internal::NormalizeSan("Bf1?!"), "Bf1");
  SPIEL_CHECK_EQ(internal::NormalizeSan("e8=Q"), "e8Q");
  SPIEL_CHECK_EQ(internal::NormalizeSan("0-0"), "O-O");
  SPIEL_CHECK_EQ(internal::NormalizeSan("0-0-0"), "O-O-O");
  SPIEL_CHECK_EQ(internal::NormalizeSan("O-O"), "O-O");
  std::cout << "  normalization OK" << std::endl;
}

void TestTokenizer() {
  auto t = internal::TokenizeMovetext(
      "1. d4 { [%eval 0.0] [%clk 0:03:00] } 1... Na6 2. Qd2 $1 (2. e4 e5) "
      "2... c5 ; trailing\n3. Qg5 1-0");
  // Move numbers, the $1 NAG, the {clk} comment, the ";" comment, the "(2. e4
  // e5)" variation and the "1-0" result must all be dropped.
  SPIEL_CHECK_EQ(t.size(), 5);
  SPIEL_CHECK_EQ(t[0], "d4");
  SPIEL_CHECK_EQ(t[1], "Na6");
  SPIEL_CHECK_EQ(t[2], "Qd2");
  SPIEL_CHECK_EQ(t[3], "c5");
  SPIEL_CHECK_EQ(t[4], "Qg5");
  std::cout << "  tokenizer OK (variations, comments, NAGs dropped)"
            << std::endl;
}

void TestNoSpaceStyle() {
  auto t = internal::TokenizeMovetext("1.d4 Nf6 2.e4 Ng4");
  SPIEL_CHECK_EQ(t.size(), 4);
  SPIEL_CHECK_EQ(t[0], "d4");
  SPIEL_CHECK_EQ(t[2], "e4");
  std::cout << "  \"1.d4\" no-space style OK" << std::endl;
}

void TestValueSignConvention(const open_spiel::Game& game) {
  const open_spiel::Player white = game.NewInitialState()->CurrentPlayer();
  std::cout << "  White is player " << white << " (OpenSpiel chess: Black=0)"
            << std::endl;
  SPIEL_CHECK_EQ(white, 1);

  Filter f;
  std::vector<Sample> samples;
  Stats s = Run(game, kGameWhiteWins, f, &samples);
  SPIEL_CHECK_EQ(s.games_used, 1);
  SPIEL_CHECK_EQ(s.samples, 7);
  SPIEL_CHECK_EQ(samples.size(), 7);
  // "1-0": White won, White is player 1, so player 0's value is -1.
  for (const Sample& smp : samples) SPIEL_CHECK_EQ(smp.value, -1.0);
  std::cout << "  1-0 -> p0 value -1.0 over 7 plies OK" << std::endl;

  s = Run(game, kGameBlackWins, f, &samples);
  SPIEL_CHECK_EQ(s.games_used, 1);
  SPIEL_CHECK_EQ(s.samples, 8);
  for (const Sample& smp : samples) SPIEL_CHECK_EQ(smp.value, 1.0);
  std::cout << "  0-1 -> p0 value +1.0 over 8 plies OK" << std::endl;

  s = Run(game, kGameDraw, f, &samples);
  SPIEL_CHECK_EQ(s.games_used, 1);
  SPIEL_CHECK_EQ(s.samples, 6);
  for (const Sample& smp : samples) SPIEL_CHECK_EQ(smp.value, 0.0);
  std::cout << "  1/2-1/2 -> p0 value 0.0 OK" << std::endl;
}

void TestCastlingAndShape(const open_spiel::Game& game) {
  Filter f;
  std::vector<Sample> samples;
  Stats s = Run(game, kGameWithCastling, f, &samples);
  SPIEL_CHECK_EQ(s.games_used, 1);
  SPIEL_CHECK_EQ(s.samples, 22);

  const int obs_size = game.ObservationTensorSize();
  for (const Sample& smp : samples) {
    SPIEL_CHECK_EQ(smp.observation.size(), obs_size);
    SPIEL_CHECK_FALSE(smp.legal_actions.empty());
    SPIEL_CHECK_NE(smp.played_action, kInvalidAction);
    // The policy target must be reachable under the legal mask, else the
    // masked softmax would put zero probability on the target.
    bool found = false;
    for (Action a : smp.legal_actions) {
      if (a == smp.played_action) { found = true; break; }
    }
    SPIEL_CHECK_TRUE(found);
    SPIEL_CHECK_LT(smp.played_action, game.NumDistinctActions());
  }
  std::cout << "  castling parsed, 22 samples, obs size " << obs_size
            << ", targets inside legal mask OK" << std::endl;
}

void TestFilters(const open_spiel::Game& game) {
  Filter f;
  std::vector<Sample> samples;

  Stats s = Run(game, kGameWrongVariant, f, &samples);
  SPIEL_CHECK_EQ(s.games_seen, 1);
  SPIEL_CHECK_EQ(s.games_used, 0);
  SPIEL_CHECK_EQ(s.skipped_variant, 1);
  SPIEL_CHECK_TRUE(samples.empty());

  f.min_elo = 1900;
  s = Run(game, kGameLowElo, f, &samples);
  SPIEL_CHECK_EQ(s.games_used, 0);
  SPIEL_CHECK_EQ(s.skipped_elo, 1);
  SPIEL_CHECK_TRUE(samples.empty());

  f.min_elo = 0;
  s = Run(game, kGameUnfinished, f, &samples);
  SPIEL_CHECK_EQ(s.games_used, 0);
  SPIEL_CHECK_EQ(s.skipped_result, 1);
  SPIEL_CHECK_TRUE(samples.empty());

  s = Run(game, kGameIllegalMove, f, &samples);
  SPIEL_CHECK_EQ(s.games_used, 0);
  SPIEL_CHECK_EQ(s.skipped_parse, 1);
  // Critical: a rejected game must contribute zero samples, not a partial
  // prefix with a result label that never happened.
  SPIEL_CHECK_TRUE(samples.empty());
  SPIEL_CHECK_EQ(s.samples, 0);
  std::cout << "  filters (variant/elo/result/illegal) OK, no partial games"
            << std::endl;
}

void TestMultiGameStream(const open_spiel::Game& game) {
  const std::string all = std::string(kGameWhiteWins) + kGameBlackWins +
                          kGameWrongVariant + kGameWithCastling + kGameDraw;
  Filter f;
  std::vector<Sample> samples;
  Stats s = Run(game, all, f, &samples);
  SPIEL_CHECK_EQ(s.games_seen, 5);
  SPIEL_CHECK_EQ(s.games_used, 4);
  SPIEL_CHECK_EQ(s.skipped_variant, 1);
  SPIEL_CHECK_EQ(s.samples, 7 + 8 + 22 + 6);
  SPIEL_CHECK_EQ(samples.size(), 43);
  std::cout << "  5-game stream -> 4 used, 43 samples OK" << std::endl;

  std::istringstream in(all);
  Collected c;
  Stats capped = ReadPgn(in, game, f,
                         [&](Sample&& x) { c.samples.push_back(std::move(x)); },
                         /*max_games=*/2);
  SPIEL_CHECK_EQ(capped.games_seen, 2);
  SPIEL_CHECK_EQ(capped.samples, 15);
  std::cout << "  max_games cap OK" << std::endl;
}

}  // namespace
}  // namespace atomic_az

int main() {
  std::cout << "pgn_atomic_test" << std::endl;
  auto game = open_spiel::LoadGame("atomic_chess");
  atomic_az::TestNormalization();
  atomic_az::TestTokenizer();
  atomic_az::TestNoSpaceStyle();
  atomic_az::TestValueSignConvention(*game);
  atomic_az::TestCastlingAndShape(*game);
  atomic_az::TestFilters(*game);
  atomic_az::TestMultiGameStream(*game);
  std::cout << "ALL TESTS PASSED" << std::endl;
  return 0;
}
