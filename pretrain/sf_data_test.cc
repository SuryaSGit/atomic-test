// Tests for sf_data.h, focused on the two sign conventions that meet in this
// file and on the value/policy target maths. Runs against a plain CPU build.

#include "sf_data.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace atomic_az {
namespace {

using open_spiel::Action;

bool Near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) < eps; }

// White (player 1) wins by exploding the black king with Qxe7 -- the same line
// verified against the engine in pgn_atomic_test.cc, re-expressed in UCI with
// synthetic Stockfish labels.
constexpr char kGame[] =
    "-1\td2d4|30|d2d4:30,e2e4:20 b8a6|-25|b8a6:-25,g8f6:-40 "
    "d1d2|45|d1d2:45,g1f3:10 c7c5|-60|c7c5:-60,e7e6:-80 "
    "d2g5|120|d2g5:120,b1c3:15 f7f5|-200|f7f5:-200,g8f6:-250 "
    "g5e7|29997|g5e7:29997,g5c5:300\n";

// Two unlabelled random opening plies, then labelled play.
constexpr char kGameWithOpening[] =
    "-1\td2d4 b8a6 d1d2|45|d1d2:45 c7c5|-60|c7c5:-60 "
    "d2g5|120|d2g5:120 f7f5|-200|f7f5:-200 g5e7|29997|g5e7:29997\n";

std::vector<Sample> Load(const open_spiel::Game& game, const std::string& text,
                         const SfOptions& opt, SfStats* out = nullptr) {
  std::istringstream in(text);
  std::vector<Sample> samples;
  SfStats s = ReadSfTsv(in, game, opt,
                        [&](Sample&& x) { samples.push_back(std::move(x)); });
  if (out) *out = s;
  return samples;
}

void TestValueSignConversion() {
  SfOptions opt;
  opt.lambda = 1.0;  // ignore the game result, isolate the search term
  opt.cp_scale = 400.0;

  // Player 1 (White) to move, +100cp for White -> good for White -> the
  // player-0-relative value must be NEGATIVE.
  const double v_white = ValueTarget(100, 0.0, /*mover=*/1, opt);
  SPIEL_CHECK_LT(v_white, 0.0);
  SPIEL_CHECK_TRUE(Near(v_white, -std::tanh(0.25)));

  // Player 0 (Black) to move, +100cp for Black -> value POSITIVE.
  const double v_black = ValueTarget(100, 0.0, /*mover=*/0, opt);
  SPIEL_CHECK_GT(v_black, 0.0);
  SPIEL_CHECK_TRUE(Near(v_black, std::tanh(0.25)));

  // Blending pulls the target toward the game result.
  opt.lambda = 0.5;
  const double blended = ValueTarget(100, -1.0, /*mover=*/1, opt);
  SPIEL_CHECK_TRUE(Near(blended, 0.5 * -std::tanh(0.25) + 0.5 * -1.0));

  // lambda=0 is pure outcome, regardless of the score or the mover.
  opt.lambda = 0.0;
  SPIEL_CHECK_TRUE(Near(ValueTarget(9999, -1.0, 1, opt), -1.0));
  SPIEL_CHECK_TRUE(Near(ValueTarget(-9999, -1.0, 0, opt), -1.0));
  std::cout << "  value sign + blend OK" << std::endl;
}

void TestPolicyTargets() {
  SfOptions opt;
  opt.policy_temp = 0.0;
  auto onehot = PolicyTarget({{7, 100}, {9, 50}, {11, -20}}, opt);
  SPIEL_CHECK_EQ(onehot.size(), 1);
  SPIEL_CHECK_EQ(onehot[0].first, 7);
  SPIEL_CHECK_TRUE(Near(onehot[0].second, 1.0));

  opt.policy_temp = 100.0;
  auto soft = PolicyTarget({{7, 100}, {9, 50}, {11, -20}}, opt);
  SPIEL_CHECK_EQ(soft.size(), 3);
  double sum = 0;
  for (const auto& [a, p] : soft) sum += p;
  SPIEL_CHECK_TRUE(Near(sum, 1.0, 1e-9));
  // Best move keeps the most mass, and mass decreases with score.
  SPIEL_CHECK_GT(soft[0].second, soft[1].second);
  SPIEL_CHECK_GT(soft[1].second, soft[2].second);
  // exp(-50/100) / (1 + exp(-0.5) + exp(-1.2))
  const double z = 1.0 + std::exp(-0.5) + std::exp(-1.2);
  SPIEL_CHECK_TRUE(Near(soft[0].second, 1.0 / z, 1e-9));
  std::cout << "  policy one-hot + softmax OK" << std::endl;
}

void TestReplay(const open_spiel::Game& game) {
  SfOptions opt;
  opt.lambda = 1.0;
  SfStats stats;
  std::vector<Sample> s = Load(game, kGame, opt, &stats);

  SPIEL_CHECK_EQ(stats.games_ok, 1);
  SPIEL_CHECK_EQ(stats.games_bad, 0);
  SPIEL_CHECK_EQ(s.size(), 7);
  SPIEL_CHECK_EQ(stats.opening_plies, 0);

  const int obs = game.ObservationTensorSize();
  for (const Sample& smp : s) {
    SPIEL_CHECK_EQ(smp.observation.size(), obs);
    SPIEL_CHECK_FALSE(smp.legal_actions.empty());
    SPIEL_CHECK_EQ(smp.policy.size(), 1);  // policy_temp == 0
    // Every policy target must be inside the legal mask.
    bool found = false;
    for (Action a : smp.legal_actions) {
      if (a == smp.policy[0].first) { found = true; break; }
    }
    SPIEL_CHECK_TRUE(found);
  }

  // game_result must stay the RAW outcome, unblended, on every sample.
  for (const Sample& smp : s) SPIEL_CHECK_EQ(smp.game_result, -1.0);
  // With lambda=1 the value target ignores the outcome, so the two differ.
  SPIEL_CHECK_NE(s[0].value, s[0].game_result);

  // Ply 0 is White (player 1) to move at +30cp -> negative for player 0.
  SPIEL_CHECK_LT(s[0].value, 0.0);
  // Ply 1 is Black (player 0) at -25cp -> also negative for player 0.
  SPIEL_CHECK_LT(s[1].value, 0.0);
  // The final position is a mate score for White; folded to ~-1 for player 0.
  SPIEL_CHECK_LT(s[6].value, -0.99);
  std::cout << "  replay: 7 samples, values oriented to player 0 OK"
            << std::endl;
}

void TestOpeningPlies(const open_spiel::Game& game) {
  SfOptions opt;
  SfStats stats;
  std::vector<Sample> s = Load(game, kGameWithOpening, opt, &stats);
  SPIEL_CHECK_EQ(stats.games_ok, 1);
  SPIEL_CHECK_EQ(stats.opening_plies, 2);
  SPIEL_CHECK_EQ(s.size(), 5);  // 7 plies - 2 unlabelled
  std::cout << "  unlabelled opening plies advance without emitting OK"
            << std::endl;
}

void TestCpFilter(const open_spiel::Game& game) {
  SfOptions opt;
  opt.max_abs_cp = 1000;  // drops the mate-scored final position
  SfStats stats;
  std::vector<Sample> s = Load(game, kGame, opt, &stats);
  SPIEL_CHECK_EQ(stats.games_ok, 1);
  SPIEL_CHECK_EQ(stats.skipped_cp, 1);
  SPIEL_CHECK_EQ(s.size(), 6);
  std::cout << "  max_abs_cp filter OK" << std::endl;
}

void TestRejectsBadLines(const open_spiel::Game& game) {
  SfOptions opt;
  SfStats stats;

  // Illegal move in the middle of the line.
  std::vector<Sample> s =
      Load(game, "-1\td2d4|30|d2d4:30 a1a8|0|a1a8:0\n", opt, &stats);
  SPIEL_CHECK_EQ(stats.games_bad, 1);
  SPIEL_CHECK_EQ(stats.games_ok, 0);
  SPIEL_CHECK_TRUE(s.empty());

  // Well-formed but never reaches a terminal state -> reject, since the stored
  // result would not describe the position.
  s = Load(game, "-1\td2d4|30|d2d4:30 b8a6|-25|b8a6:-25\n", opt, &stats);
  SPIEL_CHECK_TRUE(s.empty());

  // Missing tab.
  s = Load(game, "garbage without a tab\n", opt, &stats);
  SPIEL_CHECK_TRUE(s.empty());
  std::cout << "  malformed lines rejected with no partial output OK"
            << std::endl;
}

}  // namespace
}  // namespace atomic_az

int main() {
  std::cout << "sf_data_test" << std::endl;
  auto game = open_spiel::LoadGame("atomic_chess");
  atomic_az::TestValueSignConversion();
  atomic_az::TestPolicyTargets();
  atomic_az::TestReplay(*game);
  atomic_az::TestOpeningPlies(*game);
  atomic_az::TestCpFilter(*game);
  atomic_az::TestRejectsBadLines(*game);
  std::cout << "ALL TESTS PASSED" << std::endl;
  return 0;
}
