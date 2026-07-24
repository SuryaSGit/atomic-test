// Does search actually help on atomic_chess? Measure win rates:
//   (a) MCTS vs uniform-random, and
//   (b) MCTS-with-more-simulations vs MCTS-with-fewer.
// Colors are alternated each game to remove any first-move advantage.
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

using namespace open_spiel;

std::unique_ptr<Bot> Mcts(const Game& game, int sims, int seed) {
  auto eval = std::make_shared<algorithms::RandomRolloutEvaluator>(1, seed);
  return std::make_unique<algorithms::MCTSBot>(
      game, eval, /*uct_c=*/2.0, sims, /*max_memory_mb=*/256, /*solve=*/true,
      seed, /*verbose=*/false);
}

struct Score {
  int wins = 0, draws = 0, losses = 0;
  double points() const { return wins + 0.5 * draws; }
  int games() const { return wins + draws + losses; }
};

// Plays `n` games; make_hero(seed, seat) / make_opp(seed, seat) build bots
// bound to the given seat (player id). `hero` alternates seats each game.
template <typename MakeHero, typename MakeOpp>
Score Match(const Game& game, int n, MakeHero make_hero, MakeOpp make_opp) {
  Score s;
  for (int i = 0; i < n; ++i) {
    int hero_seat = i % 2;  // alternate colors.
    auto hero = make_hero(1000 + i, hero_seat);
    auto opp = make_opp(9000 + i, 1 - hero_seat);
    std::vector<Bot*> bots(2);
    bots[hero_seat] = hero.get();
    bots[1 - hero_seat] = opp.get();
    std::vector<double> returns = EvaluateBots(game, bots, /*seed=*/i);
    double r = returns[hero_seat];
    if (r > 0) s.wins++;
    else if (r < 0) s.losses++;
    else s.draws++;
  }
  return s;
}

void Report(const std::string& label, const Score& s) {
  std::cout << label << ": " << s.wins << "W / " << s.draws << "D / "
            << s.losses << "L over " << s.games() << "  ->  hero score "
            << (100.0 * s.points() / s.games()) << "%" << std::endl;
}

int main(int argc, char** argv) {
  int games = argc > 1 ? std::stoi(argv[1]) : 60;
  int sims = argc > 2 ? std::stoi(argv[2]) : 100;
  std::shared_ptr<const Game> game = LoadGame("atomic_chess");

  std::cout << "games=" << games << " mcts_sims=" << sims << "\n";

  // (a) MCTS vs uniform-random. Expect hero (MCTS) >> 50%.
  Score a = Match(
      *game, games,
      [&](int seed, int seat) { return Mcts(*game, sims, seed); },
      [&](int seed, int seat) { return MakeUniformRandomBot(seat, seed); });
  Report("MCTS(" + std::to_string(sims) + ") vs Random", a);

  // (b) MCTS strong vs MCTS weak. Expect strong >> 50% if search improves play.
  int strong = sims * 4, weak = sims / 5;
  Score b = Match(
      *game, games,
      [&](int seed, int seat) { return Mcts(*game, strong, seed); },
      [&](int seed, int seat) { return Mcts(*game, weak, seed); });
  Report("MCTS(" + std::to_string(strong) + ") vs MCTS(" +
             std::to_string(weak) + ")",
         b);

  // (c) Sanity baseline: Random vs Random should be ~50%.
  Score c = Match(
      *game, games,
      [&](int seed, int seat) { return MakeUniformRandomBot(seat, seed); },
      [&](int seed, int seat) {
        return MakeUniformRandomBot(seat, seed + 500000);
      });
  Report("Random vs Random (baseline)", c);

  return 0;
}
