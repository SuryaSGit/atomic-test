// Smoke test: do the generic OpenSpiel bots work on atomic_chess?
#include <iostream>
#include <memory>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

using namespace open_spiel;

int g_fail = 0;
#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      ++g_fail;                                                      \
      std::cerr << "FAIL: " << (msg) << std::endl;                   \
    }                                                                \
  } while (0)

std::unique_ptr<Bot> MctsBot(const Game& game, int sims, int seed) {
  auto evaluator =
      std::make_shared<algorithms::RandomRolloutEvaluator>(/*n_rollouts=*/1,
                                                           seed);
  return std::make_unique<algorithms::MCTSBot>(
      game, evaluator, /*uct_c=*/2.0, /*max_simulations=*/sims,
      /*max_memory_mb=*/64, /*solve=*/true, /*seed=*/seed, /*verbose=*/false);
}

// Play one full game with the two bots and validate the outcome.
void PlayGame(const std::string& label, const Game& game, Bot* b0, Bot* b1,
              int seed) {
  std::vector<Bot*> bots = {b0, b1};
  std::vector<double> returns = EvaluateBots(game, bots, seed);
  CHECK(returns.size() == 2, label + ": returns size");
  double sum = returns[0] + returns[1];
  CHECK(sum == 0.0, label + ": zero-sum (got " + absl::StrJoin(returns, ",") +
                        ")");
  for (double r : returns) {
    CHECK(r == -1.0 || r == 0.0 || r == 1.0, label + ": return in {-1,0,1}");
  }
  std::cout << label << ": returns = [" << absl::StrJoin(returns, ", ")
            << "]" << std::endl;
}

// Manually drive a game so we can assert every bot action is legal and the
// game actually terminates (EvaluateBots asserts legality internally too, but
// this makes the coverage explicit).
void StepThroughGame(const std::string& label, const Game& game,
                     std::vector<Bot*> bots, int max_steps) {
  std::unique_ptr<State> state = game.NewInitialState();
  int steps = 0;
  while (!state->IsTerminal() && steps < max_steps) {
    Player p = state->CurrentPlayer();
    CHECK(p == 0 || p == 1, label + ": valid current player");
    Action a = bots[p]->Step(*state);
    const std::vector<Action>& legal = state->LegalActions();
    CHECK(std::find(legal.begin(), legal.end(), a) != legal.end(),
          label + ": bot returned a legal action");
    state->ApplyAction(a);
    ++steps;
  }
  CHECK(state->IsTerminal(), label + ": game reached terminal within limit");
  std::cout << label << ": terminated after " << steps
            << " plies; returns = [" << absl::StrJoin(state->Returns(), ", ")
            << "]" << std::endl;
}

int main() {
  std::shared_ptr<const Game> game = LoadGame("atomic_chess");

  // 1. Uniform-random vs uniform-random, several seeds.
  for (int seed = 1; seed <= 5; ++seed) {
    auto r0 = MakeUniformRandomBot(0, seed * 7 + 1);
    auto r1 = MakeUniformRandomBot(1, seed * 7 + 2);
    PlayGame("random-vs-random seed=" + std::to_string(seed), *game, r0.get(),
             r1.get(), seed);
  }

  // 2. Stateful random bot (exercises a different generic bot path).
  {
    auto s0 = MakeStatefulRandomBot(*game, 0, 123);
    auto s1 = MakeStatefulRandomBot(*game, 1, 456);
    PlayGame("stateful-random", *game, s0.get(), s1.get(), 99);
  }

  // 3. MCTS vs uniform-random.
  {
    auto mcts = MctsBot(*game, /*sims=*/30, /*seed=*/2024);
    auto rnd = MakeUniformRandomBot(1, 2025);
    StepThroughGame("mcts-vs-random", *game, {mcts.get(), rnd.get()},
                    /*max_steps=*/400);
  }

  // 4. MCTS vs MCTS (both use search on the atomic game tree).
  {
    auto m0 = MctsBot(*game, /*sims=*/25, /*seed=*/11);
    auto m1 = MctsBot(*game, /*sims=*/25, /*seed=*/22);
    StepThroughGame("mcts-vs-mcts", *game, {m0.get(), m1.get()},
                    /*max_steps=*/400);
  }

  std::cerr << "\n==== bot_check: " << (g_fail == 0 ? "ALL OK" : "FAILURES")
            << " (" << g_fail << " failures) ====\n";
  return g_fail == 0 ? 0 : 1;
}
