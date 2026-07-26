// The training example shared by every data source (human PGNs, Fairy-Stockfish
// distillation, and anything added later). Deliberately free of LibTorch so the
// readers can be unit-tested against a plain CPU OpenSpiel build;
// az_pretrain.cc converts these into VPNetModel::TrainInputs.

#ifndef ATOMIC_AZ_PRETRAIN_SAMPLE_H_
#define ATOMIC_AZ_PRETRAIN_SAMPLE_H_

#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

namespace atomic_az {

struct Sample {
  std::vector<open_spiel::Action> legal_actions;
  std::vector<float> observation;

  // Policy target as (action, probability) pairs, matching
  // VPNetModel::TrainInputs::policy. A human PGN yields one-hot on the played
  // move; Stockfish distillation can yield a softmax over its MultiPV list.
  open_spiel::ActionsAndProbs policy;

  // ALWAYS player 0's value, never the mover's. VPNetEvaluator::Evaluate
  // returns {v, -v} for players {0, 1} and the RL learner feeds returns[0] for
  // every state in a trajectory. OpenSpiel chess maps Black -> 0, White -> 1,
  // so a white win is -1.
  double value = 0.0;
};

// Convenience for the common one-hot case.
inline open_spiel::ActionsAndProbs OneHot(open_spiel::Action a) {
  return {{a, 1.0}};
}

}  // namespace atomic_az

#endif  // ATOMIC_AZ_PRETRAIN_SAMPLE_H_
