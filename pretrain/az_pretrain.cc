// Supervised pretraining for the OpenSpiel LibTorch AlphaZero atomic-chess net.
//
// Trains the SAME VPNetModel the RL loop uses, on human games from the lichess
// open database, and writes `checkpoint--1.pt` so that alpha_zero_torch_example
// resumes from a warm net instead of random weights.
//
// Rationale: the MultiAra study (Gehrke 2021, TU Darmstadt) found that a model
// trained from zero knowledge ended ~220 Elo WEAKER than one initialised from
// human games, after 26 days on 4 Tesla V100s -- while the supervised model
// itself takes only hours on a single GPU. Atomic was also the variant where
// self-play RL improved least (+150 Elo vs +300..690 for other variants), so
// the supervised starting point matters more here, not less.
//
// Key correctness points, all verified against the OpenSpiel source:
//   * The value head is PLAYER-0-relative. VPNetEvaluator::Evaluate returns
//     {v, -v} for players {0, 1} and the RL learner feeds returns[0] for every
//     state in a trajectory. In OpenSpiel chess player 0 is BLACK, so a "1-0"
//     PGN result becomes value = -1. pgn_atomic.h handles this and
//     pgn_atomic_test.cc asserts it.
//   * Training goes through VPNetModel::Learn, so the loss, legal-action
//     masking and optimizer are byte-identical to the RL loop's.
//   * Checkpoints are written by VPNetModel::SaveCheckpoint, so the format
//     matches what LoadCheckpoint expects (model .pt + optimizer .pt pair).
//
// Build as an example target guarded by ${OPEN_SPIEL_BUILD_WITH_LIBTORCH};
// see pretrain/README.md.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/time/clock.h"
#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/algorithms/alpha_zero_torch/vpnet.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/utils/file.h"

#include "pgn_atomic.h"
#include "sf_data.h"

ABSL_FLAG(std::string, path, "", "Run directory: holds vpnet.pb + checkpoints.");
ABSL_FLAG(std::string, graph_def, "vpnet.pb", "Model config filename.");
ABSL_FLAG(std::string, device, "/cuda:0", "Torch device.");
ABSL_FLAG(std::string, game, "atomic_chess", "OpenSpiel game name.");

ABSL_FLAG(std::string, pgn, "",
          "Comma-separated plain-text PGN files. Use \"-\" for stdin "
          "(forces epochs=1), e.g. zstdcat *.pgn.zst | az_pretrain --pgn=-");
ABSL_FLAG(std::string, sf_tsv, "",
          "Comma-separated sf_label .tsv files (Fairy-Stockfish distillation). "
          "May be combined with --pgn; both feed the same shuffle buffer.");
ABSL_FLAG(std::string, val_sf_tsv, "", "Held-out sf_label .tsv for validation.");
ABSL_FLAG(double, sf_lambda, 0.7,
          "value = lambda*tanh(cp/sf_cp_scale) + (1-lambda)*game_result.");
ABSL_FLAG(double, sf_cp_scale, 400.0,
          "Centipawns mapped to |value| ~ 0.76. Calibrate for atomic.");
ABSL_FLAG(double, sf_policy_temp, 0.0,
          "Softmax temperature (cp) over MultiPV. 0 = one-hot on SF's best.");
ABSL_FLAG(int, sf_max_abs_cp, 0,
          "Drop positions with |cp| above this (0 = keep all).");
ABSL_FLAG(std::string, val_pgn, "",
          "Held-out PGN for validation. Strongly recommended: the MultiAra "
          "study found 7 epochs generalised better than 30.");
ABSL_FLAG(int, epochs, 7, "Passes over the training PGNs.");
ABSL_FLAG(int, min_elo, 1900,
          "Minimum Elo for BOTH players. 1900 is the atomic threshold for the "
          "top 10th percentile of lichess players used by MultiAra.");
ABSL_FLAG(int, min_plies, 4, "Drop games shorter than this.");
ABSL_FLAG(bool, exclude_time_forfeit, false,
          "Drop games decided on the clock (their result may not match the "
          "position).");
ABSL_FLAG(int64_t, max_games, -1, "Cap games per epoch (-1 = no cap).");

ABSL_FLAG(int, batch_size, 1024, "Training batch size.");
ABSL_FLAG(int, buffer_size, 262144,
          "Shuffle buffer, in positions. ~4.2KB each, so 262144 is ~1.1GB. "
          "Needed because consecutive positions from one game are highly "
          "correlated.");
ABSL_FLAG(int, min_buffer, 32768, "Warm up the buffer before the first step.");
ABSL_FLAG(int, reuse, 4,
          "Times each position is seen on average; a learn step runs every "
          "batch_size/reuse new positions. Matches replay_buffer_reuse.");

ABSL_FLAG(int, log_every, 50, "Log training losses every N steps.");
ABSL_FLAG(int, val_every, 500, "Run validation every N steps (0 = never).");
ABSL_FLAG(int, val_samples, 20000, "Cap on held-out positions kept in memory.");
ABSL_FLAG(int, save_every, 500, "Write checkpoint--1 every N steps.");

ABSL_FLAG(int, init_checkpoint, -2,
          "Checkpoint to start from: -2 = fresh weights, -1 = latest, N = "
          "checkpoint-N. Use -1 to continue an interrupted pretraining run.");
ABSL_FLAG(int, seed, 42, "RNG seed for batch sampling.");
ABSL_FLAG(int, epoch_offset, 0,
          "Added to the epoch number used for per-epoch checkpoint names. When "
          "chaining runs to get around a walltime cap, set this to the number "
          "of epochs already done so run 2 writes checkpoint-3,4,... instead of "
          "overwriting run 1's checkpoint-1,2,...");

// Only used when the run directory has no graph_def yet. Prefer letting
// alpha_zero_torch_example create it (see bootstrap_pretrained_run.sh) so the
// architecture cannot drift from the RL config.
ABSL_FLAG(std::string, nn_model, "resnet", "resnet or mlp.");
ABSL_FLAG(int, nn_width, 128, "Channels per residual block.");
ABSL_FLAG(int, nn_depth, 10, "Residual blocks.");
ABSL_FLAG(double, learning_rate, 0.0002, "Adam learning rate.");
ABSL_FLAG(double, weight_decay, 0.0001, "L2 regularisation.");

namespace az = open_spiel::algorithms::torch_az;
using open_spiel::Action;

namespace {

az::VPNetModel::TrainInputs ToTrainInputs(atomic_az::Sample&& s) {
  // The policy target is one-hot on the human move. VPNetModel::Learn writes
  // these sparse (action, probability) pairs into a dense masked target.
  return az::VPNetModel::TrainInputs{std::move(s.legal_actions),
                                     std::move(s.observation),
                                     std::move(s.policy), s.value};
}

struct ValStats {
  double value_mse = 0;        // vs the (possibly blended) training target
  double value_sign_acc = 0;   // agreement with the target's sign
  double result_mse = 0;       // vs the ACTUAL game outcome
  double result_sign_acc = 0;  // did it predict the real winner?
  double policy_top1 = 0;
  int64_t n = 0;
  int64_t n_decisive = 0;
  int64_t n_result = 0;
};

ValStats Validate(az::VPNetModel* model,
                  const std::vector<atomic_az::Sample>& val, int batch_size) {
  ValStats out;
  if (val.empty()) return out;
  int64_t sign_ok = 0, top1_ok = 0, result_sign_ok = 0;
  double sse = 0, result_sse = 0;

  for (size_t i = 0; i < val.size(); i += batch_size) {
    const size_t end = std::min(val.size(), i + batch_size);
    std::vector<az::VPNetModel::InferenceInputs> inputs;
    inputs.reserve(end - i);
    for (size_t j = i; j < end; ++j) {
      inputs.push_back(
          az::VPNetModel::InferenceInputs{val[j].legal_actions,
                                         val[j].observation});
    }
    std::vector<az::VPNetModel::InferenceOutputs> outputs =
        model->Inference(inputs);
    for (size_t j = 0; j < outputs.size(); ++j) {
      const atomic_az::Sample& s = val[i + j];
      const double diff = outputs[j].value - s.value;
      sse += diff * diff;
      if (s.value != 0.0) {
        out.n_decisive += 1;
        if ((outputs[j].value >= 0) == (s.value >= 0)) sign_ok += 1;
      }
      // With Stockfish distillation `value` is mostly the engine's evaluation,
      // so the target-based numbers above say little about whether the net
      // knows who wins. Score against the raw outcome too; this is the figure
      // comparable to the RL learner's value_accuracy.
      if (s.game_result != 0.0) {
        const double rdiff = outputs[j].value - s.game_result;
        result_sse += rdiff * rdiff;
        out.n_result += 1;
        if ((outputs[j].value >= 0) == (s.game_result >= 0)) result_sign_ok += 1;
      }
      Action best = open_spiel::kInvalidAction;
      double best_p = -1.0;
      for (const auto& [action, prob] : outputs[j].policy) {
        if (prob > best_p) { best_p = prob; best = action; }
      }
      open_spiel::Action target = open_spiel::kInvalidAction;
      double target_p = -1.0;
      for (const auto& [action, prob] : s.policy) {
        if (prob > target_p) { target_p = prob; target = action; }
      }
      if (best == target) top1_ok += 1;
      out.n += 1;
    }
  }
  out.value_mse = sse / out.n;
  out.policy_top1 = static_cast<double>(top1_ok) / out.n;
  out.value_sign_acc = out.n_decisive > 0
                           ? static_cast<double>(sign_ok) / out.n_decisive
                           : 0.0;
  out.result_mse = out.n_result > 0 ? result_sse / out.n_result : 0.0;
  out.result_sign_acc =
      out.n_result > 0 ? static_cast<double>(result_sign_ok) / out.n_result
                       : 0.0;
  return out;
}

std::vector<std::string> SplitFiles(const std::string& csv) {
  std::vector<std::string> out;
  for (absl::string_view piece : absl::StrSplit(csv, ',', absl::SkipEmpty())) {
    out.emplace_back(piece);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string path = absl::GetFlag(FLAGS_path);
  const std::string graph_def = absl::GetFlag(FLAGS_graph_def);
  if (path.empty()) {
    std::cerr << "--path is required." << std::endl;
    return 1;
  }
  std::vector<std::string> pgn_files = SplitFiles(absl::GetFlag(FLAGS_pgn));
  std::vector<std::string> sf_files = SplitFiles(absl::GetFlag(FLAGS_sf_tsv));
  if (pgn_files.empty() && sf_files.empty()) {
    std::cerr << "Need --pgn (human games) and/or --sf_tsv (Stockfish "
                 "distillation). Both feed the same shuffle buffer."
              << std::endl;
    return 1;
  }
  const bool from_stdin = (pgn_files.size() == 1 && pgn_files[0] == "-");

  atomic_az::SfOptions sf_opt;
  sf_opt.lambda = absl::GetFlag(FLAGS_sf_lambda);
  sf_opt.cp_scale = absl::GetFlag(FLAGS_sf_cp_scale);
  sf_opt.policy_temp = absl::GetFlag(FLAGS_sf_policy_temp);
  sf_opt.max_abs_cp = absl::GetFlag(FLAGS_sf_max_abs_cp);

  int epochs = absl::GetFlag(FLAGS_epochs);
  if (from_stdin && epochs != 1) {
    std::cerr << "[pretrain] stdin cannot be re-read; forcing epochs=1."
              << std::endl;
    epochs = 1;
  }

  auto game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game));

  // Create the model config only if absent -- overwriting it after checkpoints
  // exist would silently desynchronise architecture from weights.
  const std::string model_path = absl::StrCat(path, "/", graph_def);
  if (!open_spiel::file::Exists(model_path)) {
    std::cout << "[pretrain] creating " << model_path << " ("
              << absl::GetFlag(FLAGS_nn_model) << " "
              << absl::GetFlag(FLAGS_nn_width) << "x"
              << absl::GetFlag(FLAGS_nn_depth) << ")" << std::endl;
    SPIEL_CHECK_TRUE(az::CreateGraphDef(
        *game, absl::GetFlag(FLAGS_learning_rate),
        absl::GetFlag(FLAGS_weight_decay), path, graph_def,
        absl::GetFlag(FLAGS_nn_model), absl::GetFlag(FLAGS_nn_width),
        absl::GetFlag(FLAGS_nn_depth), /*verbose=*/true));
  } else {
    std::cout << "[pretrain] using existing " << model_path
              << " (architecture flags ignored)" << std::endl;
  }

  az::VPNetModel model(*game, path, graph_def, absl::GetFlag(FLAGS_device));
  const int init_ckpt = absl::GetFlag(FLAGS_init_checkpoint);
  if (init_ckpt != -2) {
    std::cout << "[pretrain] loading checkpoint " << init_ckpt << std::endl;
    model.LoadCheckpoint(init_ckpt);
  }

  atomic_az::Filter filter;
  filter.min_elo = absl::GetFlag(FLAGS_min_elo);
  filter.min_plies = absl::GetFlag(FLAGS_min_plies);
  filter.exclude_time_forfeit = absl::GetFlag(FLAGS_exclude_time_forfeit);

  const int batch_size = absl::GetFlag(FLAGS_batch_size);
  const int min_buffer =
      std::max(absl::GetFlag(FLAGS_min_buffer), batch_size);
  const int samples_per_step =
      std::max(1, batch_size / std::max(1, absl::GetFlag(FLAGS_reuse)));

  // Held-out set, loaded once and kept in memory.
  std::vector<atomic_az::Sample> val;
  const std::string val_pgn = absl::GetFlag(FLAGS_val_pgn);
  if (!val_pgn.empty()) {
    const int cap = absl::GetFlag(FLAGS_val_samples);
    std::ifstream in(val_pgn);
    if (!in) {
      std::cerr << "[pretrain] cannot open --val_pgn " << val_pgn << std::endl;
      return 1;
    }
    atomic_az::Stats s = atomic_az::ReadPgn(
        in, *game, filter, [&](atomic_az::Sample&& x) {
          if (static_cast<int>(val.size()) < cap) val.push_back(std::move(x));
        });
    std::cout << "[pretrain] validation: " << val.size() << " positions ("
              << s.ToString() << ")" << std::endl;
  }
  const std::string val_tsv = absl::GetFlag(FLAGS_val_sf_tsv);
  if (!val_tsv.empty()) {
    const int cap = absl::GetFlag(FLAGS_val_samples);
    std::ifstream in(val_tsv);
    if (!in) {
      std::cerr << "[pretrain] cannot open --val_sf_tsv " << val_tsv << std::endl;
      return 1;
    }
    // Cap the games read, not just the samples kept: the reader used to parse
    // the entire 2.4M-position shard and throw away all but --val_samples,
    // costing ~10 minutes of wall time per job for nothing.
    const int64_t val_games = std::max<int64_t>(1, cap / 25);
    atomic_az::SfStats s = atomic_az::ReadSfTsv(
        in, *game, sf_opt, [&](atomic_az::Sample&& x) {
          if (static_cast<int>(val.size()) < cap) val.push_back(std::move(x));
        }, val_games);
    std::cout << "[pretrain] validation (sf): " << s.ToString() << std::endl;
  }
  if (val.empty()) {
    std::cerr << "[pretrain] WARNING: no validation set (--val_pgn / "
                 "--val_sf_tsv); you will not be able to tell when the net "
                 "starts overfitting." << std::endl;
  }

  open_spiel::CircularBuffer<az::VPNetModel::TrainInputs> buffer(
      absl::GetFlag(FLAGS_buffer_size));
  std::mt19937 rng(absl::GetFlag(FLAGS_seed));

  int64_t step = 0, since_step = 0;
  az::VPNetModel::LossInfo window;
  const absl::Time start = absl::Now();
  const int log_every = absl::GetFlag(FLAGS_log_every);
  const int val_every = absl::GetFlag(FLAGS_val_every);
  const int save_every = absl::GetFlag(FLAGS_save_every);

  auto on_sample = [&](atomic_az::Sample&& s) {
    buffer.Add(ToTrainInputs(std::move(s)));
    if (buffer.Size() < min_buffer) return;
    if (++since_step < samples_per_step) return;
    since_step = 0;
    ++step;

    window += model.Learn(buffer.Sample(&rng, batch_size));

    if (step % log_every == 0) {
      const double secs = absl::ToDoubleSeconds(absl::Now() - start);
      std::cout << "step " << step << "  positions " << buffer.TotalAdded()
                << "  policy " << window.Policy() << "  value "
                << window.Value() << "  l2 " << window.L2() << "  total "
                << window.Total() << "  (" << (buffer.TotalAdded() / secs)
                << " pos/s)" << std::endl;
      window = az::VPNetModel::LossInfo();
    }
    if (val_every > 0 && step % val_every == 0 && !val.empty()) {
      ValStats v = Validate(&model, val, batch_size);
      std::cout << "  VAL step " << step << "  value_mse " << v.value_mse
                << "  value_sign_acc " << v.value_sign_acc
                << "  result_sign_acc " << v.result_sign_acc
                << "  policy_top1 " << v.policy_top1 << "  (n=" << v.n << ")"
                << std::endl;
    }
    if (save_every > 0 && step % save_every == 0) {
      model.SaveCheckpoint(az::VPNetModel::kMostRecentCheckpointStep);
    }
  };

  atomic_az::Stats total;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    for (const std::string& f : pgn_files) {
      atomic_az::Stats s;
      if (from_stdin) {
        s = atomic_az::ReadPgn(std::cin, *game, filter, on_sample,
                               absl::GetFlag(FLAGS_max_games));
      } else {
        std::ifstream in(f);
        if (!in) {
          std::cerr << "[pretrain] cannot open " << f << std::endl;
          return 1;
        }
        s = atomic_az::ReadPgn(in, *game, filter, on_sample,
                               absl::GetFlag(FLAGS_max_games));
      }
      std::cout << "[pretrain] epoch " << epoch << " file " << f << ": "
                << s.ToString() << std::endl;
      total.games_seen += s.games_seen;
      total.games_used += s.games_used;
      total.samples += s.samples;
      total.skipped_variant += s.skipped_variant;
      total.skipped_elo += s.skipped_elo;
      total.skipped_result += s.skipped_result;
      total.skipped_termination += s.skipped_termination;
      total.skipped_short += s.skipped_short;
      total.skipped_parse += s.skipped_parse;
    }
    for (const std::string& f : sf_files) {
      std::ifstream in(f);
      if (!in) {
        std::cerr << "[pretrain] cannot open " << f << std::endl;
        return 1;
      }
      atomic_az::SfStats s = atomic_az::ReadSfTsv(
          in, *game, sf_opt, on_sample, absl::GetFlag(FLAGS_max_games));
      std::cout << "[pretrain] epoch " << epoch << " file " << f << ": "
                << s.ToString() << std::endl;
      total.games_seen += s.lines;
      total.games_used += s.games_ok;
      total.samples += s.samples;
      total.skipped_parse += s.games_bad;
    }
    // A checkpoint per epoch, so a run interrupted mid-schedule is still usable
    // and so you can pick the epoch with the best validation numbers.
    const int epoch_label = epoch + absl::GetFlag(FLAGS_epoch_offset);
    const std::string ck = model.SaveCheckpoint(epoch_label);
    std::cout << "[pretrain] epoch " << epoch_label << " done, saved " << ck
              << std::endl;
    if (!val.empty()) {
      ValStats v = Validate(&model, val, batch_size);
      std::cout << "[pretrain] EPOCH " << epoch_label << " VAL  value_mse "
                << v.value_mse << "  value_sign_acc " << v.value_sign_acc
                << "  result_mse " << v.result_mse
                << "  result_sign_acc " << v.result_sign_acc
                << "  policy_top1 " << v.policy_top1 << std::endl;
    }
  }

  if (total.games_used == 0) {
    std::cerr << "[pretrain] ERROR: no games were usable. Check --min_elo, the "
                 "[Variant] tag, and that the PGN is decompressed."
              << std::endl;
    return 1;
  }

  // checkpoint--1 is what a resuming alpha_zero_torch_example loads.
  const std::string final_ck =
      model.SaveCheckpoint(az::VPNetModel::kMostRecentCheckpointStep);
  std::cout << "[pretrain] TOTAL " << total.ToString() << std::endl;
  std::cout << "[pretrain] learn steps: " << step << std::endl;
  std::cout << "[pretrain] wrote " << final_ck << ".pt (+ -optimizer.pt)"
            << std::endl;
  std::cout << "[pretrain] RL will resume from this net if " << path
            << "/config.json and a non-empty learner.jsonl exist -- see "
               "bootstrap_pretrained_run.sh"
            << std::endl;
  return 0;
}
