// Play a saved PerfectMahjong Actor against the Botzone MCR TCP server or C++ Heuristic baseline.
//
// Example:
//   perfectmahjong_eval --models ckpt_mcr --decks 25 --mode tcp --port 18766
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "algo/eval_mcr.h"
#include "torch_nn/models.h"

namespace {

std::string argValue(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    std::string models = argValue(argc, argv, "--models", "");
    int decks = std::atoi(argValue(argc, argv, "--decks", "25").c_str());
    std::string mode = argValue(argc, argv, "--mode", "auto");
    std::string host = argValue(argc, argv, "--host", "127.0.0.1");
    int port = std::atoi(argValue(argc, argv, "--port", "18766").c_str());
    std::string label = argValue(argc, argv, "--label", "snap");
    std::string csv = argValue(argc, argv, "--csv", "");
    uint64_t seed = std::strtoull(argValue(argc, argv, "--seed", "2026").c_str(), nullptr, 10);

    if (models.empty() || decks <= 0) {
        std::fprintf(stderr,
                     "usage: perfectmahjong_eval --models DIR --decks N [--mode tcp|heuristic|auto] "
                     "[--port P] [--host H] [--hidden H] [--lstm-hidden L] "
                     "[--csv FILE] [--label L] [--seed S]\n");
        return 2;
    }

    torch_nn::ModelConfig mcfg;
    mcfg.hidden = std::atoi(argValue(argc, argv, "--hidden", std::to_string(mcfg.hidden).c_str()).c_str());
    mcfg.lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", std::to_string(mcfg.lstmHidden).c_str()).c_str());

    torch_nn::Actor actor(mcfg);
    std::error_code ec;
    std::string pt = models + "/actor.pt";
    std::string bin = models + "/actor.bin";
    if (std::filesystem::exists(pt, ec)) {
        actor->load(pt);
    } else if (std::filesystem::exists(bin, ec)) {
        actor->load(bin);
    } else if (std::filesystem::exists(models, ec) && !std::filesystem::is_directory(models, ec)) {
        actor->load(models);
    } else {
        std::fprintf(stderr, "cannot find actor.pt or actor.bin in %s\n", models.c_str());
        return 1;
    }
    actor->eval();

    algo::McrEvalConfig cfg;
    cfg.mode = mode;
    cfg.host = host;
    cfg.port = port;
    cfg.decks = decks;
    cfg.seed = seed;
    cfg.csvPath = csv;
    cfg.label = label;
    cfg.verbose = true;

    auto res = algo::evaluateAgainstMcrBaseline(actor, cfg);
    if (!res.ok) {
        std::fprintf(stderr, "evaluation failed: %s\n", res.error.c_str());
        return 1;
    }

    std::printf(
        "done games %d (vs %s) | WP %.3f ADP %+.2f (delta %+.2f) | DealIn %.3f SelfDraw %.3f AvgFan %.1f\n",
        res.games, res.baselineName.c_str(), res.wp, res.adp, res.scoreDelta,
        res.dealInRate, res.selfDrawRate, res.avgFan);
    return 0;
}
