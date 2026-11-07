// PerfectMahjong CPP — Chinese Standard Mahjong (MCR) self-play PPO + Omniscient Oracle training entry point.
//
// Example:
//   perfectmahjong_train --backend mps --updates 200 --games 128 --threads 8 --out ckpt_mcr
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/version.h>

#include "algo/eval_mcr.h"
#include "algo/ppo.h"
#include "algo/rollout.h"
#include "mcr/oracle.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

namespace {

struct Args {
    int updates = 10000;
    int startUpdate = 1;
    int games = 128;
    int threads = std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, 16);
    int hidden = 256;
    int lstmHidden = 128;
    int epochs = 2;
    int minibatch = 2048;
    int snapshotEvery = 50;
    float lr = 3e-4f;
    float clip = 0.2f;
    float ent = 0.02f;
    float gamma = 0.995f;
    float lambda = 0.95f;
    float targetKL = 0.03f;
    bool oracleReward = true;
    float oracleCoef = 0.1f;
    float rewardScale = 0.05f;
    int poolSize = 16;
    int archiveSize = 16;
    int poolEvery = 20;
    float histProb = 0.2f;
    float heuristicProb = 0.1f;
    uint64_t seed = 1;
    std::string out = "ckpt_mcr";
    std::string resume;
    std::string backend = "cpu";   // cuda | mps | cpu (explicit, no auto-detection)
    int evalEvery = 100;           // evaluate every N updates
    int evalDecks = 25;            // 25 duplicate walls = 100 games
    int evalPort = 18766;          // Botzone MCR TCP server port
    std::string evalHost = "127.0.0.1";
    std::string evalMode = "auto"; // tcp | heuristic | auto
    std::string evalCsv;           // empty = default to <out>/eval_vs_botzone.csv
    std::string evalSaveDir;       // empty = default to <out>/eval_snapshots
    bool evalSave = true;
    int evalMaxSnapshots = 0;
    bool evalAsync = true;
};

const char* argValue(int argc, char** argv, const char* key, const char* def) {
    std::string k = key;
    for (int i = 1; i + 1 < argc; ++i)
        if (k == argv[i]) return argv[i + 1];
    return def;
}

bool hasFlag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i)
        if (key == std::string(argv[i])) return true;
    return false;
}

void parseArgs(int argc, char** argv, Args& a) {
    a.updates = std::atoi(argValue(argc, argv, "--updates", std::to_string(a.updates).c_str()));
    a.startUpdate = std::atoi(argValue(argc, argv, "--start-update", "1"));
    a.games = std::atoi(argValue(argc, argv, "--games", std::to_string(a.games).c_str()));
    a.threads = std::atoi(argValue(argc, argv, "--threads", std::to_string(a.threads).c_str()));
    a.hidden = std::atoi(argValue(argc, argv, "--hidden", std::to_string(a.hidden).c_str()));
    a.lstmHidden = std::atoi(argValue(argc, argv, "--lstm-hidden", std::to_string(a.lstmHidden).c_str()));
    a.epochs = std::atoi(argValue(argc, argv, "--epochs", std::to_string(a.epochs).c_str()));
    a.minibatch = std::atoi(argValue(argc, argv, "--mb", std::to_string(a.minibatch).c_str()));
    a.snapshotEvery = std::atoi(argValue(argc, argv, "--snapshot-every", std::to_string(a.snapshotEvery).c_str()));
    a.lr = float(std::atof(argValue(argc, argv, "--lr", "3e-4")));
    a.clip = float(std::atof(argValue(argc, argv, "--clip", "0.2")));
    a.ent = float(std::atof(argValue(argc, argv, "--ent", "0.02")));
    a.gamma = float(std::atof(argValue(argc, argv, "--gamma", "0.995")));
    a.lambda = float(std::atof(argValue(argc, argv, "--lambda", "0.95")));
    a.targetKL = float(std::atof(argValue(argc, argv, "--target-kl", "0.03")));
    if (hasFlag(argc, argv, "--oracle-reward")) a.oracleReward = true;
    if (hasFlag(argc, argv, "--no-oracle-reward")) a.oracleReward = false;
    a.oracleCoef = float(std::atof(argValue(argc, argv, "--oracle-coef", "0.1")));
    a.rewardScale = float(std::atof(argValue(argc, argv, "--reward-scale", "0.05")));
    a.poolSize = std::atoi(argValue(argc, argv, "--pool-size", "16"));
    a.archiveSize = std::atoi(argValue(argc, argv, "--archive-size", "16"));
    a.poolEvery = std::atoi(argValue(argc, argv, "--pool-every", "20"));
    a.histProb = float(std::atof(argValue(argc, argv, "--historical-prob", "0.2")));
    a.heuristicProb = float(std::atof(argValue(argc, argv, "--heuristic-prob", "0.1")));
    a.seed = std::atoll(argValue(argc, argv, "--seed", "1"));
    a.out = argValue(argc, argv, "--out", a.out.c_str());
    a.backend = argValue(argc, argv, "--backend", a.backend.c_str());
    const char* res = argValue(argc, argv, "--resume", "");
    if (*res) a.resume = res;
    a.evalEvery = std::atoi(argValue(argc, argv, "--eval-every", std::to_string(a.evalEvery).c_str()));
    if (hasFlag(argc, argv, "--no-eval")) {
        a.evalEvery = 0;
    }
    a.evalDecks = std::atoi(argValue(argc, argv, "--eval-decks", std::to_string(a.evalDecks).c_str()));
    a.evalPort = std::atoi(argValue(argc, argv, "--eval-port", "18766"));
    a.evalHost = argValue(argc, argv, "--eval-host", "127.0.0.1");
    a.evalMode = argValue(argc, argv, "--eval-mode", a.evalMode.c_str());
    a.evalCsv = argValue(argc, argv, "--eval-csv", "");
    a.evalSaveDir = argValue(argc, argv, "--eval-save-dir", "");
    a.evalMaxSnapshots = std::atoi(argValue(argc, argv, "--eval-max-snapshots", "0"));
    if (hasFlag(argc, argv, "--no-eval-save")) {
        a.evalSave = false;
    }
    if (hasFlag(argc, argv, "--no-eval-async") || hasFlag(argc, argv, "--eval-sync")) {
        a.evalAsync = false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    parseArgs(argc, argv, args);
    if (args.threads <= 0) args.threads = 1;

    torch::Device torchDevice = torch::kCPU;
    std::string deviceName = "CPU";
    if (args.backend == "cuda") {
        if (!torch::cuda::is_available()) {
            std::fprintf(stderr,
                         "Backend 'cuda' requested but CUDA is not available in this LibTorch build. "
                         "Pass --backend mps or --backend cpu explicitly.\n");
            return 1;
        }
        torchDevice = torch::kCUDA;
        deviceName = "GPU (CUDA)";
    } else if (args.backend == "mps") {
#if defined(__APPLE__) && (TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12))
        if (!torch::mps::is_available()) {
            std::fprintf(stderr,
                         "Backend 'mps' requested but MPS is not available. Pass --backend cpu explicitly.\n");
            return 1;
        }
        torchDevice = torch::kMPS;
        deviceName = "GPU (MPS)";
#else
        std::fprintf(stderr,
                     "Backend 'mps' requested but MPS is not supported on this platform/LibTorch build. "
                     "Pass --backend cpu explicitly.\n");
        return 1;
#endif
    } else if (args.backend == "cpu") {
        torchDevice = torch::kCPU;
        deviceName = "CPU";
    } else {
        std::fprintf(stderr,
                     "Unknown or unsupported backend '%s'. Valid values: cuda | mps | cpu.\n",
                     args.backend.c_str());
        return 1;
    }

    std::cout << "PerfectMahjong CPP training (Chinese Standard Mahjong MCR 81-Fan + Omniscient Oracle)\n";
    std::cout << "  Compute Device: " << deviceName << "\n";
    if (!args.resume.empty() && !hasFlag(argc, argv, "--start-update")) {
        std::string metaPath = args.resume + "/meta.txt";
        std::ifstream metaFile(metaPath);
        if (metaFile.is_open()) {
            std::string key;
            int val = 0;
            while (metaFile >> key >> val) {
                if (key == "update" || key == "updates") {
                    args.startUpdate = val + 1;
                }
            }
        }
    }

    std::cout << "  updates=" << args.updates
              << (args.startUpdate > 1 ? " (start=" + std::to_string(args.startUpdate) + ")" : "")
              << " games/upd=" << args.games
              << " threads=" << args.threads
              << " mb=" << args.minibatch
              << " epochs=" << args.epochs
              << " hidden=" << args.hidden
              << " lstm=" << args.lstmHidden
              << " lr=" << args.lr
              << " clip=" << args.clip
              << " ent=" << args.ent
              << " gae(gamma=" << args.gamma << ",lambda=" << args.lambda << ")"
              << " oracle=" << (args.oracleReward ? ("yes(coef=" + std::to_string(args.oracleCoef) + ")") : "no")
              << " targetKL=" << args.targetKL
              << " league(recent=" << args.poolSize << ",archive=" << args.archiveSize
              << ",every=" << args.poolEvery
              << ",hist=" << args.histProb << ",heur=" << args.heuristicProb << ")"
              << " seed=" << args.seed << std::endl;

    if (args.evalEvery > 0) {
        if (args.evalCsv.empty()) {
            args.evalCsv = args.out + "/eval_vs_botzone.csv";
        }
        if (args.evalSave && args.evalSaveDir.empty()) {
            args.evalSaveDir = args.out + "/eval_snapshots";
        }
        std::cout << "  eval vs MCR Baseline (" << args.evalMode << "): every=" << args.evalEvery
                  << " decks=" << args.evalDecks << " (" << (args.evalDecks * 4) << " duplicate games)"
                  << " target=" << args.evalHost << ":" << args.evalPort
                  << " async=" << (args.evalAsync ? "yes" : "no")
                  << " csv=" << args.evalCsv << std::endl;
    }
    std::thread evalThread;

    torch_nn::ModelConfig torchModelCfg;
    torchModelCfg.hidden = args.hidden;
    torchModelCfg.lstmHidden = args.lstmHidden;

    torch_nn::PPOConfig torchPpoCfg;
    torchPpoCfg.lr = args.lr;
    torchPpoCfg.clip = args.clip;
    torchPpoCfg.entCoef = args.ent;
    torchPpoCfg.epochs = args.epochs;
    torchPpoCfg.minibatch = args.minibatch;
    torchPpoCfg.targetKL = args.targetKL;
    torchPpoCfg.criticMaxGradNorm = 5.0f;

    torch_nn::Actor actor(torchModelCfg);
    torch_nn::Critic critic(torchModelCfg);

    // Dedicated CPU model instances for multithreaded CPU rollout simulation
    torch_nn::Actor cpuActor(torchModelCfg);
    torch_nn::Critic cpuCritic(torchModelCfg);

    auto syncWeightsToCpu = [&]() {
        torch::NoGradGuard no_grad;
        auto sa = actor->parameters();
        auto da = cpuActor->parameters();
        for (size_t i = 0; i < sa.size(); ++i) da[i].copy_(sa[i]);
        auto sc = critic->parameters();
        auto dc = cpuCritic->parameters();
        for (size_t i = 0; i < sc.size(); ++i) dc[i].copy_(sc[i]);
        cpuActor->eval();
        cpuCritic->eval();
    };

    if (!args.resume.empty()) {
        std::string actPath = args.resume + "/actor.pt";
        std::string critPath = args.resume + "/critic.pt";
        if (std::filesystem::exists(actPath)) actor->load(actPath);
        if (std::filesystem::exists(critPath)) critic->load(critPath);
    }

    auto learner = std::make_unique<torch_nn::PPOLearner>(actor, critic, torchPpoCfg, torchDevice);
    if (!args.resume.empty()) {
        std::string optPath = args.resume + "/opt.pt";
        if (std::filesystem::exists(optPath)) learner->loadOptimizer(optPath);
    }

    syncWeightsToCpu();

    std::shared_ptr<algo::HistoricalPool> histPool = std::make_shared<algo::HistoricalPool>();

    auto pushHistPool = [&](torch_nn::Actor& srcActor, int currentUpdate) {
        auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
        snap->update = currentUpdate;
        snap->actor = torch_nn::Actor(torchModelCfg);
        {
            torch::NoGradGuard no_grad;
            auto srcParams = srcActor->parameters();
            auto dstParams = snap->actor->parameters();
            for (size_t i = 0; i < srcParams.size(); ++i) {
                dstParams[i].copy_(srcParams[i]);
            }
            snap->actor->eval();
        }
        int k = args.poolEvery > 0 ? (currentUpdate / args.poolEvery) : 0;
        bool isPowerOfTwo = (k > 0) && ((k & (k - 1)) == 0);
        bool isCentennial = (currentUpdate % 100 == 0);
        if (args.archiveSize > 0 && (isPowerOfTwo || isCentennial || histPool->archive.empty())) {
            while ((int)histPool->archive.size() >= args.archiveSize && !histPool->archive.empty()) {
                if (histPool->archive.size() > 2) {
                    histPool->archive.erase(histPool->archive.begin() + 1);
                } else {
                    histPool->archive.erase(histPool->archive.begin());
                }
            }
            histPool->archive.push_back(snap);
        }
        if (args.poolSize > 0) {
            while ((int)histPool->recent.size() >= args.poolSize && !histPool->recent.empty()) {
                histPool->recent.erase(histPool->recent.begin());
            }
            histPool->recent.push_back(std::move(snap));
        }
    };

    auto saveHistoricalPool = [&](const std::string& dir) {
        if (!histPool || histPool->empty()) return;
        std::string poolDir = dir + "/hist_pool";
        std::filesystem::create_directories(poolDir);
        std::ofstream meta(poolDir + "/pool_meta.txt");
        if (!meta.is_open()) return;

        meta << "recent_count " << histPool->recent.size() << "\n";
        for (size_t i = 0; i < histPool->recent.size(); ++i) {
            const auto& snap = histPool->recent[i];
            meta << "recent " << i << " " << snap->update << "\n";
            snap->actor->save(poolDir + "/recent_" + std::to_string(i) + "_actor.pt");
        }
        meta << "archive_count " << histPool->archive.size() << "\n";
        for (size_t i = 0; i < histPool->archive.size(); ++i) {
            const auto& snap = histPool->archive[i];
            meta << "archive " << i << " " << snap->update << "\n";
            snap->actor->save(poolDir + "/archive_" + std::to_string(i) + "_actor.pt");
        }
    };

    auto loadHistoricalPool = [&](const std::string& dir) {
        std::string metaPath = dir + "/hist_pool/pool_meta.txt";
        std::ifstream meta(metaPath);
        if (!meta.is_open()) return;
        std::string tag;
        while (meta >> tag) {
            if (tag == "recent" || tag == "archive") {
                size_t idx = 0;
                int snapUpd = 0;
                meta >> idx >> snapUpd;
                auto snap = std::make_shared<algo::HistoricalActorSnapshot>();
                snap->update = snapUpd;
                snap->actor = torch_nn::Actor(torchModelCfg);
                std::string actPath = dir + "/hist_pool/" + tag + "_" + std::to_string(idx) + "_actor.pt";
                if (std::filesystem::exists(actPath)) snap->actor->load(actPath);
                snap->actor->eval();
                if (tag == "recent") histPool->recent.push_back(snap);
                else histPool->archive.push_back(snap);
            } else {
                int dummy = 0;
                meta >> dummy;
            }
        }
        std::cout << "  Restored historical pool: recent=" << histPool->recent.size()
                  << " archive=" << histPool->archive.size() << "\n";
    };

    if (!args.resume.empty()) {
        loadHistoricalPool(args.resume);
    }

    auto saveAll = [&](const std::string& dir, int currentUpd = 0) {
        std::filesystem::create_directories(dir);
        actor->save(dir + "/actor.pt");
        critic->save(dir + "/critic.pt");
        learner->saveOptimizer(dir + "/opt.pt");
        if (currentUpd > 0) {
            std::ofstream metaFile(dir + "/meta.txt");
            if (metaFile.is_open()) {
                metaFile << "update " << currentUpd << "\n"
                         << "seed " << args.seed << "\n"
                         << "lr " << args.lr << "\n"
                         << "ent " << args.ent << "\n"
                         << "oracle_reward " << (args.oracleReward ? 1 : 0) << "\n"
                         << "oracle_coef " << args.oracleCoef << "\n";
            }
            saveHistoricalPool(dir);
        }
    };

    std::filesystem::create_directories(args.out);
    saveAll(args.out, 0);

    auto trainStartWall = std::chrono::steady_clock::now();

    for (int upd = args.startUpdate; upd <= args.updates; ++upd) {
        auto wall0 = std::chrono::steady_clock::now();

        // 1. Synchronous CPU self-play rollout
        algo::ModelSet models{&cpuActor, &cpuCritic};
        algo::RolloutConfig rc;
        rc.gamesPerUpdate = args.games;
        rc.threads = args.threads;
        rc.seed = args.seed + uint64_t(upd) * 7919ULL;
        rc.historicalProb = args.histProb;
        rc.heuristicProb = args.heuristicProb;
        rc.historicalPool = histPool.get();
        rc.oracleReward = args.oracleReward;
        rc.oracleCoef = args.oracleCoef;
        rc.rewardScale = args.rewardScale;

        std::vector<algo::Transition> transitions;
        algo::RolloutStats rs;
        algo::collectRollout(models, rc, transitions, rs);

        // 2. Synchronous GPU/CPU PPO update
        auto tLearn0 = std::chrono::steady_clock::now();
        torch_nn::PPOStats ts{};
        if (!transitions.empty()) {
            try {
                auto batchTensors = torch_nn::transitionsToBatch(
                    transitions, args.gamma, args.lambda, torchDevice);
                learner->setLr(args.lr);
                learner->setEntCoef(args.ent);
                ts = learner->update(batchTensors);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "PPO update error: %s\n", e.what());
            }
        }
        double learnSecs = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - tLearn0)
                               .count();

        // 3. Sync freshly trained weights back to CPU rollout model
        syncWeightsToCpu();

        double wallSecs = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - wall0)
                              .count();
        double elapsedMin = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - trainStartWall)
                                .count() / 60.0;

        double winRate = double(rs.wins) / std::max(1, rs.games);
        double avgFan = rs.wins > 0 ? double(rs.totalFan) / rs.wins : 0.0;
        double mpg = double(rs.totalMoves) / std::max(1, rs.games);
        std::string leagueStr;
        if (rs.latestGames > 0) {
            char lbuf[64];
            std::snprintf(lbuf, sizeof(lbuf), " | vsLeague WP %.3f ADP %+.2f [%d]",
                          double(rs.latestWins) / rs.latestGames,
                          rs.latestScore / rs.latestGames,
                          rs.latestGames);
            leagueStr = lbuf;
        }

        std::printf(
            "upd %4d | wall %.1fs learn %.2fs games %d mb %d ep %d/%d | "
            "huRate %.3f avgFan %.1f moves/g %.1f%s | "
            "lr %.2e ent %.3f vL %.3f kl %.4f cf %.2f | trans %zu\n",
            upd, wallSecs, learnSecs, rs.games, args.minibatch,
            ts.epochsCompleted, args.epochs,
            winRate, avgFan, mpg, leagueStr.c_str(),
            args.lr, ts.entropy, ts.vLoss, ts.kl, ts.clipFraction,
            transitions.size());
        std::fflush(stdout);

        if (args.poolEvery > 0 && upd % args.poolEvery == 0) {
            pushHistPool(cpuActor, upd);
        }
        if (args.snapshotEvery > 0 &&
            (upd % args.snapshotEvery == 0 || upd == args.updates)) {
            saveAll(args.out, upd);
        }

        if (args.evalEvery > 0 &&
            (upd % args.evalEvery == 0 || upd == args.updates)) {
            if (evalThread.joinable()) {
                evalThread.join();
            }
            auto cloned = algo::cloneActor(cpuActor);
            algo::McrEvalConfig evalCfg;
            evalCfg.mode = args.evalMode;
            evalCfg.host = args.evalHost;
            evalCfg.port = args.evalPort;
            evalCfg.decks = args.evalDecks;
            evalCfg.seed = 2026;
            evalCfg.csvPath = args.evalCsv;
            evalCfg.label = "u" + std::to_string(upd);
            evalCfg.saveDir = args.evalSave ? (args.evalSaveDir + "/" + evalCfg.label) : "";
            evalCfg.maxSnapshots = args.evalMaxSnapshots;
            evalCfg.update = upd;
            evalCfg.elapsedMinutes = elapsedMin;
            evalCfg.verbose = false;

            auto runEval = [cloned = std::move(cloned), evalCfg]() {
                auto res = algo::evaluateAgainstMcrBaseline(cloned, evalCfg);
                if (res.ok) {
                    std::printf("[eval %s vs %s] games %d | WP %.3f ADP %+.2f (delta %+.2f) | DealIn %.3f SelfDraw %.3f AvgFan %.1f\n",
                                evalCfg.label.c_str(), res.baselineName.c_str(),
                                res.games, res.wp, res.adp, res.scoreDelta,
                                res.dealInRate, res.selfDrawRate, res.avgFan);
                } else {
                    std::printf("[eval %s] skipped: %s\n", evalCfg.label.c_str(), res.error.c_str());
                }
                std::fflush(stdout);
            };

            if (args.evalAsync) {
                evalThread = std::thread(std::move(runEval));
            } else {
                runEval();
            }
        }
    }

    if (evalThread.joinable()) evalThread.join();

    saveAll(args.out, args.updates);
    std::cout << "models saved to " << args.out << "/\n";
    return 0;
}
