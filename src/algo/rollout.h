#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "mcr/features.h"
#include "mcr/game.h"
#include "torch_nn/models.h"

namespace algo {

struct Transition {
    std::array<uint8_t, mcr::kNodeBin> imp{};
    std::array<float, mcr::kImpScalars> scalar{};
    std::array<float, mcr::kLstmSteps * mcr::kLstmIn> seq{};
    std::array<uint8_t, mcr::kExtraBin> extra{};
    std::array<float, mcr::kExtraScalars> extraScalar{};
    std::vector<std::pair<int, std::array<float, mcr::kActionSize>>> actions;
    int action = -1;
    int trajectoryId = -1; // unique per (gameId, seat): gameId * 4 + seat
    int seat = 0;
    bool terminal = false;
    float logp = 0.0f;
    float value = 0.0f;
    float reward = 0.0f;
    float adv = 0.0f;
    float ret = 0.0f;
};

struct RolloutStats {
    int games = 0;
    int wins = 0;            // games with a winner (non-draw)
    int selfDrawnWins = 0;   // games won by Tsumo (self-drawn)
    long long totalFan = 0;  // sum of winning fan across won games
    long long totalMoves = 0;
    int pureSelfPlayGames = 0;
    int latestGames = 0;     // seat-games played by LATEST policy against historical/heuristic
    int latestWins = 0;
    double latestScore = 0.0;
    long long transitions = 0;
};

struct ModelSet {
    torch_nn::Actor* actor = nullptr;
    torch_nn::Critic* critic = nullptr;
};

struct HistoricalActorSnapshot {
    torch_nn::Actor actor{nullptr};
    int update = 0;
};

struct HistoricalPool {
    std::vector<std::shared_ptr<HistoricalActorSnapshot>> recent;
    std::vector<std::shared_ptr<HistoricalActorSnapshot>> archive;

    bool empty() const {
        return recent.empty() && archive.empty();
    }
};

struct RolloutConfig {
    int gamesPerUpdate = 128;
    int threads = 4;
    uint64_t seed = 1;
    float historicalProb = 0.2f;
    float heuristicProb = 0.1f; // probability of facing built-in heuristic baseline in league
    const HistoricalPool* historicalPool = nullptr;
    bool oracleReward = true;
    float oracleCoef = 0.1f;
    float rewardScale = 0.05f;  // scales MCR terminal points (e.g. +-32 pts -> +-1.6) for stable value learning
};

void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::vector<Transition>& out,
                    RolloutStats& stats);

}  // namespace algo
