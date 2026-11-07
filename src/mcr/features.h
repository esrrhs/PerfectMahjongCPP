#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "mcr/game.h"
#include "mcr/oracle.h"
#include "mcr/tiles.h"

namespace mcr {

// Actor imperfect-information state dimensions
constexpr int kNodeBin = 43 * kNumTileTypes;      // 1462 binary features
constexpr int kImpScalars = 58;                   // 58 scalar features (incl. 4-player hand counts)
constexpr int kNodeSize = kNodeBin + kImpScalars; // 1520 total static actor features

// LSTM discard & meld sequence dimensions
constexpr int kLstmSteps = 24;                    // last 24 public discard/meld events
constexpr int kLstmIn = 88;                       // 88 features per event step

// Critic "开天眼" omniscient oracle dimensions
constexpr int kExtraBin = 17 * kNumTileTypes;     // 578 binary oracle features
constexpr int kExtraScalars = 34;                 // 34 scalar oracle features
constexpr int kExtraSize = kExtraBin + kExtraScalars; // 612 total oracle features

// Candidate action feature dimension
constexpr int kActionSize = 96;

struct LegalOption {
    int abstractId = 0;
    Action concrete;
    std::array<float, kActionSize> feature{};
};

struct EncodedState {
    std::array<uint8_t, kNodeBin> imp{};
    std::array<float, kImpScalars> scalar{};
    std::array<float, kLstmSteps * kLstmIn> seq{};
    std::array<uint8_t, kExtraBin> extra{};
    std::array<float, kExtraScalars> extraScalar{};
};

// Encodes both imperfect-information state (for Actor) and omniscient Oracle state (for Critic)
EncodedState encodeState(const Game& g);

// Generates all legal options with their 96-dim post-action features
void legalOptions(const Game& g, std::vector<LegalOption>& out);
std::vector<LegalOption> legalOptions(const Game& g);

// Computes omniscient Oracle potential advantage for `player` (used for dense step reward shaping):
//   Phi(player) - mean_{opp != player} Phi(opp)
float oraclePotentialAdvantage(const Game& g, int player);

}  // namespace mcr
