#pragma once

#include <array>
#include <vector>

#include "mcr/tiles.h"

namespace mcr {

struct FanContext {
    int playerId = 0;       // 0..3
    int seatWind = 0;       // 0..3 (East, South, West, North)
    int prevalentWind = 0;  // 0..3
    bool selfDrawn = false;
    bool fourthTile = false;
    bool aboutKong = false;
    bool wallLast = false;
    int flowerCount = 0;
};

struct HandPotential {
    int shanten = 6;
    int usefulTypes = 0;
    int usefulRemaining = 0;
    int qualifyingWaits = 0;
    float expectedFan = 0.0f;
    float structure = 0.0f;
    std::array<bool, kNumTileTypes> usefulMask{};
};

// Computes exact shanten (-1 = win, 0 = tenpai, 1..6 = shanten) and optionally
// fills usefulOut[0..33] with tiles that strictly reduce shanten (or win if shanten==0).
// Supports both 3k+1 (13/10/7/4/1) standing tiles and 3k+2 (14/11/8/5/2) standing tiles.
int computeShanten(const TileCounts& standingCounts,
                   const std::vector<Meld>& melds,
                   std::array<bool, kNumTileTypes>* usefulOut = nullptr);

// Checks if the given standingCounts (which ALREADY includes winTile, so total is 3k+2)
// forms a complete hand (standard 4 melds + 1 pair, 7 pairs, 13 orphans, knitted, etc.).
bool isCompleteHand(const TileCounts& standingCountsWithWinTile,
                    const std::vector<Meld>& melds,
                    int winTile);

// Computes official MCR fan count using mahjong::calculate_fan.
// Note: standingCountsWithWinTile ALREADY includes winTile (total 3k+2 tiles).
// Returns 0 if the hand is not a valid winning hand.
int computeFan(const TileCounts& standingCountsWithWinTile,
               const std::vector<Meld>& melds,
               int winTile,
               const FanContext& ctx);

// Returns true iff the hand is a valid winning hand AND its official fan >= minFan.
bool canHu(const TileCounts& standingCountsWithWinTile,
           const std::vector<Meld>& melds,
           int winTile,
           const FanContext& ctx,
           int minFan = 8);

// Evaluates hand potential on a 3k+1 (13-tile equivalent) standing hand after an action,
// given all publicly visible tiles (discards + all melds, excluding self concealed hand).
HandPotential evaluatePotential(const TileCounts& standingCountsAfter,
                                const std::vector<Meld>& meldsAfter,
                                const TileCounts& publicVisibleCounts,
                                int playerId,
                                int prevalentWind,
                                int minFan = 8);

}  // namespace mcr
