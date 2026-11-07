#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "mcr/oracle.h"
#include "mcr/tiles.h"

namespace mcr {

enum class Phase : int8_t {
    Draw = 1,
    Discard = 2,
    Claim = 3,
    Terminal = 4
};

enum class EventKind : int8_t {
    DRAW = 1,
    PLAY = 2,
    CHI = 3,
    PENG = 4,
    GANG = 5,
    ANGANG = 6,
    BUGANG = 7,
    HU = 8,
    DRAW_GAME = 10
};

struct GameEvent {
    EventKind kind = EventKind::DRAW;
    int8_t player = -1;
    int8_t tile = -1;
    int8_t discard = -1;
    int8_t loser = -1;
    int16_t fan = 0;
    bool tsumogiri = false;
    bool claimed = false;
    std::array<int8_t, 4> handCounts{13, 13, 13, 13};
    int16_t wallRemaining = 84;
};

struct Game {
    int minFan = 8;
    std::vector<int8_t> wall;
    std::array<TileCounts, kNumPlayers> hand{};
    std::array<std::vector<Meld>, kNumPlayers> melds{};
    std::array<std::vector<int8_t>, kNumPlayers> discards{};
    std::array<std::vector<bool>, kNumPlayers> discardTsumogiri{};
    std::array<std::vector<bool>, kNumPlayers> discardClaimed{};
    std::vector<GameEvent> events;

    int currentPlayer = 0;
    int prevalentWind = 0;
    Phase phase = Phase::Draw;
    int drawnTile = -1;
    int lastDiscardPlayer = -1;
    int lastDiscardTile = -1;

    std::vector<int> pendingClaimers;
    std::array<Action, kNumPlayers> claimResponses{};
    std::array<bool, kNumPlayers> hasClaimResponse{};

    bool over = false;
    int winner = -1;
    int loser = -1;
    int fanCount = 0;
    bool selfDrawn = false;
    std::array<int, kNumPlayers> scores{0, 0, 0, 0};

    bool aboutKong = false;
    bool claimHuOnly = false;
    int pendingBugangPlayer = -1;
    int pendingBugangTile = -1;
    int turnCount = 0;

    void deal(Rng& rng, int wind = -1);
    void resetWithWall(const std::vector<int8_t>& customWall, int wind = 0);

    std::vector<Action> legalActions(int playerId = -1) const;
    void step(const Action& action);

    FanContext makeFanContext(int player, bool isSelfDrawn) const;
    TileCounts publicVisibleCounts() const;
    int concealedCount(int player) const { return hand[player].total(); }

private:
    void drawNextTile();
    std::vector<Action> claimActionsFor(int player) const;
    void recordClaim(const Action& action);
    void resolveClaims();
    void finishHu(int player, bool isSelfDrawn);
    void advanceAutoPassClaimers();
    void pushEvent(EventKind kind, int player, int tile = -1, int discard = -1,
                   bool tsumo = false, int lsr = -1, int fan = 0);
};

// Public-information deal-in risk estimate for discarding `tile` by `player`
float estimateDealInRisk(const Game& g, int player, int tile);

// Simulates applying `action` to `player`'s hand and melds
void simulateActionOnHand(const Game& g, int player, const Action& action,
                          TileCounts& handOut, std::vector<Meld>& meldsOut);

// Built-in fast C++ heuristic policy (8-fan waits + shanten + ukeire + structure - risk)
Action heuristicMove(const Game& g, const std::vector<Action>& legal);

}  // namespace mcr
