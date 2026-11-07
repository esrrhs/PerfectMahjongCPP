#include "mcr/game.h"

#include <algorithm>
#include <stdexcept>

namespace mcr {

void Game::deal(Rng& rng, int wind) {
    std::vector<int8_t> w = makeFullWall();
    // Fisher-Yates shuffle
    for (size_t i = w.size() - 1; i > 0; --i) {
        size_t j = rng.nextU32(static_cast<uint32_t>(i + 1));
        std::swap(w[i], w[j]);
    }
    int pw = (wind >= 0) ? (wind & 3) : static_cast<int>(rng.nextU32(4));
    resetWithWall(w, pw);
}

void Game::resetWithWall(const std::vector<int8_t>& customWall, int wind) {
    if (customWall.size() < 53) {
        throw std::invalid_argument("Wall must have at least 53 tiles");
    }
    wall = customWall;
    for (int p = 0; p < kNumPlayers; ++p) {
        hand[p].clear();
        melds[p].clear();
        discards[p].clear();
        discardTsumogiri[p].clear();
        discardClaimed[p].clear();
    }
    events.clear();
    events.reserve(160);

    currentPlayer = 0;
    prevalentWind = wind & 3;
    phase = Phase::Draw;
    drawnTile = -1;
    lastDiscardPlayer = -1;
    lastDiscardTile = -1;
    pendingClaimers.clear();
    hasClaimResponse.fill(false);

    over = false;
    winner = -1;
    loser = -1;
    fanCount = 0;
    selfDrawn = false;
    scores = {0, 0, 0, 0};

    aboutKong = false;
    claimHuOnly = false;
    pendingBugangPlayer = -1;
    pendingBugangTile = -1;
    turnCount = 0;

    // Deal 13 tiles to each player in turn (popping from back of wall, matching MahjongEnv)
    for (int r = 0; r < 13; ++r) {
        for (int p = 0; p < kNumPlayers; ++p) {
            int t = wall.back();
            wall.pop_back();
            hand[p].add(t, 1);
        }
    }
    drawNextTile();
}

void Game::pushEvent(EventKind kind, int player, int tile, int discard,
                     bool tsumo, int lsr, int fan) {
    GameEvent ev;
    ev.kind = kind;
    ev.player = static_cast<int8_t>(player);
    ev.tile = static_cast<int8_t>(tile);
    ev.discard = static_cast<int8_t>(discard);
    ev.loser = static_cast<int8_t>(lsr);
    ev.fan = static_cast<int16_t>(fan);
    ev.tsumogiri = tsumo;
    ev.claimed = false;
    for (int p = 0; p < kNumPlayers; ++p) {
        ev.handCounts[p] = static_cast<int8_t>(hand[p].total());
    }
    ev.wallRemaining = static_cast<int16_t>(wall.size());
    events.push_back(ev);
}

void Game::drawNextTile() {
    if (wall.empty()) {
        over = true;
        phase = Phase::Terminal;
        scores = {0, 0, 0, 0};
        pushEvent(EventKind::DRAW_GAME, -1);
        return;
    }
    int tile = wall.back();
    wall.pop_back();
    drawnTile = tile;
    hand[currentPlayer].add(tile, 1);
    phase = Phase::Discard;
    ++turnCount;
    pushEvent(EventKind::DRAW, currentPlayer, tile);
}

FanContext Game::makeFanContext(int player, bool isSelfDrawn) const {
    int targetTile = isSelfDrawn ? drawnTile : lastDiscardTile;
    int visible = 0;
    if (targetTile >= 0 && targetTile < kNumTileTypes) {
        for (int p = 0; p < kNumPlayers; ++p) {
            for (int8_t d : discards[p]) {
                if (d == targetTile) ++visible;
            }
            for (const Meld& m : melds[p]) {
                int inMeld = 0;
                for (int k = 0; k < m.tileCount; ++k) {
                    if (m.tiles[k] == targetTile) ++inMeld;
                }
                visible += inMeld;
                if (m.fromPlayer != p && inMeld > 0 && m.claimedTile == targetTile) {
                    visible -= 1;
                }
            }
        }
    }
    FanContext ctx;
    ctx.playerId = player;
    ctx.seatWind = player;
    ctx.prevalentWind = prevalentWind;
    ctx.selfDrawn = isSelfDrawn;
    ctx.fourthTile = (targetTile >= 0) && (visible + (isSelfDrawn ? 1 : 0) >= 4);
    ctx.aboutKong = aboutKong;
    ctx.wallLast = wall.empty();
    ctx.flowerCount = 0;
    return ctx;
}

TileCounts Game::publicVisibleCounts() const {
    TileCounts vis{};
    for (int p = 0; p < kNumPlayers; ++p) {
        for (size_t i = 0; i < discards[p].size(); ++i) {
            if (!discardClaimed[p][i]) {
                vis.add(discards[p][i], 1);
            }
        }
        for (const Meld& m : melds[p]) {
            for (int k = 0; k < m.tileCount; ++k) {
                vis.add(m.tiles[k], 1);
            }
        }
    }
    return vis;
}

std::vector<Action> Game::claimActionsFor(int player) const {
    std::vector<Action> actions;
    actions.reserve(16);
    actions.push_back(Action::pass());

    int source = lastDiscardPlayer;
    int tile = lastDiscardTile;
    if (player == source || tile < 0) return actions;

    // 1. Check Ron / Hu
    TileCounts countsWithTile = hand[player];
    countsWithTile.add(tile, 1);
    if (canHu(countsWithTile, melds[player], tile, makeFanContext(player, false), minFan)) {
        actions.push_back(Action::hu(tile));
    }
    if (claimHuOnly) {
        return actions;
    }

    // Cannot claim Chow/Pong/Kong on the very last tile when wall is empty
    if (wall.empty()) {
        return actions;
    }

    const TileCounts& hc = hand[player];

    // 2. Check Peng (requires >= 2 copies in hand, followed by an immediate discard)
    if (hc.c[tile] >= 2) {
        TileCounts rem = hc;
        rem.remove(tile, 2);
        for (int d = 0; d < kNumTileTypes; ++d) {
            if (rem.c[d] > 0) {
                actions.push_back(Action::peng(tile, d));
            }
        }
    }

    // 3. Check MingGang (requires >= 3 copies in hand and replacement tile available in wall)
    if (hc.c[tile] >= 3 && !wall.empty()) {
        actions.push_back(Action::minggang(tile));
    }

    // 4. Check Chi (only next player (source + 1) % 4, and only for suited tiles 0..26)
    if (player == (source + 1) % 4 && isSuited(tile)) {
        int base = tile - (tile % 9);
        int rankIdx = tile % 9;
        int startMin = std::max(0, rankIdx - 2);
        int startMax = std::min(6, rankIdx);
        for (int start = startMin; start <= startMax; ++start) {
            int t0 = base + start;
            int t1 = base + start + 1;
            int t2 = base + start + 2;
            TileCounts rem = hc;
            bool ok = true;
            for (int st : {t0, t1, t2}) {
                if (st == tile) continue;
                if (rem.c[st] == 0) {
                    ok = false;
                    break;
                }
                rem.remove(st, 1);
            }
            if (ok) {
                for (int d = 0; d < kNumTileTypes; ++d) {
                    if (rem.c[d] > 0) {
                        actions.push_back(Action::chi(tile, t0, d));
                    }
                }
            }
        }
    }

    return actions;
}

std::vector<Action> Game::legalActions(int playerId) const {
    int player = (playerId < 0) ? currentPlayer : playerId;
    if (over || player != currentPlayer) return {};

    if (phase == Phase::Claim) {
        return claimActionsFor(player);
    }

    // Phase::Discard
    std::vector<Action> actions;
    actions.reserve(18);
    const TileCounts& hc = hand[player];

    for (int t = 0; t < kNumTileTypes; ++t) {
        if (hc.c[t] > 0) {
            actions.push_back(Action::play(t));
        }
    }

    if (drawnTile >= 0 &&
        canHu(hc, melds[player], drawnTile, makeFanContext(player, true), minFan)) {
        actions.push_back(Action::hu(drawnTile));
    }

    if (!wall.empty()) {
        for (int t = 0; t < kNumTileTypes; ++t) {
            if (hc.c[t] == 4) {
                actions.push_back(Action::angang(t));
            }
        }
        for (const Meld& m : melds[player]) {
            if (m.kind == ActionType::PENG && hc.c[m.tiles[0]] >= 1) {
                actions.push_back(Action::bugang(m.tiles[0]));
            }
        }
    }

    return actions;
}

void Game::finishHu(int player, bool isSelfDrawn) {
    int winTile = isSelfDrawn ? drawnTile : lastDiscardTile;
    TileCounts counts = hand[player];
    if (!isSelfDrawn) {
        counts.add(winTile, 1);
    }
    fanCount = std::max(minFan, computeFan(counts, melds[player], winTile, makeFanContext(player, isSelfDrawn)));
    over = true;
    phase = Phase::Terminal;
    winner = player;
    loser = isSelfDrawn ? -1 : lastDiscardPlayer;
    selfDrawn = isSelfDrawn;

    if (isSelfDrawn) {
        for (int i = 0; i < kNumPlayers; ++i) {
            scores[i] = (i == winner) ? 3 * (8 + fanCount) : -(8 + fanCount);
        }
    } else {
        for (int i = 0; i < kNumPlayers; ++i) {
            scores[i] = -8;
        }
        scores[winner] = 24 + fanCount;
        scores[loser] -= fanCount;
    }
    pushEvent(EventKind::HU, player, winTile, -1, false, loser, fanCount);
}

void Game::advanceAutoPassClaimers() {
    while (!over && phase == Phase::Claim) {
        auto actions = claimActionsFor(currentPlayer);
        if (actions.size() == 1 && actions[0].kind == ActionType::PASS) {
            recordClaim(Action::pass());
        } else {
            break;
        }
    }
}

void Game::recordClaim(const Action& action) {
    claimResponses[currentPlayer] = action;
    hasClaimResponse[currentPlayer] = true;

    int nextClaimer = -1;
    for (int p : pendingClaimers) {
        if (!hasClaimResponse[p]) {
            nextClaimer = p;
            break;
        }
    }
    if (nextClaimer >= 0) {
        currentPlayer = nextClaimer;
    } else {
        resolveClaims();
    }
}

void Game::resolveClaims() {
    int source = lastDiscardPlayer;
    int tile = lastDiscardTile;

    // Priority: HU > PENG / GANG > CHI (in seat turn order from discarder)
    for (ActionType targetKind : {ActionType::HU, ActionType::PENG, ActionType::GANG, ActionType::CHI}) {
        for (int player : pendingClaimers) {
            const Action& act = claimResponses[player];
            if (act.kind != targetKind) continue;
            if (targetKind == ActionType::CHI && player != (source + 1) % 4) continue;

            currentPlayer = player;
            if (targetKind == ActionType::HU) {
                finishHu(player, false);
                return;
            }
            if (claimHuOnly || wall.empty()) continue;

            // Mark the source player's last discard as claimed
            if (source >= 0 && !discardClaimed[source].empty()) {
                discardClaimed[source].back() = true;
            }
            if (!events.empty() && events.back().kind == EventKind::PLAY) {
                events.back().claimed = true;
            }

            if (targetKind == ActionType::CHI || targetKind == ActionType::PENG) {
                if (targetKind == ActionType::CHI) {
                    for (int st : {int(act.seqStart), int(act.seqStart) + 1, int(act.seqStart) + 2}) {
                        if (st != tile) hand[player].remove(st, 1);
                    }
                    melds[player].push_back(Meld::makeChi(act.seqStart, tile, source));
                } else {
                    hand[player].remove(tile, 2);
                    melds[player].push_back(Meld::makePeng(tile, source));
                }

                // Immediately discard act.discard
                hand[player].remove(act.discard, 1);
                discards[player].push_back(act.discard);
                discardTsumogiri[player].push_back(false);
                discardClaimed[player].push_back(false);
                lastDiscardPlayer = player;
                lastDiscardTile = act.discard;
                aboutKong = false;
                claimHuOnly = false;
                drawnTile = -1;
                ++turnCount;

                pushEvent(targetKind == ActionType::CHI ? EventKind::CHI : EventKind::PENG,
                          player, tile, act.discard);

                pendingClaimers = {(player + 1) % 4, (player + 2) % 4, (player + 3) % 4};
                hasClaimResponse.fill(false);
                currentPlayer = pendingClaimers[0];
                phase = Phase::Claim;
                return;
            }

            if (targetKind == ActionType::GANG) {
                hand[player].remove(tile, 3);
                melds[player].push_back(Meld::makeGang(tile, source, false));
                pushEvent(EventKind::GANG, player, tile);
                lastDiscardPlayer = -1;
                lastDiscardTile = -1;
                phase = Phase::Draw;
                aboutKong = true;
                claimHuOnly = false;
                drawNextTile();
                return;
            }
        }
    }

    // Nobody claimed HU / PENG / GANG / CHI:
    if (pendingBugangPlayer >= 0) {
        int player = pendingBugangPlayer;
        int bTile = pendingBugangTile;
        hand[player].remove(bTile, 1);
        for (Meld& m : melds[player]) {
            if (m.kind == ActionType::PENG && m.tiles[0] == bTile) {
                m = Meld::makeGang(bTile, m.fromPlayer, true);
                break;
            }
        }
        pendingBugangPlayer = -1;
        pendingBugangTile = -1;
        claimHuOnly = false;
        currentPlayer = player;
        lastDiscardPlayer = -1;
        lastDiscardTile = -1;
        phase = Phase::Draw;
        aboutKong = true;
        drawNextTile();
        return;
    }

    // Normal discard passed by all 3 opponents -> next player draws
    currentPlayer = (source + 1) % 4;
    lastDiscardPlayer = -1;
    lastDiscardTile = -1;
    phase = Phase::Draw;
    aboutKong = false;
    claimHuOnly = false;
    drawNextTile();
}

void Game::step(const Action& action) {
    if (over) return;

    int player = currentPlayer;
    if (action.kind == ActionType::HU) {
        if (phase == Phase::Claim) {
            recordClaim(action);
        } else {
            finishHu(player, true);
        }
    } else if (action.kind == ActionType::PLAY) {
        bool isTsumoDiscard = (action.tile == drawnTile);
        hand[player].remove(action.tile, 1);
        discards[player].push_back(action.tile);
        discardTsumogiri[player].push_back(isTsumoDiscard);
        discardClaimed[player].push_back(false);
        lastDiscardPlayer = player;
        lastDiscardTile = action.tile;
        aboutKong = false;
        claimHuOnly = false;
        drawnTile = -1;

        pushEvent(EventKind::PLAY, player, action.tile, action.tile, isTsumoDiscard);

        pendingClaimers = {(player + 1) % 4, (player + 2) % 4, (player + 3) % 4};
        hasClaimResponse.fill(false);
        currentPlayer = pendingClaimers[0];
        phase = Phase::Claim;
    } else if (action.kind == ActionType::PASS ||
               action.kind == ActionType::CHI ||
               action.kind == ActionType::PENG) {
        recordClaim(action);
    } else if (action.kind == ActionType::GANG) {
        if (phase == Phase::Claim) {
            recordClaim(action);
        } else {
            // Concealed Kong (AnGang)
            int t = action.tile;
            hand[player].remove(t, 4);
            melds[player].push_back(Meld::makeGang(t, player, false));
            pushEvent(EventKind::ANGANG, player, t);
            lastDiscardPlayer = -1;
            lastDiscardTile = -1;
            phase = Phase::Draw;
            aboutKong = true;
            claimHuOnly = false;
            drawNextTile();
        }
    } else if (action.kind == ActionType::BUGANG) {
        int t = action.tile;
        pushEvent(EventKind::BUGANG, player, t);
        lastDiscardPlayer = player;
        lastDiscardTile = t;
        pendingBugangPlayer = player;
        pendingBugangTile = t;
        pendingClaimers = {(player + 1) % 4, (player + 2) % 4, (player + 3) % 4};
        hasClaimResponse.fill(false);
        currentPlayer = pendingClaimers[0];
        phase = Phase::Claim;
        claimHuOnly = true;
        aboutKong = true;
    }

    if (!over && phase == Phase::Claim) {
        advanceAutoPassClaimers();
    }
}

float estimateDealInRisk(const Game& g, int player, int tile) {
    if (tile < 0 || tile >= kNumTileTypes) return 0.0f;
    TileCounts vis = g.publicVisibleCounts();
    for (int t = 0; t < kNumTileTypes; ++t) {
        vis.add(t, g.hand[player].c[t]);
    }

    float maxRisk = 0.0f;
    for (int opp = 0; opp < kNumPlayers; ++opp) {
        if (opp == player) continue;
        const auto& river = g.discards[opp];
        bool inRiver = std::find(river.begin(), river.end(), static_cast<int8_t>(tile)) != river.end();
        if (inRiver) continue;

        float risk = std::max(0.0f, 1.0f - static_cast<float>(vis.c[tile]) / 4.0f);
        if (tile >= 27) {
            risk *= 0.6f;
        } else {
            int base = tile - (tile % 9);
            int rank = tile % 9;
            bool sujiLeft = (rank >= 3) &&
                            (std::find(river.begin(), river.end(), static_cast<int8_t>(base + rank - 3)) != river.end());
            bool sujiRight = (rank <= 5) &&
                             (std::find(river.begin(), river.end(), static_cast<int8_t>(base + rank + 3)) != river.end());
            if (sujiLeft || sujiRight) {
                risk *= 0.65f;
            }
        }
        float openness = static_cast<float>(g.melds[opp].size());
        risk *= (1.0f + 0.15f * openness);
        if (risk > maxRisk) maxRisk = risk;
    }
    return maxRisk;
}

void simulateActionOnHand(const Game& g, int player, const Action& action,
                          TileCounts& handOut, std::vector<Meld>& meldsOut) {
    handOut = g.hand[player];
    meldsOut = g.melds[player];

    int src = (g.lastDiscardPlayer >= 0) ? g.lastDiscardPlayer : player;
    switch (action.kind) {
        case ActionType::PASS:
        case ActionType::HU:
            break;
        case ActionType::PLAY:
            handOut.remove(action.tile, 1);
            break;
        case ActionType::CHI:
            for (int st : {int(action.seqStart), int(action.seqStart) + 1, int(action.seqStart) + 2}) {
                if (st != action.tile) handOut.remove(st, 1);
            }
            handOut.remove(action.discard, 1);
            meldsOut.push_back(Meld::makeChi(action.seqStart, action.tile, src));
            break;
        case ActionType::PENG:
            handOut.remove(action.tile, 2);
            handOut.remove(action.discard, 1);
            meldsOut.push_back(Meld::makePeng(action.tile, src));
            break;
        case ActionType::GANG:
            handOut.remove(action.tile, (g.phase == Phase::Claim) ? 3 : 4);
            meldsOut.push_back(Meld::makeGang(action.tile, (g.phase == Phase::Claim) ? src : player, false));
            break;
        case ActionType::BUGANG:
            handOut.remove(action.tile, 1);
            for (Meld& m : meldsOut) {
                if (m.kind == ActionType::PENG && m.tiles[0] == action.tile) {
                    m = Meld::makeGang(action.tile, m.fromPlayer, true);
                    break;
                }
            }
            break;
    }
}

Action heuristicMove(const Game& g, const std::vector<Action>& legal) {
    if (legal.empty()) return Action::pass();
    for (const Action& a : legal) {
        if (a.kind == ActionType::HU) return a;
    }
    if (legal.size() == 1) return legal[0];

    int player = g.currentPlayer;
    TileCounts pubVis = g.publicVisibleCounts();

    int bestIdx = 0;
    float bestScore = -1e30f;

    for (size_t i = 0; i < legal.size(); ++i) {
        const Action& act = legal[i];
        TileCounts nextHand;
        std::vector<Meld> nextMelds;
        simulateActionOnHand(g, player, act, nextHand, nextMelds);
        HandPotential pot = evaluatePotential(nextHand, nextMelds, pubVis, player, g.prevalentWind, g.minFan);
        float meldPenalty = (act.kind == ActionType::CHI || act.kind == ActionType::PENG) ? 0.25f : 0.0f;
        int discTile = (act.kind == ActionType::PLAY) ? int(act.tile)
                       : ((act.kind == ActionType::CHI || act.kind == ActionType::PENG) ? int(act.discard) : -1);
        float risk = estimateDealInRisk(g, player, discTile);

        float score = pot.qualifyingWaits * 1000.0f +
                      pot.expectedFan * 0.2f -
                      pot.shanten * 100.0f +
                      pot.usefulRemaining * 2.0f +
                      pot.structure -
                      meldPenalty -
                      risk * 0.5f;
        if (score > bestScore || (score == bestScore && act.key() > legal[bestIdx].key())) {
            bestScore = score;
            bestIdx = static_cast<int>(i);
        }
    }
    return legal[bestIdx];
}

}  // namespace mcr
