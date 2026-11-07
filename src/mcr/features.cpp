#include "mcr/features.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mcr {

namespace {

inline int relSeat(int absSeat, int observer) {
    return (absSeat - observer + 4) & 3;
}

inline void writeTileThermometer(uint8_t* dst, const TileCounts& tc) {
    // 4 channels x 34 tiles: >=1, >=2, >=3, ==4
    for (int t = 0; t < kNumTileTypes; ++t) {
        uint8_t cnt = tc.c[t];
        if (cnt >= 1) dst[0 * kNumTileTypes + t] = 1;
        if (cnt >= 2) dst[1 * kNumTileTypes + t] = 1;
        if (cnt >= 3) dst[2 * kNumTileTypes + t] = 1;
        if (cnt >= 4) dst[3 * kNumTileTypes + t] = 1;
    }
}

}  // namespace

EncodedState encodeState(const Game& g) {
    EncodedState e{};
    int ego = g.currentPlayer;
    TileCounts pubVis = g.publicVisibleCounts();

    // Evaluate ego current hand potential
    HandPotential egoPot = evaluatePotential(
        g.hand[ego], g.melds[ego], pubVis, ego, g.prevalentWind, g.minFan);

    uint8_t* pImp = e.imp.data();

    // 1. Channels 0..3: Ego concealed hand counts (4 x 34 = 136)
    writeTileThermometer(pImp + 0 * kNumTileTypes, g.hand[ego]);

    // 2. Channel 4: Drawn tile one-hot (34)
    if (g.phase == Phase::Discard && g.drawnTile >= 0 && g.drawnTile < kNumTileTypes) {
        pImp[4 * kNumTileTypes + g.drawnTile] = 1;
    }

    // 3. Channel 5: Last discard tile one-hot (34)
    if (g.phase == Phase::Claim && g.lastDiscardTile >= 0 && g.lastDiscardTile < kNumTileTypes) {
        pImp[5 * kNumTileTypes + g.lastDiscardTile] = 1;
    }

    // 4. Channels 6..21: 4 players' discard river counts (4 players x 4 copies x 34 = 544)
    for (int rel = 0; rel < kNumPlayers; ++rel) {
        int p = (ego + rel) & 3;
        TileCounts riverCounts{};
        for (int8_t d : g.discards[p]) {
            riverCounts.add(d, 1);
        }
        writeTileThermometer(pImp + (6 + rel * 4) * kNumTileTypes, riverCounts);
    }

    // 5. Channels 22..37: 4 players' melds (CHI, PENG, MINGGANG/BUGANG, ANGANG) (4 x 4 x 34 = 544)
    for (int rel = 0; rel < kNumPlayers; ++rel) {
        int p = (ego + rel) & 3;
        uint8_t* mBase = pImp + (22 + rel * 4) * kNumTileTypes;
        for (const Meld& m : g.melds[p]) {
            if (m.kind == ActionType::CHI) {
                for (int k = 0; k < m.tileCount; ++k) {
                    int t = m.tiles[k];
                    if (t >= 0 && t < kNumTileTypes) mBase[0 * kNumTileTypes + t] = 1;
                }
            } else if (m.kind == ActionType::PENG) {
                int t = m.tiles[0];
                if (t >= 0 && t < kNumTileTypes) mBase[1 * kNumTileTypes + t] = 1;
            } else if (m.kind == ActionType::GANG) {
                int t = m.tiles[0];
                if (t >= 0 && t < kNumTileTypes) {
                    if (m.fromPlayer == p && !m.promotedKong) {
                        mBase[3 * kNumTileTypes + t] = 1; // AnGang
                    } else {
                        mBase[2 * kNumTileTypes + t] = 1; // MingGang / BuGang
                    }
                }
            }
        }
    }

    // 6. Channels 38..41: Total visible tiles (ego hand + all public rivers/melds) (4 x 34 = 136)
    TileCounts totalVis = pubVis;
    for (int t = 0; t < kNumTileTypes; ++t) {
        totalVis.add(t, g.hand[ego].c[t]);
    }
    writeTileThermometer(pImp + 38 * kNumTileTypes, totalVis);

    // 7. Channel 42: Ego useful / waiting tiles mask (34)
    for (int t = 0; t < kNumTileTypes; ++t) {
        if (egoPot.usefulMask[t]) {
            pImp[42 * kNumTileTypes + t] = 1;
        }
    }

    // 8. Imperfect-information Scalars (kImpScalars = 58)
    float* s = e.scalar.data();
    for (int rel = 0; rel < kNumPlayers; ++rel) {
        int p = (ego + rel) & 3;
        int hc = g.hand[p].total();
        s[0 + rel] = static_cast<float>(hc) / 14.0f;

        // 5-bucket one-hot for concealed hand size: 13+, 10..12, 7..9, 4..6, 0..3
        int bucket = (hc >= 13) ? 0 : (hc >= 10) ? 1 : (hc >= 7) ? 2 : (hc >= 4) ? 3 : 4;
        s[4 + rel * 5 + bucket] = 1.0f;

        s[24 + rel] = static_cast<float>(g.melds[p].size()) / 4.0f;
        int kongs = 0;
        bool openHand = false;
        for (const Meld& m : g.melds[p]) {
            if (m.kind == ActionType::GANG) ++kongs;
            if (m.fromPlayer != p || m.promotedKong) openHand = true;
        }
        s[28 + rel] = static_cast<float>(kongs) / 4.0f;
        s[32 + rel] = openHand ? 0.0f : 1.0f;
        s[36 + rel] = std::min(1.0f, static_cast<float>(g.discards[p].size()) / 24.0f);
    }

    s[40] = static_cast<float>(g.wall.size()) / 84.0f;
    s[41] = g.wall.empty() ? 1.0f : 0.0f;
    s[42] = g.aboutKong ? 1.0f : 0.0f;
    s[43] = g.claimHuOnly ? 1.0f : 0.0f;
    s[44] = (g.phase == Phase::Claim) ? 1.0f : 0.0f;
    if (g.lastDiscardPlayer >= 0) {
        s[45 + relSeat(g.lastDiscardPlayer, ego)] = 1.0f;
    }
    s[49 + (g.prevalentWind & 3)] = 1.0f;
    s[53 + (ego & 3)] = 1.0f;
    s[57] = static_cast<float>(egoPot.shanten) / 6.0f;

    // 9. LSTM Sequence of last 24 public discard/meld events (kLstmSteps = 24, kLstmIn = 88)
    std::vector<const GameEvent*> pubEvents;
    pubEvents.reserve(g.events.size());
    for (const GameEvent& ev : g.events) {
        if (ev.kind == EventKind::PLAY || ev.kind == EventKind::CHI ||
            ev.kind == EventKind::PENG || ev.kind == EventKind::GANG ||
            ev.kind == EventKind::ANGANG || ev.kind == EventKind::BUGANG) {
            pubEvents.push_back(&ev);
        }
    }
    int nEv = static_cast<int>(pubEvents.size());
    int startEv = std::max(0, nEv - kLstmSteps);
    int padSteps = kLstmSteps - (nEv - startEv);

    for (int idx = startEv; idx < nEv; ++idx) {
        int stepSlot = padSteps + (idx - startEv);
        float* row = e.seq.data() + stepSlot * kLstmIn;
        const GameEvent& ev = *pubEvents[idx];

        int rSeat = (ev.player >= 0) ? relSeat(ev.player, ego) : 0;
        row[0 + rSeat] = 1.0f;

        int kindSlot = 0;
        switch (ev.kind) {
            case EventKind::PLAY: kindSlot = 0; break;
            case EventKind::CHI: kindSlot = 1; break;
            case EventKind::PENG: kindSlot = 2; break;
            case EventKind::GANG: kindSlot = 3; break;
            case EventKind::ANGANG: kindSlot = 4; break;
            case EventKind::BUGANG: kindSlot = 5; break;
            default: kindSlot = 0; break;
        }
        row[4 + kindSlot] = 1.0f;

        int primaryTile = (ev.kind == EventKind::CHI || ev.kind == EventKind::PENG)
                              ? int(ev.discard)
                              : int(ev.tile);
        if (primaryTile >= 0 && primaryTile < kNumTileTypes) {
            row[10 + primaryTile] = 1.0f;
        }
        if (ev.tile >= 0 && ev.tile < kNumTileTypes) {
            row[44 + ev.tile] = 1.0f;
        }

        row[78 + (ev.tsumogiri ? 1 : 0)] = 1.0f;
        row[80 + (ev.claimed ? 1 : 0)] = 1.0f;

        for (int rel = 0; rel < kNumPlayers; ++rel) {
            int p = (ego + rel) & 3;
            row[82 + rel] = static_cast<float>(ev.handCounts[p]) / 14.0f;
        }
        row[86] = static_cast<float>(ev.wallRemaining) / 84.0f;
        row[87] = static_cast<float>(idx + 1) / 128.0f;
    }

    // 10. "开天眼" Omniscient Oracle Features for Critic (kExtraBin = 578, kExtraScalars = 34)
    uint8_t* pExt = e.extra.data();
    float* es = e.extraScalar.data();

    int minOppShanten = 6;
    for (int rel = 1; rel < kNumPlayers; ++rel) {
        int opp = (ego + rel) & 3;
        int oppIdx = rel - 1; // 0, 1, 2

        // Opponent true concealed hand (4 x 34)
        writeTileThermometer(pExt + (oppIdx * 4) * kNumTileTypes, g.hand[opp]);

        // Opponent true shanten and waiting/useful tiles
        HandPotential oppPot = evaluatePotential(
            g.hand[opp], g.melds[opp], pubVis, opp, g.prevalentWind, g.minFan);
        minOppShanten = std::min(minOppShanten, oppPot.shanten);

        uint8_t* waitRow = pExt + (12 + oppIdx) * kNumTileTypes;
        for (int t = 0; t < kNumTileTypes; ++t) {
            if (oppPot.usefulMask[t]) waitRow[t] = 1;
        }

        es[0 + oppIdx] = static_cast<float>(oppPot.shanten) / 6.0f;
        es[3 + oppIdx] = (oppPot.shanten <= 0) ? 1.0f : 0.0f;
        es[6 + oppIdx] = static_cast<float>(oppPot.qualifyingWaits) / 16.0f;
        es[9 + oppIdx] = std::min(1.0f, oppPot.expectedFan / 64.0f);
        es[12 + oppIdx] = static_cast<float>(oppPot.usefulRemaining) / 34.0f;
        es[15 + oppIdx] = oppPot.structure / 10.0f;

        // Opponent suit & honor counts
        int sCnt[4] = {0, 0, 0, 0};
        for (int t = 0; t < kNumTileTypes; ++t) {
            sCnt[tileSuit(t)] += g.hand[opp].c[t];
        }
        for (int su = 0; su < 4; ++su) {
            es[22 + oppIdx * 4 + su] = static_cast<float>(sCnt[su]) / 14.0f;
        }
    }

    // Wall look-ahead: next 4 tiles (channel 15) and next 16 tiles (channel 16)
    // Note: wall.back() is the very next tile to be drawn.
    int wallLen = static_cast<int>(g.wall.size());
    int egoUsefulInWall = 0;
    int egoUsefulInNext8 = 0;
    TileCounts wallCounts{};
    for (int i = 0; i < wallLen; ++i) {
        int t = g.wall[wallLen - 1 - i];
        wallCounts.add(t, 1);
        if (i < 4) pExt[15 * kNumTileTypes + t] = 1;
        if (i < 16) pExt[16 * kNumTileTypes + t] = 1;
        if (egoPot.usefulMask[t]) {
            ++egoUsefulInWall;
            if (i < 8) ++egoUsefulInNext8;
        }
    }

    int trueWaitsInWall = 0;
    if (egoPot.shanten <= 0) {
        for (int t = 0; t < kNumTileTypes; ++t) {
            if (egoPot.usefulMask[t]) trueWaitsInWall += wallCounts.c[t];
        }
    }

    es[18] = static_cast<float>(egoUsefulInWall) / 34.0f;
    es[19] = static_cast<float>(egoUsefulInNext8) / 8.0f;
    es[20] = static_cast<float>(minOppShanten - egoPot.shanten) / 6.0f;
    es[21] = static_cast<float>(trueWaitsInWall) / 16.0f;

    return e;
}

void legalOptions(const Game& g, std::vector<LegalOption>& out) {
    out.clear();
    std::vector<Action> acts = g.legalActions();
    if (acts.empty()) return;
    out.reserve(acts.size());

    int player = g.currentPlayer;
    TileCounts pubVis = g.publicVisibleCounts();
    int curShanten = computeShanten(g.hand[player], g.melds[player], nullptr);

    for (const Action& act : acts) {
        LegalOption opt;
        opt.abstractId = act.abstractId();
        opt.concrete = act;

        float* f = opt.feature.data();
        f[static_cast<int>(act.kind)] = 1.0f;
        f[7] = act.concealedKong ? 1.0f : 0.0f;

        if (act.tile >= 0 && act.tile < kNumTileTypes) {
            f[8 + act.tile] = 1.0f;
        }

        int discTile = (act.kind == ActionType::PLAY) ? int(act.tile)
                       : ((act.kind == ActionType::CHI || act.kind == ActionType::PENG) ? int(act.discard) : -1);
        if (discTile >= 0 && discTile < kNumTileTypes) {
            f[42 + discTile] = 1.0f;
        }

        if (act.kind == ActionType::CHI && act.seqStart >= 0 && act.seqStart < 27) {
            int suit = act.seqStart / 9;
            int rank0 = act.seqStart % 9;
            if (suit >= 0 && suit < 3) f[76 + suit] = 1.0f;
            if (rank0 >= 0 && rank0 < 7) f[79 + rank0] = 1.0f;
        }

        TileCounts nextHand;
        std::vector<Meld> nextMelds;
        simulateActionOnHand(g, player, act, nextHand, nextMelds);
        HandPotential pot = evaluatePotential(nextHand, nextMelds, pubVis, player, g.prevalentWind, g.minFan);
        float risk = (discTile >= 0) ? estimateDealInRisk(g, player, discTile) : 0.0f;

        f[86] = static_cast<float>(pot.shanten) / 6.0f;
        f[87] = static_cast<float>(curShanten - pot.shanten) / 2.0f;
        f[88] = static_cast<float>(pot.usefulTypes) / 34.0f;
        f[89] = static_cast<float>(pot.usefulRemaining) / 34.0f;
        f[90] = static_cast<float>(pot.qualifyingWaits) / 16.0f;
        f[91] = std::min(1.0f, pot.expectedFan / 64.0f);
        f[92] = pot.structure / 10.0f;
        f[93] = risk;
        f[94] = (discTile >= 0 && isTerminalOrHonor(discTile)) ? 1.0f : 0.0f;
        f[95] = (act.kind == ActionType::HU) ? 1.0f : 0.0f;

        out.push_back(opt);
    }
}

std::vector<LegalOption> legalOptions(const Game& g) {
    std::vector<LegalOption> out;
    legalOptions(g, out);
    return out;
}

float oraclePotentialAdvantage(const Game& g, int player) {
    TileCounts pubVis = g.publicVisibleCounts();
    auto phi = [&](int p) -> float {
        HandPotential pot = evaluatePotential(g.hand[p], g.melds[p], pubVis, p, g.prevalentWind, g.minFan);
        return -static_cast<float>(pot.shanten) +
               0.02f * static_cast<float>(pot.usefulRemaining) +
               0.15f * static_cast<float>(std::min(pot.qualifyingWaits, 8)) +
               0.05f * pot.structure;
    };
    float selfPhi = phi(player);
    float oppSum = 0.0f;
    for (int p = 0; p < kNumPlayers; ++p) {
        if (p != player) oppSum += phi(p);
    }
    return selfPhi - (oppSum / 3.0f);
}

}  // namespace mcr
