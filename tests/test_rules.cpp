#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "mcr/features.h"
#include "mcr/game.h"
#include "mcr/oracle.h"
#include "mcr/tiles.h"

namespace {

TEST(TilesTest, CodeRoundTripAndFullWall) {
    for (int t = 0; t < mcr::kNumTileTypes; ++t) {
        std::string code = mcr::tileName(t);
        EXPECT_EQ(mcr::parseTileName(code), t) << "Failed round-trip for tile " << t << " (" << code << ")";
    }

    mcr::Rng rng(42);
    std::vector<int8_t> wall = mcr::makeFullWall(rng);
    EXPECT_EQ(wall.size(), static_cast<size_t>(mcr::kFullWallSize));

    mcr::TileCounts counts;
    for (int8_t t : wall) {
        ASSERT_GE(t, 0);
        ASSERT_LT(t, mcr::kNumTileTypes);
        counts.add(t, 1);
    }
    for (int t = 0; t < mcr::kNumTileTypes; ++t) {
        EXPECT_EQ(counts.c[t], 4);
    }
}

TEST(TilesTest, ActionAbstractIdBijection) {
    std::unordered_set<int> ids;
    ids.insert(mcr::Action::pass().abstractId());
    ids.insert(mcr::Action::hu().abstractId());
    for (int t = 0; t < mcr::kNumTileTypes; ++t) {
        ids.insert(mcr::Action::play(t).abstractId());
        ids.insert(mcr::Action::angang(t).abstractId());
        ids.insert(mcr::Action::minggang(t).abstractId());
        ids.insert(mcr::Action::bugang(t).abstractId());
        for (int d = 0; d < mcr::kNumTileTypes; ++d) {
            ids.insert(mcr::Action::peng(t, d).abstractId());
        }
    }
    for (int s = 0; s < 3; ++s) {
        for (int r = 0; r < 7; ++r) {
            int seqStart = s * 9 + r;
            for (int pos = 0; pos < 3; ++pos) {
                int claimed = seqStart + pos;
                for (int d = 0; d < mcr::kNumTileTypes; ++d) {
                    int id = mcr::Action::chi(claimed, seqStart, d).abstractId();
                    EXPECT_GE(id, 0);
                    EXPECT_LT(id, mcr::kNumActions);
                    ids.insert(id);
                }
            }
        }
    }
    // 1(PASS) + 1(HU) + 34(PLAY) + 34(ANGANG) + 34(MINGGANG) + 34(BUGANG) + 1156(PENG) + 714(CHI) = 2008 < 2048
    EXPECT_EQ(ids.size(), 2008u);
}

TEST(OracleTest, FanCalculator81FanExamples) {
    // 1. Big Four Winds (大四喜, 88 fan)
    mcr::TileCounts bigFourWinds;
    bigFourWinds.add(mcr::parseTileName("F1"), 3);
    bigFourWinds.add(mcr::parseTileName("F2"), 3);
    bigFourWinds.add(mcr::parseTileName("F3"), 3);
    bigFourWinds.add(mcr::parseTileName("F4"), 3);
    bigFourWinds.add(mcr::parseTileName("W1"), 2);

    EXPECT_EQ(mcr::computeShanten(bigFourWinds, {}), -1);
    EXPECT_TRUE(mcr::isCompleteHand(bigFourWinds, {}, mcr::parseTileName("W1")));

    mcr::FanContext ctx;
    ctx.selfDrawn = true;
    int fan = mcr::computeFan(bigFourWinds, {}, mcr::parseTileName("W1"), ctx);
    EXPECT_GE(fan, 88);
    EXPECT_TRUE(mcr::canHu(bigFourWinds, {}, mcr::parseTileName("W1"), ctx, 8));

    // 2. Thirteen Orphans (十三幺, 88 fan)
    mcr::TileCounts orphans;
    for (const char* code : {"W1", "W9", "T1", "T9", "B1", "B9", "F1", "F2", "F3", "F4", "J1", "J2", "J3"}) {
        orphans.add(mcr::parseTileName(code), 1);
    }
    orphans.add(mcr::parseTileName("F1"), 1);
    EXPECT_EQ(mcr::computeShanten(orphans, {}), -1);
    int orphanFan = mcr::computeFan(orphans, {}, mcr::parseTileName("F1"), ctx);
    EXPECT_GE(orphanFan, 88);

    // 3. Seven Pairs (七对, 24 fan)
    mcr::TileCounts sevenPairs;
    for (const char* code : {"W1", "W3", "T2", "T8", "B4", "B7", "J1"}) {
        sevenPairs.add(mcr::parseTileName(code), 2);
    }
    EXPECT_EQ(mcr::computeShanten(sevenPairs, {}), -1);
    int spFan = mcr::computeFan(sevenPairs, {}, mcr::parseTileName("J1"), ctx);
    EXPECT_GE(spFan, 24);

    // 4. Low-fan hand (< 8 fan) rejected by 8-fan minimum requirement on Ron
    mcr::TileCounts lowFanHand;
    std::vector<mcr::Meld> melds;
    melds.push_back(mcr::Meld::makeChi(mcr::parseTileName("W1"), mcr::parseTileName("W2"), 3));
    melds.push_back(mcr::Meld::makePeng(mcr::parseTileName("T2"), 2));

    // Concealed: B2,B3,B4 + W6,W7,W8 + pair of B5 (All Simples = 2 fan or Mixed Double Chow = 1 fan, < 8 fan)
    for (const char* code : {"B2", "B3", "B4", "W6", "W7", "W8"}) {
        lowFanHand.add(mcr::parseTileName(code), 1);
    }
    lowFanHand.add(mcr::parseTileName("B5"), 2);

    mcr::FanContext ronCtx;
    ronCtx.seatWind = 0;
    ronCtx.prevalentWind = 0;
    ronCtx.selfDrawn = false;
    EXPECT_TRUE(mcr::isCompleteHand(lowFanHand, melds, mcr::parseTileName("W8")));
    int lowFan = mcr::computeFan(lowFanHand, melds, mcr::parseTileName("W8"), ronCtx);
    EXPECT_GT(lowFan, 0);
    EXPECT_LT(lowFan, 8);
    EXPECT_FALSE(mcr::canHu(lowFanHand, melds, mcr::parseTileName("W8"), ronCtx, 8));
}

TEST(GameTest, SelfPlayAndHeuristicCompleteCleanly) {
    mcr::Rng rng(12345);
    for (int gameIdx = 0; gameIdx < 5; ++gameIdx) {
        mcr::Game g;
        g.deal(rng);
        EXPECT_FALSE(g.over);
        EXPECT_EQ(g.phase, mcr::Phase::Discard);
        EXPECT_EQ(g.concealedCount(0), 14);
        EXPECT_EQ(g.concealedCount(1), 13);
        EXPECT_EQ(g.concealedCount(2), 13);
        EXPECT_EQ(g.concealedCount(3), 13);

        int steps = 0;
        while (!g.over && steps < 400) {
            auto legal = g.legalActions();
            ASSERT_FALSE(legal.empty());
            mcr::Action act = mcr::heuristicMove(g, legal);
            g.step(act);
            steps++;
        }
        EXPECT_TRUE(g.over);
        int sumScores = g.scores[0] + g.scores[1] + g.scores[2] + g.scores[3];
        EXPECT_EQ(sumScores, 0);
        if (g.winner >= 0) {
            EXPECT_GE(g.fanCount, 8);
            if (g.selfDrawn) {
                EXPECT_EQ(g.scores[g.winner], 3 * (8 + g.fanCount));
            } else {
                EXPECT_GE(g.loser, 0);
                EXPECT_EQ(g.scores[g.winner], 24 + g.fanCount);
                EXPECT_EQ(g.scores[g.loser], -(8 + g.fanCount));
            }
        }
    }
}

TEST(FeaturesTest, EncodesHandCountsDiscardSeqMeldsAndOracle) {
    mcr::Rng rng(99);
    mcr::Game g;
    g.deal(rng);

    for (int i = 0; i < 8 && !g.over; ++i) {
        auto legal = g.legalActions();
        g.step(mcr::heuristicMove(g, legal));
    }

    mcr::EncodedState enc = mcr::encodeState(g);

    // Check 4-player concealed hand counts in scalar[0..3]
    int obs = g.currentPlayer;
    for (int rel = 0; rel < 4; ++rel) {
        int p = (obs + rel) % 4;
        float expectedNorm = static_cast<float>(g.concealedCount(p)) / 14.0f;
        EXPECT_NEAR(enc.scalar[rel], expectedNorm, 1e-5f);
    }

    // Check LSTM sequence is populated for recent events
    float seqNorm = std::accumulate(enc.seq.begin(), enc.seq.end(), 0.0f,
                                    [](float acc, float x) { return acc + std::abs(x); });
    EXPECT_GT(seqNorm, 0.0f);

    // Check Omniscient Critic extra features include opponents' concealed hands (3 * 4 * 34)
    int oppOnes = 0;
    for (int i = 0; i < 12 * mcr::kNumTileTypes; ++i) {
        oppOnes += enc.extra[i];
    }
    int expectedOppTiles = 0;
    for (int rel = 1; rel < 4; ++rel) {
        expectedOppTiles += g.concealedCount((obs + rel) % 4);
    }
    EXPECT_EQ(oppOnes, expectedOppTiles);

    // Check legalOptions generates non-empty 96-dim features
    auto opts = mcr::legalOptions(g);
    EXPECT_FALSE(opts.empty());
    for (const auto& o : opts) {
        EXPECT_GE(o.abstractId, 0);
        EXPECT_LT(o.abstractId, mcr::kNumActions);
    }
}

}  // namespace
