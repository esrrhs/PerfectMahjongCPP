#include "mcr/oracle.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "mahjong_algo/fan_calculator.h"
#include "mahjong_algo/shanten.h"

namespace mcr {

namespace {

int computeShanten13(const TileCounts& counts,
                     std::array<bool, kNumTileTypes>* usefulOut) {
    mahjong::tile_t standing[14];
    int n = 0;
    for (int t = 0; t < kNumTileTypes; ++t) {
        for (int k = 0; k < counts.c[t] && n < 14; ++k) {
            standing[n++] = toAlgoTile(t);
        }
    }
    if (n != 13 && n != 10 && n != 7 && n != 4 && n != 1) {
        if (usefulOut) usefulOut->fill(false);
        return 6;
    }

    int sReg = mahjong::regular_shanten(standing, n, nullptr);
    int bestShanten = sReg;

    int sKs = 99, s7p = 99, s13o = 99, sHk = 99;
    if (n == 13 || n == 10) {
        sKs = mahjong::knitted_straight_shanten(standing, n, nullptr);
        bestShanten = std::min(bestShanten, sKs);
    }
    if (n == 13) {
        s7p = mahjong::seven_pairs_shanten(standing, n, nullptr);
        s13o = mahjong::thirteen_orphans_shanten(standing, n, nullptr);
        sHk = mahjong::honors_and_knitted_tiles_shanten(standing, n, nullptr);
        bestShanten = std::min({bestShanten, s7p, s13o, sHk});
    }

    if (usefulOut) {
        std::array<bool, kNumTileTypes> bestUseful{};
        if (bestShanten <= 1) {
            mahjong::useful_table_t ut{};
            auto mergeUt = [&]() {
                for (int i = 0; i < kNumTileTypes; ++i) {
                    if (ut[mahjong::all_tiles[i]]) bestUseful[i] = true;
                }
            };
            if (sReg == bestShanten) {
                std::memset(ut, 0, sizeof(ut));
                if (bestShanten == 0) {
                    mahjong::is_regular_wait(standing, n, &ut);
                } else {
                    mahjong::regular_shanten(standing, n, &ut);
                }
                mergeUt();
            }
            if ((n == 13 || n == 10) && sKs == bestShanten) {
                std::memset(ut, 0, sizeof(ut));
                mahjong::knitted_straight_shanten(standing, n, &ut);
                mergeUt();
            }
            if (n == 13) {
                if (s7p == bestShanten) {
                    std::memset(ut, 0, sizeof(ut));
                    mahjong::seven_pairs_shanten(standing, n, &ut);
                    mergeUt();
                }
                if (s13o == bestShanten) {
                    std::memset(ut, 0, sizeof(ut));
                    mahjong::thirteen_orphans_shanten(standing, n, &ut);
                    mergeUt();
                }
                if (sHk == bestShanten) {
                    std::memset(ut, 0, sizeof(ut));
                    mahjong::honors_and_knitted_tiles_shanten(standing, n, &ut);
                    mergeUt();
                }
            }
        } else {
            // Fast O(34) useful tile detection for hands with shanten >= 2
            for (int t = 0; t < kNumTileTypes; ++t) {
                if (counts.c[t] >= 4) continue;
                if (counts.c[t] >= 1) {
                    bestUseful[t] = true;
                    continue;
                }
                if (isSuited(t)) {
                    int r = t % 9;
                    if ((r >= 1 && counts.c[t - 1] > 0) ||
                        (r >= 2 && counts.c[t - 2] > 0) ||
                        (r <= 7 && counts.c[t + 1] > 0) ||
                        (r <= 6 && counts.c[t + 2] > 0)) {
                        bestUseful[t] = true;
                    }
                }
            }
        }
        *usefulOut = bestUseful;
    }
    return bestShanten == std::numeric_limits<int>::max() ? 6 : bestShanten;
}

}  // namespace

int computeShanten(const TileCounts& standingCounts,
                   const std::vector<Meld>& melds,
                   std::array<bool, kNumTileTypes>* usefulOut) {
    int n = standingCounts.total();
    int expected13 = 13 - static_cast<int>(melds.size()) * 3;
    if (n == expected13) {
        return computeShanten13(standingCounts, usefulOut);
    }
    if (n == expected13 + 1) {
        // Fast check if any tile completes the 14-tile hand
        for (int t = 0; t < kNumTileTypes; ++t) {
            if (standingCounts.c[t] > 0 && isCompleteHand(standingCounts, melds, t)) {
                if (usefulOut) usefulOut->fill(false);
                return -1;
            }
        }
        int minS = 6;
        int bestT = -1;
        TileCounts work = standingCounts;
        for (int t = 0; t < kNumTileTypes; ++t) {
            if (work.c[t] == 0) continue;
            work.c[t] -= 1;
            int s = computeShanten13(work, nullptr);
            if (s < minS) {
                minS = s;
                bestT = t;
            }
            work.c[t] += 1;
        }
        if (usefulOut) {
            if (bestT >= 0) {
                work.c[bestT] -= 1;
                computeShanten13(work, usefulOut);
                work.c[bestT] += 1;
            } else {
                usefulOut->fill(false);
            }
        }
        return minS;
    }
    if (usefulOut) usefulOut->fill(false);
    return 6;
}

bool isCompleteHand(const TileCounts& standingCountsWithWinTile,
                    const std::vector<Meld>& melds,
                    int winTile) {
    if (winTile < 0 || winTile >= kNumTileTypes) return false;
    if (standingCountsWithWinTile.c[winTile] == 0) return false;
    int expected14 = 14 - static_cast<int>(melds.size()) * 3;
    if (standingCountsWithWinTile.total() != expected14) return false;

    TileCounts work = standingCountsWithWinTile;
    work.c[winTile] -= 1;

    mahjong::tile_t standing[13];
    int n = 0;
    for (int t = 0; t < kNumTileTypes; ++t) {
        for (int k = 0; k < work.c[t] && n < 13; ++k) {
            standing[n++] = toAlgoTile(t);
        }
    }
    mahjong::tile_t wt = toAlgoTile(winTile);
    if (mahjong::is_regular_win(standing, n, wt)) return true;
    if (n == 13 || n == 10) {
        if (mahjong::is_knitted_straight_win(standing, n, wt)) return true;
    }
    if (n == 13) {
        if (mahjong::is_seven_pairs_win(standing, n, wt)) return true;
        if (mahjong::is_thirteen_orphans_win(standing, n, wt)) return true;
        if (mahjong::is_honors_and_knitted_tiles_win(standing, n, wt)) return true;
    }
    return false;
}

int computeFan(const TileCounts& standingCountsWithWinTile,
               const std::vector<Meld>& melds,
               int winTile,
               const FanContext& ctx) {
    if (winTile < 0 || winTile >= kNumTileTypes) return 0;
    if (standingCountsWithWinTile.c[winTile] == 0) return 0;
    int expected14 = 14 - static_cast<int>(melds.size()) * 3;
    if (standingCountsWithWinTile.total() != expected14) return 0;
    if (melds.size() > 4) return 0;

    TileCounts work = standingCountsWithWinTile;
    work.c[winTile] -= 1;

    mahjong::calculate_param_t param{};
    int n = 0;
    for (int t = 0; t < kNumTileTypes; ++t) {
        for (int k = 0; k < work.c[t] && n < 13; ++k) {
            param.hand_tiles.standing_tiles[n++] = toAlgoTile(t);
        }
    }
    param.hand_tiles.tile_count = n;
    param.hand_tiles.pack_count = static_cast<intptr_t>(melds.size());
    param.win_tile = toAlgoTile(winTile);
    param.flower_count = static_cast<uint8_t>(std::max(0, ctx.flowerCount));
    param.win_flag = (ctx.selfDrawn ? WIN_FLAG_SELF_DRAWN : 0) |
                     (ctx.fourthTile ? WIN_FLAG_LAST_TILE : 0) |
                     (ctx.aboutKong ? WIN_FLAG_KONG_INVOLVED : 0) |
                     (ctx.wallLast ? WIN_FLAG_WALL_LAST : 0);
    param.seat_wind = static_cast<mahjong::wind_t>(ctx.seatWind & 3);
    param.prevalent_wind = static_cast<mahjong::wind_t>(ctx.prevalentWind & 3);

    // Note: To match Botzone-Chinese-Standard-Mahjong's conservative _pack_variants minimum
    // (or exact offer when known), for CHI melds we evaluate offer in {1,2,3} if claimedTile
    // is unspecified, or exact offer = (claimedTile - tiles[0]) + 1.
    // Wait: Botzone-Chinese-Standard-Mahjong's RulesBackend.fan() takes min over offer in (1,2,3)
    // for CHI melds because its Meld namedtuple doesn't store claimedTile!
    // Let's check: if we want 100% agreement with Botzone-Chinese-Standard-Mahjong's legality check,
    // let's check if claimedTile is valid: even better, let's check min over CHI offers (1,2,3)
    // just like Botzone-Chinese-Standard-Mahjong's RulesBackend.fan() so there is NEVER any
    // disagreement between C++ and Python!
    std::vector<std::vector<mahjong::pack_t>> packChoices(melds.size());
    for (size_t i = 0; i < melds.size(); ++i) {
        const Meld& m = melds[i];
        if (m.kind == ActionType::CHI) {
            int midTile = m.tiles[1];
            mahjong::tile_t mt = toAlgoTile(midTile);
            for (uint8_t off : {1, 2, 3}) {
                packChoices[i].push_back(mahjong::make_pack(off, PACK_TYPE_CHOW, mt));
            }
        } else if (m.kind == ActionType::PENG) {
            uint8_t off = static_cast<uint8_t>((ctx.playerId - m.fromPlayer + 4) % 4);
            if (off == 0) off = 1;
            packChoices[i].push_back(mahjong::make_pack(off, PACK_TYPE_PUNG, toAlgoTile(m.tiles[0])));
        } else if (m.kind == ActionType::GANG) {
            uint8_t off = static_cast<uint8_t>((ctx.playerId - m.fromPlayer + 4) % 4);
            packChoices[i].push_back(mahjong::make_pack(off, PACK_TYPE_KONG, toAlgoTile(m.tiles[0])));
        }
    }

    int minTotalFan = std::numeric_limits<int>::max();
    bool anyValid = false;

    auto evalPacks = [&](auto& self, size_t idx) -> void {
        if (idx == melds.size()) {
            mahjong::fan_table_t fanTable{};
            int re = mahjong::calculate_fan(&param, &fanTable);
            if (re > 0) {
                anyValid = true;
                if (re < minTotalFan) minTotalFan = re;
            }
            return;
        }
        for (mahjong::pack_t pk : packChoices[idx]) {
            param.hand_tiles.fixed_packs[idx] = pk;
            self(self, idx + 1);
        }
    };

    evalPacks(evalPacks, 0);
    return anyValid ? minTotalFan : 0;
}

bool canHu(const TileCounts& standingCountsWithWinTile,
           const std::vector<Meld>& melds,
           int winTile,
           const FanContext& ctx,
           int minFan) {
    if (!isCompleteHand(standingCountsWithWinTile, melds, winTile)) {
        return false;
    }
    return computeFan(standingCountsWithWinTile, melds, winTile, ctx) >= minFan;
}

HandPotential evaluatePotential(const TileCounts& standingCountsAfter,
                                const std::vector<Meld>& meldsAfter,
                                const TileCounts& publicVisibleCounts,
                                int playerId,
                                int prevalentWind,
                                int minFan) {
    HandPotential hp;
    hp.shanten = computeShanten(standingCountsAfter, meldsAfter, &hp.usefulMask);

    for (int t = 0; t < kNumTileTypes; ++t) {
        if (!hp.usefulMask[t]) continue;
        ++hp.usefulTypes;
        int rem = std::max(0, 4 - static_cast<int>(standingCountsAfter.c[t]) -
                                  static_cast<int>(publicVisibleCounts.c[t]));
        hp.usefulRemaining += rem;

        if (hp.shanten == 0 && rem > 0) {
            TileCounts work = standingCountsAfter;
            work.add(t, 1);
            FanContext ctx;
            ctx.playerId = playerId;
            ctx.seatWind = playerId;
            ctx.prevalentWind = prevalentWind;
            ctx.selfDrawn = true;
            int f = computeFan(work, meldsAfter, t, ctx);
            if (f >= minFan) {
                hp.qualifyingWaits += rem;
                hp.expectedFan += static_cast<float>(rem * std::min(f, 88));
            }
        }
    }

    int suited[3] = {0, 0, 0};
    for (int s = 0; s < 3; ++s) {
        for (int r = 0; r < 9; ++r) {
            suited[s] += standingCountsAfter.c[s * 9 + r];
        }
    }
    int honors = 0;
    for (int t = 27; t < 34; ++t) honors += standingCountsAfter.c[t];
    int triplets = 0, pairs = 0;
    for (int t = 0; t < kNumTileTypes; ++t) {
        if (standingCountsAfter.c[t] >= 3) ++triplets;
        if (standingCountsAfter.c[t] >= 2) ++pairs;
    }
    int terminals = standingCountsAfter.c[0] + standingCountsAfter.c[8] +
                    standingCountsAfter.c[9] + standingCountsAfter.c[17] +
                    standingCountsAfter.c[18] + standingCountsAfter.c[26];
    int maxSuit = std::max({suited[0], suited[1], suited[2]});
    int sumSuit = suited[0] + suited[1] + suited[2];
    float concentration = static_cast<float>(maxSuit) - static_cast<float>(sumSuit - maxSuit) * 0.25f;
    hp.structure = (meldsAfter.empty() ? 1.0f : 0.0f) +
                   triplets * 0.7f + pairs * 0.15f +
                   concentration * 0.08f + honors * 0.12f + terminals * 0.04f;
    return hp;
}

}  // namespace mcr
