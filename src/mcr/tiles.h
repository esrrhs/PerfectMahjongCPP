#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "mahjong_algo/tile.h"

namespace mcr {

constexpr int kNumTileTypes = 34;
constexpr int kFullWallSize = 136;
constexpr int kNumPlayers = 4;
constexpr int kNumActions = 2048;

// Canonical tile indices 0..33:
//   0..8   : W1..W9 (Characters / 万子)
//   9..17  : T1..T9 (Bamboo / 条子)
//   18..26 : B1..B9 (Dots / 饼子)
//   27..30 : F1..F4 (East, South, West, North Winds / 东南西北风)
//   31..33 : J1..J3 (Zhong, Fa, Bai Dragons / 中发白箭牌)

inline bool isSuited(int tile) { return tile >= 0 && tile < 27; }
inline bool isHonor(int tile) { return tile >= 27 && tile < 34; }
inline bool isWind(int tile) { return tile >= 27 && tile < 31; }
inline bool isDragon(int tile) { return tile >= 31 && tile < 34; }
inline bool isTerminal(int tile) {
    return isSuited(tile) && (tile % 9 == 0 || tile % 9 == 8);
}
inline bool isTerminalOrHonor(int tile) {
    return isHonor(tile) || isTerminal(tile);
}
inline int tileSuit(int tile) { return isSuited(tile) ? (tile / 9) : 3; }
inline int tileRank(int tile) { return isSuited(tile) ? (tile % 9 + 1) : (tile - 26); }

// Convert 0..33 tile index <-> mahjong::tile_t
inline mahjong::tile_t toAlgoTile(int tile) {
    return mahjong::all_tiles[tile];
}
int fromAlgoTile(mahjong::tile_t t);

const char* tileName(int tile);
int parseTileName(const std::string& s);

// Deterministic 64-bit SplitMix64 PRNG
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed = 1) : state(seed ? seed : 1ULL) {}

    uint64_t nextU64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    uint32_t nextU32(uint32_t bound) {
        return bound ? static_cast<uint32_t>(nextU64() % bound) : 0;
    }

    int nextInt(int bound) {
        return bound > 0 ? static_cast<int>(nextU32(static_cast<uint32_t>(bound))) : 0;
    }

    float uniform01() {
        return static_cast<float>((nextU64() >> 40) * (1.0 / 16777216.0));
    }
};

struct TileCounts {
    std::array<uint8_t, kNumTileTypes> c{};

    void clear() { c.fill(0); }

    int total() const {
        int sum = 0;
        for (int i = 0; i < kNumTileTypes; ++i) sum += c[i];
        return sum;
    }

    void add(int tile, int count = 1) {
        if (tile >= 0 && tile < kNumTileTypes) {
            c[tile] = static_cast<uint8_t>(c[tile] + count);
        }
    }

    void remove(int tile, int count = 1) {
        if (tile >= 0 && tile < kNumTileTypes) {
            c[tile] = static_cast<uint8_t>(c[tile] >= count ? c[tile] - count : 0);
        }
    }

    std::vector<int> toSortedList() const {
        std::vector<int> res;
        res.reserve(14);
        for (int t = 0; t < kNumTileTypes; ++t) {
            for (int k = 0; k < c[t]; ++k) res.push_back(t);
        }
        return res;
    }

    std::string toString() const;
    static TileCounts parse(const std::string& s);

    bool operator==(const TileCounts& o) const { return c == o.c; }
    bool operator!=(const TileCounts& o) const { return c != o.c; }
};

enum class ActionType : int8_t {
    PASS = 0,
    PLAY = 1,
    CHI = 2,
    PENG = 3,
    GANG = 4,
    BUGANG = 5,
    HU = 6
};

const char* actionTypeName(ActionType kind);

struct Meld {
    ActionType kind = ActionType::PASS; // CHI, PENG, GANG
    std::array<int8_t, 4> tiles{-1, -1, -1, -1};
    int8_t tileCount = 0;
    int8_t fromPlayer = -1;   // 0..3; if fromPlayer == owner, concealed kong (ANGANG)
    int8_t claimedTile = -1;  // 0..33
    bool promotedKong = false;

    static Meld makeChi(int seqStart, int claimed, int from) {
        Meld m;
        m.kind = ActionType::CHI;
        m.tiles = {static_cast<int8_t>(seqStart),
                   static_cast<int8_t>(seqStart + 1),
                   static_cast<int8_t>(seqStart + 2), -1};
        m.tileCount = 3;
        m.fromPlayer = static_cast<int8_t>(from);
        m.claimedTile = static_cast<int8_t>(claimed);
        return m;
    }

    static Meld makePeng(int tile, int from) {
        Meld m;
        m.kind = ActionType::PENG;
        m.tiles = {static_cast<int8_t>(tile), static_cast<int8_t>(tile),
                   static_cast<int8_t>(tile), -1};
        m.tileCount = 3;
        m.fromPlayer = static_cast<int8_t>(from);
        m.claimedTile = static_cast<int8_t>(tile);
        return m;
    }

    static Meld makeGang(int tile, int from, bool promoted = false) {
        Meld m;
        m.kind = ActionType::GANG;
        m.tiles = {static_cast<int8_t>(tile), static_cast<int8_t>(tile),
                   static_cast<int8_t>(tile), static_cast<int8_t>(tile)};
        m.tileCount = 4;
        m.fromPlayer = static_cast<int8_t>(from);
        m.claimedTile = static_cast<int8_t>(tile);
        m.promotedKong = promoted;
        return m;
    }
};

// Bijective abstract action ID mapping (0 .. 2007 < kNumActions = 2048):
//   0           : PASS
//   1           : HU
//   2 .. 35     : PLAY(tile)               [34]
//   36 .. 69    : ANGANG(tile)             [34]
//   70 .. 103   : MINGGANG(tile)           [34]
//   104 .. 137  : BUGANG(tile)             [34]
//   138 .. 1293 : PENG(tile, discard)      [34 * 34 = 1156]
//   1294 .. 2007: CHI(seqIdx, discard)     [21 * 34 = 714], seqIdx = (seqStart/9)*7 + (seqStart%9)
struct Action {
    ActionType kind = ActionType::PASS;
    int8_t tile = -1;       // primary / claimed tile (0..33)
    int8_t seqStart = -1;   // for CHI: start tile of (seqStart, seqStart+1, seqStart+2)
    int8_t discard = -1;    // for CHI and PENG: immediate post-claim discard tile (0..33)
    bool concealedKong = false;

    static Action pass() { return Action{ActionType::PASS, -1, -1, -1, false}; }
    static Action hu(int winTile = -1) {
        return Action{ActionType::HU, static_cast<int8_t>(winTile), -1, -1, false};
    }
    static Action play(int t) {
        return Action{ActionType::PLAY, static_cast<int8_t>(t), -1, -1, false};
    }
    static Action angang(int t) {
        return Action{ActionType::GANG, static_cast<int8_t>(t), -1, -1, true};
    }
    static Action minggang(int t) {
        return Action{ActionType::GANG, static_cast<int8_t>(t), -1, -1, false};
    }
    static Action bugang(int t) {
        return Action{ActionType::BUGANG, static_cast<int8_t>(t), -1, -1, false};
    }
    static Action peng(int t, int disc) {
        return Action{ActionType::PENG, static_cast<int8_t>(t), -1, static_cast<int8_t>(disc), false};
    }
    static Action chi(int claimed, int seqStartTile, int disc) {
        return Action{ActionType::CHI, static_cast<int8_t>(claimed),
                      static_cast<int8_t>(seqStartTile), static_cast<int8_t>(disc), false};
    }

    int abstractId() const;

    // Canonical key matching Botzone-Chinese-Standard-Mahjong's Action.key()
    std::tuple<int, int, int, int> key() const {
        return {static_cast<int>(kind), static_cast<int>(tile),
                static_cast<int>(seqStart), static_cast<int>(discard)};
    }

    bool operator==(const Action& o) const {
        return kind == o.kind && tile == o.tile && seqStart == o.seqStart && discard == o.discard;
    }
    bool operator!=(const Action& o) const { return !(*this == o); }

    std::string toString() const;
    static Action parse(const std::string& s);
};

std::vector<int8_t> makeFullWall();
inline std::vector<int8_t> makeFullWall(Rng& rng) {
    std::vector<int8_t> w = makeFullWall();
    for (size_t i = w.size() - 1; i > 0; --i) {
        size_t j = rng.nextU32(static_cast<uint32_t>(i + 1));
        std::swap(w[i], w[j]);
    }
    return w;
}

}  // namespace mcr
