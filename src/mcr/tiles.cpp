#include "mcr/tiles.h"

#include <sstream>
#include <stdexcept>

namespace mcr {

namespace {

constexpr const char* kTileNames[kNumTileTypes] = {
    "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
    "F1", "F2", "F3", "F4", "J1", "J2", "J3"
};

}  // namespace

int fromAlgoTile(mahjong::tile_t t) {
    for (int i = 0; i < kNumTileTypes; ++i) {
        if (mahjong::all_tiles[i] == t) return i;
    }
    return -1;
}

const char* tileName(int tile) {
    if (tile < 0 || tile >= kNumTileTypes) return "??";
    return kTileNames[tile];
}

int parseTileName(const std::string& s) {
    for (int i = 0; i < kNumTileTypes; ++i) {
        if (s == kTileNames[i]) return i;
    }
    return -1;
}

std::string TileCounts::toString() const {
    std::string out;
    for (int t = 0; t < kNumTileTypes; ++t) {
        for (int k = 0; k < c[t]; ++k) {
            if (!out.empty()) out.push_back(',');
            out += tileName(t);
        }
    }
    return out.empty() ? "-" : out;
}

TileCounts TileCounts::parse(const std::string& s) {
    TileCounts tc;
    if (s.empty() || s == "-") return tc;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        int t = parseTileName(item);
        if (t >= 0) {
            tc.add(t);
        } else {
            int idx = std::stoi(item);
            if (idx >= 0 && idx < kNumTileTypes) tc.add(idx);
        }
    }
    return tc;
}

const char* actionTypeName(ActionType kind) {
    switch (kind) {
        case ActionType::PASS: return "PASS";
        case ActionType::PLAY: return "PLAY";
        case ActionType::CHI: return "CHI";
        case ActionType::PENG: return "PENG";
        case ActionType::GANG: return "GANG";
        case ActionType::BUGANG: return "BUGANG";
        case ActionType::HU: return "HU";
    }
    return "PASS";
}

int Action::abstractId() const {
    switch (kind) {
        case ActionType::PASS:
            return 0;
        case ActionType::HU:
            return 1;
        case ActionType::PLAY:
            return 2 + std::clamp<int>(tile, 0, 33);
        case ActionType::GANG:
            if (concealedKong) {
                return 36 + std::clamp<int>(tile, 0, 33);
            } else {
                return 70 + std::clamp<int>(tile, 0, 33);
            }
        case ActionType::BUGANG:
            return 104 + std::clamp<int>(tile, 0, 33);
        case ActionType::PENG:
            return 138 + std::clamp<int>(tile, 0, 33) * 34 + std::clamp<int>(discard, 0, 33);
        case ActionType::CHI: {
            int suit = std::clamp<int>(seqStart / 9, 0, 2);
            int rank0 = std::clamp<int>(seqStart % 9, 0, 6);
            int seqIdx = suit * 7 + rank0;
            return 1294 + seqIdx * 34 + std::clamp<int>(discard, 0, 33);
        }
    }
    return 0;
}

std::string Action::toString() const {
    std::ostringstream oss;
    oss << static_cast<int>(kind) << ":" << static_cast<int>(tile) << ":"
        << static_cast<int>(seqStart) << ":" << static_cast<int>(discard);
    return oss.str();
}

Action Action::parse(const std::string& s) {
    std::stringstream ss(s);
    std::string p0, p1, p2, p3;
    std::getline(ss, p0, ':');
    std::getline(ss, p1, ':');
    std::getline(ss, p2, ':');
    std::getline(ss, p3, ':');
    Action a;
    a.kind = static_cast<ActionType>(std::stoi(p0));
    a.tile = static_cast<int8_t>(std::stoi(p1));
    a.seqStart = static_cast<int8_t>(std::stoi(p2));
    a.discard = static_cast<int8_t>(std::stoi(p3));
    return a;
}

std::vector<int8_t> makeFullWall() {
    std::vector<int8_t> wall;
    wall.reserve(kFullWallSize);
    for (int t = 0; t < kNumTileTypes; ++t) {
        for (int k = 0; k < 4; ++k) {
            wall.push_back(static_cast<int8_t>(t));
        }
    }
    return wall;
}

}  // namespace mcr
