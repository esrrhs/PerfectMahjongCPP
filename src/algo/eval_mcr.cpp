#include "algo/eval_mcr.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET_VAL (-1)
#endif

#include <torch/torch.h>

#include "mcr/features.h"
#include "mcr/game.h"
#include "mcr/oracle.h"

namespace algo {

namespace {

struct SocketSubsystem {
    bool ok = false;
    SocketSubsystem() {
#if defined(_WIN32)
        WSADATA wsa;
        ok = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
#else
        ok = true;
#endif
    }
    ~SocketSubsystem() {
#if defined(_WIN32)
        if (ok) WSACleanup();
#endif
    }
};

void closeSocket(socket_t s) {
#if defined(_WIN32)
    if (s != INVALID_SOCKET_VAL) closesocket(s);
#else
    if (s >= 0) close(s);
#endif
}

bool sendAll(socket_t s, const std::string& msg) {
    const char* p = msg.data();
    int left = static_cast<int>(msg.size());
    while (left > 0) {
        int n = send(s, p, left, 0);
        if (n <= 0) return false;
        p += n;
        left -= n;
    }
    return true;
}

bool recvLine(socket_t s, std::string& line) {
    line.clear();
    char c;
    while (true) {
        int n = recv(s, &c, 1, 0);
        if (n <= 0) return false;
        if (c == '\n') return true;
        if (c != '\r') line.push_back(c);
    }
}

const char* phaseStr(mcr::Phase p) {
    switch (p) {
        case mcr::Phase::Draw: return "draw";
        case mcr::Phase::Discard: return "discard";
        case mcr::Phase::Claim: return "claim";
        case mcr::Phase::Terminal: return "terminal";
    }
    return "discard";
}

const char* eventKindStr(mcr::EventKind k) {
    switch (k) {
        case mcr::EventKind::DRAW: return "DRAW";
        case mcr::EventKind::PLAY: return "PLAY";
        case mcr::EventKind::CHI: return "CHI";
        case mcr::EventKind::PENG: return "PENG";
        case mcr::EventKind::GANG: return "GANG";
        case mcr::EventKind::ANGANG: return "GANG";
        case mcr::EventKind::BUGANG: return "BUGANG";
        case mcr::EventKind::HU: return "HU";
        case mcr::EventKind::DRAW_GAME: return "DRAW";
    }
    return "PLAY";
}

std::string encodeHand(const mcr::TileCounts& hand) {
    std::string s;
    for (int t = 0; t < mcr::kNumTileTypes; ++t) {
        for (int c = 0; c < hand.c[t]; ++c) {
            if (!s.empty()) s.push_back(',');
            s += std::to_string(t);
        }
    }
    return s.empty() ? "-" : s;
}

std::string encodeDiscards(const mcr::Game& g) {
    std::string out;
    for (int p = 0; p < mcr::kNumPlayers; ++p) {
        if (p > 0) out.push_back(';');
        if (g.discards[p].empty()) {
            out.push_back('-');
            continue;
        }
        for (size_t i = 0; i < g.discards[p].size(); ++i) {
            if (i > 0) out.push_back(',');
            out += std::to_string(int(g.discards[p][i]));
        }
    }
    return out;
}

std::string encodeMelds(const mcr::Game& g) {
    std::string out;
    for (int p = 0; p < mcr::kNumPlayers; ++p) {
        if (p > 0) out.push_back(';');
        if (g.melds[p].empty()) {
            out.push_back('-');
            continue;
        }
        for (size_t i = 0; i < g.melds[p].size(); ++i) {
            if (i > 0) out.push_back('|');
            const mcr::Meld& m = g.melds[p][i];
            int botzoneKind = static_cast<int>(m.kind);
            std::string tilesStr;
            for (int k = 0; k < m.tileCount; ++k) {
                if (k > 0) tilesStr.push_back(',');
                tilesStr += std::to_string(int(m.tiles[k]));
            }
            out += std::to_string(botzoneKind) + ":" + std::to_string(int(m.fromPlayer)) + ":" + tilesStr;
        }
    }
    return out;
}

std::string encodeEvents(const mcr::Game& g) {
    if (g.events.empty()) return "-";
    size_t start = g.events.size() > 64 ? (g.events.size() - 64) : 0;
    std::string out;
    for (size_t i = start; i < g.events.size(); ++i) {
        if (i > start) out.push_back(';');
        const mcr::GameEvent& ev = g.events[i];
        out += std::string(eventKindStr(ev.kind)) + ":" +
               std::to_string(int(ev.player)) + ":" +
               std::to_string(int(ev.tile)) + ":" +
               std::to_string(int(ev.discard));
    }
    return out;
}

std::string encodeLegalActions(const std::vector<mcr::Action>& legal) {
    std::string out;
    for (size_t i = 0; i < legal.size(); ++i) {
        if (i > 0) out.push_back(';');
        const mcr::Action& a = legal[i];
        int s0 = -1, s1 = -1, s2 = -1;
        if (a.kind == mcr::ActionType::CHI) {
            s0 = int(a.seqStart);
            s1 = s0 + 1;
            s2 = s0 + 2;
        }
        out += std::to_string(int(a.kind)) + "," +
               std::to_string(int(a.tile)) + "," +
               std::to_string(s0) + "," +
               std::to_string(s1) + "," +
               std::to_string(s2) + "," +
               std::to_string(int(a.discard));
    }
    return out;
}

bool askMcrServer(socket_t sock, const mcr::Game& g, const std::vector<mcr::Action>& legal,
                  mcr::Action& chosen) {
    std::ostringstream o;
    std::string lastDisc = (g.lastDiscardTile >= 0 && g.lastDiscardPlayer >= 0)
                               ? (std::to_string(g.lastDiscardPlayer) + "," + std::to_string(g.lastDiscardTile))
                               : "-";
    std::string flags = std::to_string(g.wall.empty() ? 1 : 0) + "," +
                        std::to_string(g.aboutKong ? 1 : 0) + "," +
                        std::to_string(g.claimHuOnly ? 1 : 0);

    o << g.currentPlayer << '\t'
      << phaseStr(g.phase) << '\t'
      << g.prevalentWind << '\t'
      << g.wall.size() << '\t'
      << lastDisc << '\t'
      << flags << '\t'
      << encodeHand(g.hand[g.currentPlayer]) << '\t'
      << encodeDiscards(g) << '\t'
      << encodeMelds(g) << '\t'
      << encodeEvents(g) << '\t'
      << encodeLegalActions(legal);

    if (!sendAll(sock, o.str() + "\n")) return false;
    std::string reply;
    if (!recvLine(sock, reply)) return false;

    try {
        int idx = std::stoi(reply);
        if (idx >= 0 && idx < static_cast<int>(legal.size())) {
            chosen = legal[idx];
            return true;
        }
    } catch (...) {
    }
    chosen = legal.front();
    return true;
}

mcr::Action ourMove(const torch_nn::Actor& actor, const mcr::Game& g,
                    const std::vector<mcr::LegalOption>& options) {
    torch::NoGradGuard no_grad;

    int64_t L = static_cast<int64_t>(options.size());
    torch::Tensor xImpT = torch::empty({1, torch_nn::kImpInput}, torch::kFloat32);
    torch::Tensor seqT = torch::empty({1, torch_nn::kLstmSteps, torch_nn::kLstmIn}, torch::kFloat32);
    torch::Tensor actFeatT = torch::empty({L, torch_nn::kActionInput}, torch::kFloat32);
    torch::Tensor actSampleT = torch::zeros({L}, torch::kLong);
    torch::Tensor actIdT = torch::empty({L}, torch::kLong);

    mcr::EncodedState e = mcr::encodeState(g);
    float* pXImp = xImpT.data_ptr<float>();
    for (int j = 0; j < mcr::kNodeBin; ++j) pXImp[j] = static_cast<float>(e.imp[j]);
    std::memcpy(pXImp + mcr::kNodeBin, e.scalar.data(), mcr::kImpScalars * sizeof(float));

    std::memcpy(seqT.data_ptr<float>(), e.seq.data(),
                torch_nn::kLstmSteps * torch_nn::kLstmIn * sizeof(float));

    float* pActFeat = actFeatT.data_ptr<float>();
    int64_t* pActId = actIdT.data_ptr<int64_t>();
    for (int64_t i = 0; i < L; ++i) {
        std::memcpy(pActFeat + i * torch_nn::kActionInput, options[i].feature.data(),
                    torch_nn::kActionInput * sizeof(float));
        pActId[i] = options[i].abstractId;
    }

    torch::Tensor logitsT = const_cast<torch_nn::Actor&>(actor)->forward(
        xImpT, seqT, actFeatT, actSampleT, actIdT);
    const float* pLogits = logitsT.data_ptr<float>();

    int bestIdx = 0;
    float bestL = pLogits[options[0].abstractId];
    for (int64_t i = 1; i < L; ++i) {
        float v = pLogits[options[i].abstractId];
        if (v > bestL) {
            bestL = v;
            bestIdx = static_cast<int>(i);
        }
    }
    return options[bestIdx].concrete;
}

struct EvalTally {
    int games = 0;
    int wins = 0;
    int selfDraws = 0;
    int dealIns = 0;
    long long totalWinFan = 0;
    double score = 0.0;
};

bool playOneGame(const std::vector<int8_t>& wall, int prevalentWind, int ourSeat,
                 const torch_nn::Actor& actor, bool useTcp, socket_t sock,
                 EvalTally& tally) {
    mcr::Game g;
    g.resetWithWall(wall, prevalentWind);

    int guard = 0;
    while (!g.over && guard++ < 400) {
        int seat = g.currentPlayer;
        if (seat == ourSeat) {
            std::vector<mcr::LegalOption> opts;
            mcr::legalOptions(g, opts);
            if (opts.empty()) return false;
            mcr::Action act = (opts.size() == 1) ? opts.front().concrete : ourMove(actor, g, opts);
            g.step(act);
        } else {
            std::vector<mcr::Action> legal = g.legalActions(seat);
            if (legal.empty()) return false;
            mcr::Action act;
            if (legal.size() == 1) {
                act = legal.front();
            } else if (useTcp) {
                if (!askMcrServer(sock, g, legal, act)) return false;
            } else {
                act = mcr::heuristicMove(g, legal);
            }
            g.step(act);
        }
    }
    if (!g.over) return false;

    tally.games += 1;
    tally.score += static_cast<double>(g.scores[ourSeat]);
    if (g.winner == ourSeat) {
        tally.wins += 1;
        if (g.selfDrawn) tally.selfDraws += 1;
        tally.totalWinFan += g.fanCount;
    }
    if (g.loser == ourSeat) {
        tally.dealIns += 1;
    }
    return true;
}

}  // namespace

std::shared_ptr<torch_nn::Actor> cloneActor(const torch_nn::Actor& src) {
    auto dst = std::make_shared<torch_nn::Actor>(src->config);
    torch::NoGradGuard no_grad;
    auto srcParams = src->parameters();
    auto dstParams = (*dst)->parameters();
    for (size_t i = 0; i < srcParams.size(); ++i) {
        dstParams[i].copy_(srcParams[i]);
    }
    return dst;
}

McrEvalResult evaluateAgainstMcrBaseline(std::shared_ptr<const torch_nn::Actor> actor,
                                         const McrEvalConfig& cfg) {
    if (!actor) {
        McrEvalResult r;
        r.error = "null actor";
        return r;
    }
    return evaluateAgainstMcrBaseline(*actor, cfg);
}

McrEvalResult evaluateAgainstMcrBaseline(const torch_nn::Actor& actor,
                                         const McrEvalConfig& cfg) {
    McrEvalResult res;

    // Save model snapshot and metadata if saveDir is set
    if (!cfg.saveDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg.saveDir, ec);
        actor->save(cfg.saveDir + "/actor.bin");
        std::string metaPath = cfg.saveDir + "/meta.txt";
        if (FILE* mf = std::fopen(metaPath.c_str(), "w")) {
            std::fprintf(mf, "update %d\nminutes %.2f\nlabel %s\n",
                         cfg.update, cfg.elapsedMinutes,
                         (cfg.label.empty() ? ("u" + std::to_string(cfg.update)) : cfg.label).c_str());
            std::fclose(mf);
        }

        if (cfg.maxSnapshots > 0) {
            try {
                std::filesystem::path parent = std::filesystem::path(cfg.saveDir).parent_path();
                if (std::filesystem::exists(parent) && std::filesystem::is_directory(parent)) {
                    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> dirs;
                    for (const auto& entry : std::filesystem::directory_iterator(parent)) {
                        if (entry.is_directory()) {
                            dirs.emplace_back(entry.last_write_time(), entry.path());
                        }
                    }
                    if (static_cast<int>(dirs.size()) > cfg.maxSnapshots) {
                        std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
                            return a.first < b.first;
                        });
                        int toRemove = static_cast<int>(dirs.size()) - cfg.maxSnapshots;
                        for (int i = 0; i < toRemove; ++i) {
                            std::filesystem::remove_all(dirs[i].second, ec);
                        }
                    }
                }
            } catch (...) {
            }
        }
    }

    bool useTcp = (cfg.mode == "tcp" || cfg.mode == "auto");
    socket_t sock = INVALID_SOCKET_VAL;
    SocketSubsystem sockSub;

    if (useTcp) {
        if (!sockSub.ok) {
            if (cfg.mode == "tcp") {
                res.error = "network subsystem init failed";
                return res;
            }
            useTcp = false;
        } else {
            sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (sock == INVALID_SOCKET_VAL) {
                if (cfg.mode == "tcp") {
                    res.error = "failed to create socket";
                    return res;
                }
                useTcp = false;
            } else {
#if defined(_WIN32)
                DWORD timeoutMs = 30000;
                setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeoutMs, sizeof(timeoutMs));
                setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeoutMs, sizeof(timeoutMs));
#else
                struct timeval tv;
                tv.tv_sec = 30;
                tv.tv_usec = 0;
                setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
                setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#endif
                sockaddr_in addr{};
                addr.sin_family = AF_INET;
                addr.sin_port = htons(static_cast<uint16_t>(cfg.port));
                if (inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) <= 0 ||
                    connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
                    closeSocket(sock);
                    sock = INVALID_SOCKET_VAL;
                    if (cfg.mode == "tcp") {
                        res.error = "cannot connect to Botzone MCR server at " + cfg.host + ":" +
                                    std::to_string(cfg.port);
                        return res;
                    }
                    useTcp = false;
                }
            }
        }
    }

    res.baselineName = useTcp ? "botzone_mcr" : "heuristic";

    uint64_t seed = cfg.seed;
    if (seed == 0) {
        seed = static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    mcr::Rng rng(seed);

    int targetDecks = std::max(1, cfg.decks);
    EvalTally tally;
    int decksPlayed = 0;

    for (int d = 0; d < targetDecks; ++d) {
        std::vector<int8_t> wall = mcr::makeFullWall(rng);
        int wind = d % mcr::kNumPlayers;

        bool deckOk = true;
        // 4-way duplicate evaluation: candidate rotates through East(0), South(1), West(2), North(3)
        for (int seat = 0; seat < mcr::kNumPlayers; ++seat) {
            if (!playOneGame(wall, wind, seat, actor, useTcp, sock, tally)) {
                res.error = "evaluation game failed on deck " + std::to_string(d) +
                            " seat " + std::to_string(seat);
                deckOk = false;
                break;
            }
        }
        if (!deckOk) break;
        decksPlayed += 1;

        if (cfg.verbose && (d + 1) % 5 == 0) {
            std::printf("deck %d/%d (%d games vs %s)  WP %.3f  ADP %+.2f  DealIn %.3f\n",
                        d + 1, targetDecks, tally.games, res.baselineName.c_str(),
                        tally.games ? double(tally.wins) / tally.games : 0.0,
                        tally.games ? tally.score / tally.games : 0.0,
                        tally.games ? double(tally.dealIns) / tally.games : 0.0);
            std::fflush(stdout);
        }
    }

    if (sock != INVALID_SOCKET_VAL) {
        sendAll(sock, "QUIT\n");
        closeSocket(sock);
    }

    res.decks = decksPlayed;
    res.games = tally.games;
    if (res.games > 0) {
        res.ok = (decksPlayed == targetDecks);
        res.wp = static_cast<double>(tally.wins) / res.games;
        res.adp = tally.score / res.games;
        res.scoreDelta = res.adp * (4.0 / 3.0);
        res.dealInRate = static_cast<double>(tally.dealIns) / res.games;
        res.selfDrawRate = static_cast<double>(tally.selfDraws) / res.games;
        res.avgFan = tally.wins > 0 ? static_cast<double>(tally.totalWinFan) / tally.wins : 0.0;
    }

    if (res.ok && !cfg.csvPath.empty()) {
        std::filesystem::path p(cfg.csvPath);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
        }
        bool needHeader = true;
        if (FILE* probe = std::fopen(cfg.csvPath.c_str(), "rb")) {
            std::fseek(probe, 0, SEEK_END);
            needHeader = (std::ftell(probe) <= 0);
            std::fclose(probe);
        }
        if (FILE* f = std::fopen(cfg.csvPath.c_str(), "ab")) {
            if (needHeader) {
                std::fprintf(f, "update,minutes,timestamp,label,baseline,decks,games,wp,adp,score_delta,deal_in_rate,self_draw_rate,avg_fan\n");
            }
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            char timeBuf[32];
            std::tm tmNow{};
#if defined(_WIN32)
            localtime_s(&tmNow, &t);
#else
            localtime_r(&t, &tmNow);
#endif
            std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &tmNow);
            std::string label = cfg.label.empty() ? ("u" + std::to_string(cfg.update)) : cfg.label;
            std::fprintf(f, "%d,%.2f,%s,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n",
                         cfg.update, cfg.elapsedMinutes, timeBuf, label.c_str(),
                         res.baselineName.c_str(), res.decks, res.games,
                         res.wp, res.adp, res.scoreDelta,
                         res.dealInRate, res.selfDrawRate, res.avgFan);
            std::fclose(f);
        }
    }

    if (!cfg.saveDir.empty() && res.ok) {
        std::string metaPath = cfg.saveDir + "/meta.txt";
        if (FILE* mf = std::fopen(metaPath.c_str(), "a")) {
            std::fprintf(mf, "baseline %s\ngames %d\nwp %.4f\nadp %.4f\nscore_delta %.4f\ndeal_in_rate %.4f\navg_fan %.2f\n",
                         res.baselineName.c_str(), res.games, res.wp, res.adp,
                         res.scoreDelta, res.dealInRate, res.avgFan);
            std::fclose(mf);
        }
    }

    return res;
}

}  // namespace algo
