#include "algo/rollout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "algo/ppo.h"
#include "mcr/features.h"
#include "mcr/game.h"
#include "mcr/oracle.h"

namespace algo {

namespace {

enum SeatRole : int8_t {
    ROLE_LATEST = 0,
    ROLE_HISTORICAL = 1,
    ROLE_HEURISTIC = 2
};

int sampleMaskedProbs(const float* probs, const std::vector<mcr::LegalOption>& opts, mcr::Rng& rng) {
    float r = rng.uniform01();
    float cum = 0.0f;
    int chosen = opts.back().abstractId;
    for (const auto& o : opts) {
        cum += probs[o.abstractId];
        if (r <= cum) {
            chosen = o.abstractId;
            break;
        }
    }
    return chosen;
}

const mcr::LegalOption& findByAbstract(const std::vector<mcr::LegalOption>& opts, int id) {
    for (const auto& o : opts) {
        if (o.abstractId == id) return o;
    }
    return opts.front();
}

const HistoricalActorSnapshot* sampleHistoricalSnapshot(const HistoricalPool* pool, mcr::Rng& rng) {
    if (!pool || pool->empty()) return nullptr;
    bool useRecent = !pool->recent.empty() &&
                     (pool->archive.empty() || rng.uniform01() < 0.7f);
    const auto& vec = useRecent ? pool->recent : pool->archive;
    if (vec.empty()) return nullptr;
    int idx = rng.nextInt(static_cast<int>(vec.size()));
    return vec[idx].get();
}

void workerBatch(const ModelSet& models, int nGames, int gameOffset, uint64_t seed,
                 float historicalProb, float heuristicProb, const HistoricalPool* historicalPool,
                 bool oracleReward, float oracleCoef, float rewardScale,
                 std::vector<Transition>& out, RolloutStats& stats) {
    torch::NoGradGuard no_grad;

    mcr::Rng rng(seed);
    std::vector<mcr::Game> games(nGames);
    std::vector<std::array<SeatRole, mcr::kNumPlayers>> seatRoles(nGames);
    std::vector<std::array<const HistoricalActorSnapshot*, mcr::kNumPlayers>> gameHist(nGames);
    std::vector<std::array<std::vector<Transition>, mcr::kNumPlayers>> traj(nGames);
    std::vector<int> moves(nGames, 0);

    bool hasPool = (historicalPool != nullptr && !historicalPool->empty());

    for (int i = 0; i < nGames; ++i) {
        games[i].deal(rng);
        for (int s = 0; s < mcr::kNumPlayers; ++s) {
            seatRoles[i][s] = ROLE_LATEST;
            gameHist[i][s] = nullptr;
            traj[i][s].reserve(48);
        }

        float p = rng.uniform01();
        if (p < heuristicProb) {
            // 1 learning seat vs 3 heuristic seats (or 2v2 random mix)
            int learnSeat = rng.nextInt(mcr::kNumPlayers);
            for (int s = 0; s < mcr::kNumPlayers; ++s) {
                if (s != learnSeat) seatRoles[i][s] = ROLE_HEURISTIC;
            }
        } else if (hasPool && p < heuristicProb + historicalProb) {
            const HistoricalActorSnapshot* snap = sampleHistoricalSnapshot(historicalPool, rng);
            if (snap) {
                int learnSeat = rng.nextInt(mcr::kNumPlayers);
                for (int s = 0; s < mcr::kNumPlayers; ++s) {
                    if (s != learnSeat) {
                        seatRoles[i][s] = ROLE_HISTORICAL;
                        gameHist[i][s] = snap;
                    }
                }
            }
        }
    }

    struct ActiveStep {
        int gameIdx = 0;
        int seat = 0;
        SeatRole role = ROLE_LATEST;
        const HistoricalActorSnapshot* histSnap = nullptr;
        mcr::EncodedState enc;
        std::vector<mcr::LegalOption> opts;
    };

    std::vector<ActiveStep> active;
    active.reserve(nGames);

    while (true) {
        active.clear();
        for (int i = 0; i < nGames; ++i) {
            if (games[i].over) continue;
            if (moves[i] >= 400) {
                games[i].over = true;
                games[i].phase = mcr::Phase::Terminal;
                continue;
            }

            std::vector<mcr::LegalOption> opts;
            mcr::legalOptions(games[i], opts);
            if (opts.empty()) {
                games[i].over = true;
                games[i].phase = mcr::Phase::Terminal;
                continue;
            }

            int seat = games[i].currentPlayer;
            SeatRole role = seatRoles[i][seat];
            if (role == ROLE_HEURISTIC) {
                // Fast heuristic step immediately without neural forward
                std::vector<mcr::Action> legalActs;
                legalActs.reserve(opts.size());
                for (const auto& o : opts) legalActs.push_back(o.concrete);
                mcr::Action chosenAct = mcr::heuristicMove(games[i], legalActs);
                games[i].step(chosenAct);
                moves[i] += 1;
                continue;
            }

            ActiveStep st;
            st.gameIdx = i;
            st.seat = seat;
            st.role = role;
            st.histSnap = gameHist[i][seat];
            st.enc = mcr::encodeState(games[i]);
            st.opts = std::move(opts);
            active.push_back(std::move(st));
        }

        if (active.empty()) {
            // Check if any games still have heuristic steps to finish
            bool anyAlive = false;
            for (int i = 0; i < nGames; ++i) {
                if (!games[i].over) {
                    anyAlive = true;
                    break;
                }
            }
            if (!anyAlive) break;
            continue;
        }

        // Partition active steps into latest vs historical snapshots
        std::vector<int> latestIndices;
        std::unordered_map<const HistoricalActorSnapshot*, std::vector<int>> histIndices;
        for (int k = 0; k < static_cast<int>(active.size()); ++k) {
            if (active[k].role == ROLE_LATEST || !active[k].histSnap) {
                latestIndices.push_back(k);
            } else {
                histIndices[active[k].histSnap].push_back(k);
            }
        }

        // Batched forward for ROLE_LATEST (Actor + Critic)
        if (!latestIndices.empty()) {
            int64_t B = static_cast<int64_t>(latestIndices.size());
            int64_t totalActions = 0;
            for (int idx : latestIndices) {
                totalActions += static_cast<int64_t>(active[idx].opts.size());
            }

            torch::Tensor xImpT = torch::empty({B, torch_nn::kImpInput}, torch::kFloat32);
            torch::Tensor seqT = torch::empty({B, torch_nn::kLstmSteps, torch_nn::kLstmIn}, torch::kFloat32);
            torch::Tensor extraT = torch::empty({B, torch_nn::kExtraInput}, torch::kFloat32);
            torch::Tensor actFeatT = torch::empty({totalActions, torch_nn::kActionInput}, torch::kFloat32);
            torch::Tensor actSampleT = torch::empty({totalActions}, torch::kLong);
            torch::Tensor actIdT = torch::empty({totalActions}, torch::kLong);

            float* pXImp = xImpT.data_ptr<float>();
            float* pSeq = seqT.data_ptr<float>();
            float* pExtra = extraT.data_ptr<float>();
            float* pActFeat = actFeatT.data_ptr<float>();
            int64_t* pActSample = actSampleT.data_ptr<int64_t>();
            int64_t* pActId = actIdT.data_ptr<int64_t>();

            int64_t curAct = 0;
            for (int64_t b = 0; b < B; ++b) {
                const ActiveStep& st = active[latestIndices[b]];
                float* impRow = pXImp + b * torch_nn::kImpInput;
                for (int j = 0; j < mcr::kNodeBin; ++j) impRow[j] = static_cast<float>(st.enc.imp[j]);
                std::memcpy(impRow + mcr::kNodeBin, st.enc.scalar.data(), mcr::kImpScalars * sizeof(float));

                std::memcpy(pSeq + b * (torch_nn::kLstmSteps * torch_nn::kLstmIn),
                            st.enc.seq.data(),
                            torch_nn::kLstmSteps * torch_nn::kLstmIn * sizeof(float));

                float* extraDst = pExtra + b * torch_nn::kExtraInput;
                for (int j = 0; j < mcr::kExtraBin; ++j) extraDst[j] = static_cast<float>(st.enc.extra[j]);
                std::memcpy(extraDst + mcr::kExtraBin, st.enc.extraScalar.data(), mcr::kExtraScalars * sizeof(float));

                for (const auto& opt : st.opts) {
                    std::memcpy(pActFeat + curAct * torch_nn::kActionInput, opt.feature.data(),
                                torch_nn::kActionInput * sizeof(float));
                    pActSample[curAct] = b;
                    pActId[curAct] = opt.abstractId;
                    curAct++;
                }
            }

            torch::Tensor logitsT = (*models.actor)->forward(xImpT, seqT, actFeatT, actSampleT, actIdT);
            torch::Tensor valuesT = (*models.critic)->forward(xImpT, seqT, extraT);
            const float* pLogits = logitsT.data_ptr<float>();
            const float* pValues = valuesT.data_ptr<float>();

            std::vector<float> probs(torch_nn::kNumActions, 0.0f);
            for (int64_t b = 0; b < B; ++b) {
                ActiveStep& st = active[latestIndices[b]];
                int gi = st.gameIdx;
                int seat = st.seat;
                const float* rowLogits = pLogits + b * torch_nn::kNumActions;

                maskedSoftmax(rowLogits, torch_nn::kNumActions, probs.data());

                int chosenId = sampleMaskedProbs(probs.data(), st.opts, rng);
                float pChosen = std::max(probs[chosenId], 1e-8f);
                const mcr::LegalOption& chosenOpt = findByAbstract(st.opts, chosenId);

                Transition tr;
                tr.imp = st.enc.imp;
                tr.scalar = st.enc.scalar;
                tr.seq = st.enc.seq;
                tr.extra = st.enc.extra;
                tr.extraScalar = st.enc.extraScalar;
                tr.actions.reserve(st.opts.size());
                for (const auto& o : st.opts) {
                    tr.actions.emplace_back(o.abstractId, o.feature);
                }
                tr.action = chosenId;
                tr.trajectoryId = (gameOffset + gi) * mcr::kNumPlayers + seat;
                tr.seat = seat;
                tr.logp = std::log(pChosen);
                tr.value = pValues[b];

                float potBefore = 0.0f;
                if (oracleReward) {
                    potBefore = mcr::oraclePotentialAdvantage(games[gi], seat);
                }

                games[gi].step(chosenOpt.concrete);
                moves[gi] += 1;

                if (oracleReward && !games[gi].over) {
                    float potAfter = mcr::oraclePotentialAdvantage(games[gi], seat);
                    float delta = std::clamp(potAfter - potBefore, -1.0f, 1.0f);
                    tr.reward += oracleCoef * delta;
                }

                traj[gi][seat].push_back(std::move(tr));
            }
        }

        // Batched forward for ROLE_HISTORICAL snapshots
        for (const auto& kv : histIndices) {
            const HistoricalActorSnapshot* snap = kv.first;
            const std::vector<int>& idxs = kv.second;
            int64_t B = static_cast<int64_t>(idxs.size());
            int64_t totalActions = 0;
            for (int idx : idxs) {
                totalActions += static_cast<int64_t>(active[idx].opts.size());
            }

            torch::Tensor xImpT = torch::empty({B, torch_nn::kImpInput}, torch::kFloat32);
            torch::Tensor seqT = torch::empty({B, torch_nn::kLstmSteps, torch_nn::kLstmIn}, torch::kFloat32);
            torch::Tensor actFeatT = torch::empty({totalActions, torch_nn::kActionInput}, torch::kFloat32);
            torch::Tensor actSampleT = torch::empty({totalActions}, torch::kLong);
            torch::Tensor actIdT = torch::empty({totalActions}, torch::kLong);

            float* pXImp = xImpT.data_ptr<float>();
            float* pSeq = seqT.data_ptr<float>();
            float* pActFeat = actFeatT.data_ptr<float>();
            int64_t* pActSample = actSampleT.data_ptr<int64_t>();
            int64_t* pActId = actIdT.data_ptr<int64_t>();

            int64_t curAct = 0;
            for (int64_t b = 0; b < B; ++b) {
                const ActiveStep& st = active[idxs[b]];
                float* impRow = pXImp + b * torch_nn::kImpInput;
                for (int j = 0; j < mcr::kNodeBin; ++j) impRow[j] = static_cast<float>(st.enc.imp[j]);
                std::memcpy(impRow + mcr::kNodeBin, st.enc.scalar.data(), mcr::kImpScalars * sizeof(float));

                std::memcpy(pSeq + b * (torch_nn::kLstmSteps * torch_nn::kLstmIn),
                            st.enc.seq.data(),
                            torch_nn::kLstmSteps * torch_nn::kLstmIn * sizeof(float));

                for (const auto& opt : st.opts) {
                    std::memcpy(pActFeat + curAct * torch_nn::kActionInput, opt.feature.data(),
                                torch_nn::kActionInput * sizeof(float));
                    pActSample[curAct] = b;
                    pActId[curAct] = opt.abstractId;
                    curAct++;
                }
            }

            torch::Tensor logitsT = const_cast<torch_nn::Actor&>(snap->actor)->forward(
                xImpT, seqT, actFeatT, actSampleT, actIdT);
            const float* pLogits = logitsT.data_ptr<float>();

            std::vector<float> probs(torch_nn::kNumActions, 0.0f);
            for (int64_t b = 0; b < B; ++b) {
                ActiveStep& st = active[idxs[b]];
                int gi = st.gameIdx;
                const float* rowLogits = pLogits + b * torch_nn::kNumActions;

                maskedSoftmax(rowLogits, torch_nn::kNumActions, probs.data());

                int chosenId = sampleMaskedProbs(probs.data(), st.opts, rng);
                const mcr::LegalOption& chosenOpt = findByAbstract(st.opts, chosenId);
                games[gi].step(chosenOpt.concrete);
                moves[gi] += 1;
            }
        }
    }

    // Finalize episode returns and stats
    for (int i = 0; i < nGames; ++i) {
        const mcr::Game& g = games[i];
        stats.games += 1;
        stats.totalMoves += moves[i];
        if (g.winner >= 0) {
            stats.wins += 1;
            if (g.selfDrawn) stats.selfDrawnWins += 1;
            stats.totalFan += g.fanCount;
        }

        bool pureSelfPlay = true;
        for (int s = 0; s < mcr::kNumPlayers; ++s) {
            if (seatRoles[i][s] != ROLE_LATEST) pureSelfPlay = false;
        }
        if (pureSelfPlay) {
            stats.pureSelfPlayGames += 1;
        } else {
            for (int s = 0; s < mcr::kNumPlayers; ++s) {
                if (seatRoles[i][s] == ROLE_LATEST) {
                    stats.latestGames += 1;
                    if (g.winner == s) stats.latestWins += 1;
                    stats.latestScore += static_cast<double>(g.scores[s]);
                }
            }
        }

        for (int s = 0; s < mcr::kNumPlayers; ++s) {
            if (seatRoles[i][s] == ROLE_LATEST && !traj[i][s].empty()) {
                traj[i][s].back().terminal = true;
                traj[i][s].back().reward += static_cast<float>(g.scores[s]) * rewardScale;
                for (auto& tr : traj[i][s]) {
                    out.push_back(std::move(tr));
                    stats.transitions += 1;
                }
            }
        }
    }
}

}  // namespace

void collectRollout(const ModelSet& models, const RolloutConfig& cfg,
                    std::vector<Transition>& out, RolloutStats& stats) {
    out.clear();
    stats = {};

    int nThreads = std::max(1, cfg.threads);
    int totalGames = std::max(1, cfg.gamesPerUpdate);
    nThreads = std::min(nThreads, totalGames);

    std::vector<std::vector<Transition>> threadOut(nThreads);
    std::vector<RolloutStats> threadStats(nThreads);
    std::vector<std::thread> workers;
    workers.reserve(nThreads);

    int base = totalGames / nThreads;
    int rem = totalGames % nThreads;
    int offset = 0;

    for (int t = 0; t < nThreads; ++t) {
        int count = base + (t < rem ? 1 : 0);
        int myOffset = offset;
        offset += count;
        uint64_t seed = cfg.seed * 1315423911ULL + static_cast<uint64_t>(t + 1) * 2654435761ULL;

        workers.emplace_back([&, t, count, myOffset, seed]() {
            workerBatch(models, count, myOffset, seed,
                        cfg.historicalProb, cfg.heuristicProb, cfg.historicalPool,
                        cfg.oracleReward, cfg.oracleCoef, cfg.rewardScale,
                        threadOut[t], threadStats[t]);
        });
    }

    for (auto& th : workers) th.join();

    size_t totalTrans = 0;
    for (int t = 0; t < nThreads; ++t) {
        totalTrans += threadOut[t].size();
    }
    out.reserve(totalTrans);

    for (int t = 0; t < nThreads; ++t) {
        for (auto& tr : threadOut[t]) {
            out.push_back(std::move(tr));
        }
        stats.games += threadStats[t].games;
        stats.wins += threadStats[t].wins;
        stats.selfDrawnWins += threadStats[t].selfDrawnWins;
        stats.totalFan += threadStats[t].totalFan;
        stats.totalMoves += threadStats[t].totalMoves;
        stats.pureSelfPlayGames += threadStats[t].pureSelfPlayGames;
        stats.latestGames += threadStats[t].latestGames;
        stats.latestWins += threadStats[t].latestWins;
        stats.latestScore += threadStats[t].latestScore;
        stats.transitions += threadStats[t].transitions;
    }
}

}  // namespace algo
