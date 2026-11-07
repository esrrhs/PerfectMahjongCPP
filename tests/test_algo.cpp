#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "algo/eval_mcr.h"
#include "algo/ppo.h"
#include "algo/rollout.h"
#include "torch_nn/models.h"

namespace {

TEST(PpoTest, MaskedSoftmaxNormalizesOverLegalActions) {
    std::vector<float> logits = {1.0f, 3.0f, -1e9f, -1e9f, 2.0f};
    std::vector<float> probs(logits.size(), 0.0f);
    algo::maskedSoftmax(logits.data(), static_cast<int>(logits.size()), probs.data());

    EXPECT_NEAR(probs[2], 0.0f, 1e-6f);
    EXPECT_NEAR(probs[3], 0.0f, 1e-6f);
    float sum = probs[0] + probs[1] + probs[4];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    EXPECT_GT(probs[1], probs[4]);
    EXPECT_GT(probs[4], probs[0]);
}

TEST(PpoTest, AssignEpisodeReturnsSeparatesTrajectories) {
    std::vector<algo::Transition> trans(4);
    // Trajectory 10 (steps 0 and 2)
    trans[0].trajectoryId = 10;
    trans[0].value = 0.5f;
    trans[0].reward = 0.1f;
    trans[0].terminal = false;

    trans[2].trajectoryId = 10;
    trans[2].value = 0.8f;
    trans[2].reward = 1.0f;
    trans[2].terminal = true;

    // Trajectory 11 (steps 1 and 3)
    trans[1].trajectoryId = 11;
    trans[1].value = -0.2f;
    trans[1].reward = 0.0f;
    trans[1].terminal = false;

    trans[3].trajectoryId = 11;
    trans[3].value = -0.5f;
    trans[3].reward = -1.0f;
    trans[3].terminal = true;

    algo::assignEpisodeReturns(trans, 1.0f, 0.95f);

    EXPECT_GT(trans[2].adv, 0.0f);
    EXPECT_LT(trans[3].adv, 0.0f);
    EXPECT_NEAR(trans[2].ret, trans[2].adv + trans[2].value, 1e-5f);
    EXPECT_NEAR(trans[0].ret, trans[0].adv + trans[0].value, 1e-5f);
}

TEST(RolloutAndEvalTest, CollectRolloutAndHeuristicDuplicateEval) {
    torch_nn::ModelConfig mcfg;
    mcfg.hidden = 64;
    mcfg.lstmHidden = 32;
    torch_nn::Actor actor(mcfg);
    torch_nn::Critic critic(mcfg);
    actor->eval();
    critic->eval();

    algo::ModelSet models{&actor, &critic};
    algo::RolloutConfig rcfg;
    rcfg.gamesPerUpdate = 4;
    rcfg.threads = 2;
    rcfg.seed = 77;
    rcfg.oracleReward = true;
    rcfg.oracleCoef = 0.1f;

    std::vector<algo::Transition> out;
    algo::RolloutStats stats;
    algo::collectRollout(models, rcfg, out, stats);

    EXPECT_EQ(stats.games, 4);
    EXPECT_FALSE(out.empty());
    EXPECT_EQ(out.size(), static_cast<size_t>(stats.transitions));

    // Verify 4-way duplicate evaluation against heuristic baseline
    algo::McrEvalConfig ecfg;
    ecfg.mode = "heuristic";
    ecfg.decks = 1; // 1 wall = 4 duplicate games
    ecfg.seed = 2026;
    auto res = algo::evaluateAgainstMcrBaseline(actor, ecfg);
    EXPECT_TRUE(res.ok);
    EXPECT_EQ(res.decks, 1);
    EXPECT_EQ(res.games, 4);
    EXPECT_EQ(res.baselineName, "heuristic");
}

}  // namespace
