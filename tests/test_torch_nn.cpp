#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include <torch/torch.h>

#include "algo/rollout.h"
#include "torch_nn/models.h"
#include "torch_nn/ppo_learner.h"
#include "torch_nn/tensor_bridge.h"

namespace {

TEST(TorchNNTest, ForwardBackwardAndCheckpointRoundTrip) {
    torch_nn::ModelConfig mcfg;
    mcfg.hidden = 64;
    mcfg.lstmHidden = 32;

    torch_nn::Actor actor(mcfg);
    torch_nn::Critic critic(mcfg);

    algo::ModelSet models{&actor, &critic};
    algo::RolloutConfig rcfg;
    rcfg.gamesPerUpdate = 2;
    rcfg.threads = 1;
    rcfg.seed = 101;

    std::vector<algo::Transition> transitions;
    algo::RolloutStats rstats;
    algo::collectRollout(models, rcfg, transitions, rstats);
    ASSERT_FALSE(transitions.empty());

    torch_nn::PPOConfig pcfg;
    pcfg.lr = 3e-4f;
    pcfg.epochs = 1;
    pcfg.minibatch = 64;
    torch_nn::PPOLearner learner(actor, critic, pcfg, torch::kCPU);

    auto batch = torch_nn::transitionsToBatch(transitions, 0.995f, 0.95f, torch::kCPU);
    auto stats = learner.update(batch);
    EXPECT_GE(stats.epochsCompleted, 1);
    EXPECT_TRUE(std::isfinite(stats.pgLoss));
    EXPECT_TRUE(std::isfinite(stats.vLoss));

    // Save and reload checkpoint
    auto tmpDir = std::filesystem::temp_directory_path() / "perfectmahjong_test_ckpt";
    std::filesystem::create_directories(tmpDir);
    std::string actPath = (tmpDir / "actor.pt").string();
    std::string critPath = (tmpDir / "critic.pt").string();
    actor->save(actPath);
    critic->save(critPath);

    torch_nn::Actor actor2(mcfg);
    torch_nn::Critic critic2(mcfg);
    actor2->load(actPath);
    critic2->load(critPath);

    auto p1 = actor->parameters();
    auto p2 = actor2->parameters();
    ASSERT_EQ(p1.size(), p2.size());
    for (size_t i = 0; i < p1.size(); ++i) {
        EXPECT_TRUE(torch::allclose(p1[i], p2[i]));
    }

    std::error_code ec;
    std::filesystem::remove_all(tmpDir, ec);
}

}  // namespace
