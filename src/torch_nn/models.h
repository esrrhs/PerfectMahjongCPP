#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

#include "mcr/features.h"

namespace torch_nn {

constexpr int64_t kImpInput = mcr::kNodeSize;        // 1520
constexpr int64_t kExtraInput = mcr::kExtraSize;     // 612
constexpr int64_t kLstmSteps = mcr::kLstmSteps;      // 24
constexpr int64_t kLstmIn = mcr::kLstmIn;            // 88
constexpr int64_t kNumActions = mcr::kNumActions;    // 2048
constexpr int64_t kActionInput = mcr::kActionSize;   // 96

struct ModelConfig {
    int64_t hidden = 256;
    int64_t lstmHidden = 128;
};

// Actor (Imperfect Information Policy Network)
// LSTM(88 -> lstmHidden) over 24 public discard/meld steps
// node = concat(xImp, lstm_h_last) -> [B, 1520 + lstmHidden]
// joint = concat(node[actionSample], actionFeat) -> [L, 1520 + lstmHidden + 96]
// Decomposed first linear layer + 4-layer MLP -> scatter scores [L, 1] to logits [B, 2048]
class ActorImpl : public torch::nn::Module {
public:
    ActorImpl() : ActorImpl(ModelConfig{}) {}
    explicit ActorImpl(const ModelConfig& cfg);

    torch::Tensor forward(const torch::Tensor& xImp,
                          const torch::Tensor& seq,
                          const torch::Tensor& actionFeat,
                          const torch::Tensor& actionSample,
                          const torch::Tensor& actionId) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, head{nullptr};
};
TORCH_MODULE(Actor);

// Critic ("开天眼" Omniscient Perfect Information Value Network)
// LSTM(88 -> lstmHidden) over 24 public discard/meld steps
// node = concat(xImp, lstm_h_last) -> [B, 1520 + lstmHidden]
// all = concat(node, extra) -> [B, 1520 + lstmHidden + 612]
// 4-layer MLP -> scalar value estimate [B, 1]
class CriticImpl : public torch::nn::Module {
public:
    CriticImpl() : CriticImpl(ModelConfig{}) {}
    explicit CriticImpl(const ModelConfig& cfg);

    torch::Tensor forward(const torch::Tensor& xImp,
                          const torch::Tensor& seq,
                          const torch::Tensor& extra) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear c1{nullptr}, c2{nullptr}, c3{nullptr}, c4{nullptr}, out{nullptr};
};
TORCH_MODULE(Critic);

}  // namespace torch_nn
