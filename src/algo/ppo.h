#pragma once

#include <vector>

#include "algo/rollout.h"

namespace algo {

struct PPOStats {
    double pgLoss = 0.0;
    double vLoss = 0.0;
    double entropy = 0.0;
    double lastEpochKL = 0.0;
    double clipFraction = 0.0;
    int epochsCompleted = 0;
};

// Computes GAE(gamma, lambda) advantages and returns per trajectoryId (gameId * 4 + seat)
void assignEpisodeReturns(std::vector<Transition>& tr, float gamma, float lambda);

// Safe masked softmax with fallback to uniform distribution if all logits underflow/NaN
void maskedSoftmax(const float* logits, int n, float* probs);

}  // namespace algo
