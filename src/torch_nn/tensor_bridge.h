#pragma once

#include <torch/torch.h>
#include <vector>

#include "algo/rollout.h"
#include "torch_nn/ppo_learner.h"

namespace torch_nn {

BatchTensors transitionsToBatch(const std::vector<algo::Transition>& tr,
                                float gamma,
                                float lambda,
                                torch::Device device = torch::kCPU);

}  // namespace torch_nn
