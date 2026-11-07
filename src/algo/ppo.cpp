#include "algo/ppo.h"

#include <cmath>
#include <unordered_map>
#include <vector>

namespace algo {

void assignEpisodeReturns(std::vector<Transition>& tr, float gamma, float lambda) {
    int N = static_cast<int>(tr.size());
    if (N == 0) return;

    std::unordered_map<int, int> tidMap;
    std::vector<int> localTid(N, -1);
    int numTraj = 0;
    for (int i = 0; i < N; ++i) {
        int tid = tr[i].trajectoryId;
        if (tid >= 0) {
            auto [it, inserted] = tidMap.try_emplace(tid, numTraj);
            if (inserted) ++numTraj;
            localTid[i] = it->second;
        }
    }

    std::vector<int> head(numTraj, -1);
    std::vector<int> nextTr(N, -1);
    for (int i = 0; i < N; ++i) {
        int t = localTid[i];
        if (t >= 0) {
            nextTr[i] = head[t];
            head[t] = i;
        }
    }

    const float gaeFactor = gamma * lambda;
    for (int t = 0; t < numTraj; ++t) {
        int curr = head[t];
        if (curr < 0) continue;
        float gae = 0.0f;
        float nextValue = 0.0f;
        bool isFirstInBackward = true;
        while (curr >= 0) {
            float vNext = (isFirstInBackward || tr[curr].terminal) ? 0.0f : nextValue;
            float delta = tr[curr].reward + gamma * vNext - tr[curr].value;
            gae = delta + (tr[curr].terminal ? 0.0f : gaeFactor * gae);
            tr[curr].adv = gae;
            tr[curr].ret = tr[curr].adv + tr[curr].value;
            nextValue = tr[curr].value;
            isFirstInBackward = false;
            curr = nextTr[curr];
        }
    }
}

void maskedSoftmax(const float* logits, int n, float* probs) {
    float mx = -1e30f;
    for (int a = 0; a < n; ++a) {
        if (std::isfinite(logits[a]) && logits[a] > mx) mx = logits[a];
    }
    float sum = 0.0f;
    for (int a = 0; a < n; ++a) {
        probs[a] = (logits[a] < -1e8f || !std::isfinite(logits[a])) ? 0.0f : std::exp(logits[a] - mx);
        sum += probs[a];
    }
    if (sum > 0.0f && std::isfinite(sum)) {
        float inv = 1.0f / sum;
        for (int a = 0; a < n; ++a) probs[a] *= inv;
    } else {
        float uni = 1.0f / float(std::max(1, n));
        for (int a = 0; a < n; ++a) probs[a] = uni;
    }
}

}  // namespace algo
