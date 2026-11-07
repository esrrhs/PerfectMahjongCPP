#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mcr/tiles.h"
#include "torch_nn/models.h"

namespace algo {

struct McrEvalConfig {
    std::string mode = "tcp";    // "tcp" (Botzone MCR server), "heuristic" (C++ heuristic), or "auto"
    std::string host = "127.0.0.1";
    int port = 18766;
    int decks = 25;              // 25 walls x 4 duplicate seats = 100 games
    uint64_t seed = 2026;        // fixed seed for reproducible duplicate walls
    std::string csvPath = "";    // if not empty, append evaluation metrics to CSV
    std::string saveDir = "";    // if not empty, save cloned model to disk for backtracking
    int maxSnapshots = 0;        // 0 = keep all, >0 = retain most recent N snapshots
    std::string label = "";      // label or tag
    int update = 0;              // training update step
    double elapsedMinutes = 0.0; // elapsed minutes since training start
    bool verbose = false;        // print progress during duplicate evaluation
};

struct McrEvalResult {
    bool ok = false;
    std::string baselineName = "botzone_mcr";
    int decks = 0;               // number of duplicate walls completed
    int games = 0;               // 4 * decks
    double wp = 0.0;             // candidate win rate (wins / games)
    double adp = 0.0;            // candidate average MCR score per game
    double scoreDelta = 0.0;     // candidate avg score minus baseline avg score (4/3 * adp)
    double dealInRate = 0.0;     // rate at which candidate dealt into opponent Ron
    double selfDrawRate = 0.0;   // rate at which candidate won by Tsumo
    double avgFan = 0.0;         // average fan on candidate's winning hands
    std::string error;
};

// Deep copies an Actor model onto the heap for thread-safe or checkpoint evaluation.
std::shared_ptr<torch_nn::Actor> cloneActor(const torch_nn::Actor& src);

// Evaluates the given Actor using 4-way duplicate rotation per wall against Botzone MCR TCP server or C++ Heuristic.
McrEvalResult evaluateAgainstMcrBaseline(const torch_nn::Actor& actor,
                                         const McrEvalConfig& cfg);

McrEvalResult evaluateAgainstMcrBaseline(std::shared_ptr<const torch_nn::Actor> actor,
                                         const McrEvalConfig& cfg);

}  // namespace algo
