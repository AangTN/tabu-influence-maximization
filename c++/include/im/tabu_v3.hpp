#pragma once

#include <deque>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "im/shared.hpp"

namespace im {

struct HistoryEntry {
    std::vector<int> solution;
    double estimate = 0.0;
    int iter_id = 0;
};

struct EliteEntry {
    std::vector<int> solution;
    double estimate = 0.0;
    double mc_spread = 0.0;
    int iter_id = 0;
};

struct TabuV3Timing {
    double phase_sample_subgraphs = 0.0;
    double phase_init = 0.0;
    double phase_search = 0.0;
    int iterations_completed = 0;
    int neighbor_evaluations = 0;
    int global_mc_validations = 0;
    double global_best_spread = 0.0;
    int final_no_improve_count = 0;
    int stopped_no_valid_move = 0;
    int deep_backtracks = 0;
    int cycle_hits = 0;
    int periodic_restarts = 0;
    int diversify_iters = 0;
    int elite_archive_size_max = 0;
    std::string restart_policy = "adaptive";
};

struct TabuV3Result {
    std::set<int> best_solution;
    TabuV3Timing timing;
};

struct TabuV3Params {
    int k = 50;
    double p = 0.05;
    int c = 100;
    int r = 100;
    int t_min = 5;
    int t_max = 9;
    int max_iter = 3000;
    int max_no_improve = 1000;
    std::optional<int> seed = 10;
    std::optional<int> subgraph_seed = 10;
    int global_eval_runs = 1000;
    std::optional<int> global_eval_seed = 10;
    double unreached_ratio = 0.80;
    double influence_ratio = 0.0;
    double random_ratio = 0.20;
    bool use_deep_backtracking = true;
    int backtrack_depth = 5;
    double backtrack_trigger_ratio = 0.60;
    double backtrack_tenure_decay = 0.70;
    int elite_pool_size = 4;
    double jump_ratio = 0.10;
    std::string restart_policy = "adaptive";
    int restart_interval = 50;
    double diversify_random_ratio = 0.30;
    int diversify_steps = 3;
    bool verbose = true;
};

TabuV3Result tabu_search_v3(const Graph& graph, const TabuV3Params& params);

}  // namespace im
