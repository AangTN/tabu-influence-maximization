#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "im/shared.hpp"
#include "im/types.hpp"

namespace im {

struct CELFNode {
    int node = -1;
    double marginal_gain = 0.0;
    int iteration = 0;
};

struct CelfTiming {
    double phase_sample_subgraphs = 0.0;
    double phase_init_queue = 0.0;
    double phase_k_loop = 0.0;
    double phase_recompute_only = 0.0;
    int stale_recompute_count = 0;
    std::string subgraph_source = "generated";
    int subgraph_count = 0;
};

struct CelfResult {
    std::set<int> selected;
    CelfTiming timing;
};

struct CelfParams {
    int k = 50;
    double p = 0.05;
    int num_subgraphs = 100;
    std::optional<int> seed = 10;
    std::optional<int> subgraph_seed = 10;
    bool verbose = true;
};

CelfResult celf(
    const Graph& graph,
    const CelfParams& params,
    const std::vector<ReachabilityGraph>* pre_sampled_subgraphs = nullptr);

}  // namespace im
