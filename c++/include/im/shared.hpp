#pragma once

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "im/bitmask.hpp"
#include "im/types.hpp"

namespace im {

struct ReachabilityGraph {
    std::vector<BitMask> masks_by_node;
};

Graph load_graph(const std::string& file_path);

std::vector<ReachabilityGraph> generate_live_edge_subgraphs(
    const Graph& graph,
    double p,
    int k,
    std::optional<int> seed);

int spread_on_live_edge_subgraph(
    const ReachabilityGraph& reachability_graph,
    const std::set<int>& seed_set);

double average_spread_on_live_edge_subgraphs(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& seed_set);

double monte_carlo_ic(
    const Graph& graph,
    const std::set<int>& seed_set,
    double p,
    int r,
    std::optional<int> seed);

}  // namespace im
