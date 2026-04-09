#pragma once

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace im {

struct Graph {
    std::vector<std::vector<int>> adj;
    std::vector<int> index_to_node;
    std::unordered_map<int, int> node_to_index;
};

inline int node_count(const Graph& graph) {
    return static_cast<int>(graph.adj.size());
}

inline std::vector<int> sorted_seed_vector(const std::set<int>& seeds) {
    return std::vector<int>(seeds.begin(), seeds.end());
}

}  // namespace im
