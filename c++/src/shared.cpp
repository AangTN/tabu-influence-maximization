#include "im/shared.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace im {
namespace {

std::mt19937 make_rng(std::optional<int> seed) {
    if (seed.has_value()) {
        return std::mt19937(static_cast<std::uint32_t>(*seed));
    }
    std::random_device rd;
    return std::mt19937(rd());
}

std::string trim_copy(const std::string& input) {
    const std::size_t begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const std::size_t end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

bool infer_directed_from_filename(const std::string& file_path) {
    std::string name = std::filesystem::path(file_path).filename().string();
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (name.find("facebook_combined") != std::string::npos) {
        return false;
    }
    if (name.find("wiki-vote") != std::string::npos || name.find("wiki_vote") != std::string::npos) {
        return true;
    }
    return true;
}

ReachabilityGraph compute_reachability_graph(const std::vector<std::vector<int>>& live_graph) {
    const int total_nodes = static_cast<int>(live_graph.size());
    const std::size_t word_count = static_cast<std::size_t>((total_nodes + 63) / 64);

    ReachabilityGraph reachability;
    reachability.masks_by_node.reserve(total_nodes);

    for (int source = 0; source < total_nodes; ++source) {
        std::vector<unsigned char> visited(static_cast<std::size_t>(total_nodes), 0U);
        std::queue<int> frontier;

        visited[static_cast<std::size_t>(source)] = 1U;
        frontier.push(source);

        BitMask visited_mask(word_count);
        visited_mask.set_bit(static_cast<std::size_t>(source));

        while (!frontier.empty()) {
            const int current_node = frontier.front();
            frontier.pop();

            for (int neighbor : live_graph[static_cast<std::size_t>(current_node)]) {
                if (visited[static_cast<std::size_t>(neighbor)] == 0U) {
                    visited[static_cast<std::size_t>(neighbor)] = 1U;
                    visited_mask.set_bit(static_cast<std::size_t>(neighbor));
                    frontier.push(neighbor);
                }
            }
        }

        reachability.masks_by_node.push_back(std::move(visited_mask));
    }

    return reachability;
}

}  // namespace

Graph load_graph(const std::string& file_path, std::optional<bool> directed) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open dataset file: " + file_path);
    }

    const bool is_directed = directed.value_or(infer_directed_from_filename(file_path));

    std::vector<std::pair<int, int>> edges;
    std::vector<int> unique_nodes;
    std::unordered_map<int, int> seen_nodes;

    std::string raw_line;
    while (std::getline(input, raw_line)) {
        const std::string line = trim_copy(raw_line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int u = 0;
        int v = 0;
        if (!(iss >> u >> v)) {
            continue;
        }

        edges.emplace_back(u, v);

        if (seen_nodes.find(u) == seen_nodes.end()) {
            seen_nodes[u] = 1;
            unique_nodes.push_back(u);
        }
        if (seen_nodes.find(v) == seen_nodes.end()) {
            seen_nodes[v] = 1;
            unique_nodes.push_back(v);
        }
    }

    std::sort(unique_nodes.begin(), unique_nodes.end());

    Graph graph;
    graph.index_to_node = unique_nodes;
    graph.adj.assign(unique_nodes.size(), {});

    for (std::size_t idx = 0; idx < unique_nodes.size(); ++idx) {
        graph.node_to_index[unique_nodes[idx]] = static_cast<int>(idx);
    }

    for (const auto& edge : edges) {
        const int u_idx = graph.node_to_index[edge.first];
        const int v_idx = graph.node_to_index[edge.second];
        graph.adj[static_cast<std::size_t>(u_idx)].push_back(v_idx);
        if (!is_directed && u_idx != v_idx) {
            graph.adj[static_cast<std::size_t>(v_idx)].push_back(u_idx);
        }
    }

    std::cout << "-> Loaded graph successfully!" << '\n';
    std::cout << "-> Graph type: " << (is_directed ? "directed" : "undirected") << '\n';
    std::cout << "-> Total vertices |V|: " << unique_nodes.size() << '\n';

    return graph;
}

std::vector<ReachabilityGraph> generate_live_edge_subgraphs(
    const Graph& graph,
    double p,
    int k,
    std::optional<int> seed) {
    if (k <= 0) {
        return {};
    }

    std::mt19937 rng = make_rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const int total_nodes = node_count(graph);
    std::vector<ReachabilityGraph> sampled_subgraphs;
    sampled_subgraphs.reserve(static_cast<std::size_t>(k));

    for (int sample_id = 0; sample_id < k; ++sample_id) {
        std::vector<std::vector<int>> sampled_graph(static_cast<std::size_t>(total_nodes));

        for (int node = 0; node < total_nodes; ++node) {
            auto& live_neighbors = sampled_graph[static_cast<std::size_t>(node)];
            const auto& neighbors = graph.adj[static_cast<std::size_t>(node)];
            live_neighbors.reserve(neighbors.size());

            for (int neighbor : neighbors) {
                if (dist(rng) < p) {
                    live_neighbors.push_back(neighbor);
                }
            }
        }

        sampled_subgraphs.push_back(compute_reachability_graph(sampled_graph));
    }

    return sampled_subgraphs;
}

int spread_on_live_edge_subgraph(
    const ReachabilityGraph& reachability_graph,
    const std::set<int>& seed_set) {
    if (reachability_graph.masks_by_node.empty()) {
        return 0;
    }

    const std::size_t word_count = reachability_graph.masks_by_node.front().word_count();
    BitMask combined_mask(word_count);

    for (int seed : seed_set) {
        if (seed >= 0 && seed < static_cast<int>(reachability_graph.masks_by_node.size())) {
            combined_mask.or_with(reachability_graph.masks_by_node[static_cast<std::size_t>(seed)]);
        }
    }

    return static_cast<int>(combined_mask.count_bits());
}

double average_spread_on_live_edge_subgraphs(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& seed_set) {
    if (sampled_subgraphs.empty()) {
        return 0.0;
    }

    std::int64_t total_spread = 0;
    for (const auto& sample : sampled_subgraphs) {
        total_spread += spread_on_live_edge_subgraph(sample, seed_set);
    }

    return static_cast<double>(total_spread) / static_cast<double>(sampled_subgraphs.size());
}

double monte_carlo_ic(
    const Graph& graph,
    const std::set<int>& seed_set,
    double p,
    int r,
    std::optional<int> seed) {
    if (r <= 0) {
        return 0.0;
    }

    const int total_nodes = node_count(graph);
    if (total_nodes == 0) {
        return 0.0;
    }

    std::vector<unsigned char> active_nodes(static_cast<std::size_t>(total_nodes), 0U);
    std::mt19937 rng = make_rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const std::vector<int> seeds = sorted_seed_vector(seed_set);
    double total_spread = 0.0;

    for (int run = 0; run < r; ++run) {
        std::queue<int> new_active;
        std::vector<int> visited_in_this_run;
        visited_in_this_run.reserve(static_cast<std::size_t>(total_nodes / 10 + 1));

        for (int source : seeds) {
            if (source >= 0 && source < total_nodes && active_nodes[static_cast<std::size_t>(source)] == 0U) {
                active_nodes[static_cast<std::size_t>(source)] = 1U;
                new_active.push(source);
                visited_in_this_run.push_back(source);
            }
        }

        int current_spread = static_cast<int>(visited_in_this_run.size());

        while (!new_active.empty()) {
            const int current_node = new_active.front();
            new_active.pop();

            for (int neighbor : graph.adj[static_cast<std::size_t>(current_node)]) {
                if (active_nodes[static_cast<std::size_t>(neighbor)] == 0U && dist(rng) < p) {
                    active_nodes[static_cast<std::size_t>(neighbor)] = 1U;
                    new_active.push(neighbor);
                    visited_in_this_run.push_back(neighbor);
                    ++current_spread;
                }
            }
        }

        total_spread += static_cast<double>(current_spread);

        for (int node : visited_in_this_run) {
            active_nodes[static_cast<std::size_t>(node)] = 0U;
        }
    }

    return total_spread / static_cast<double>(r);
}

}  // namespace im
