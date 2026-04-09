#include "im/celf.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <vector>

namespace im {
namespace {

struct CELFNodeCompare {
    bool operator()(const CELFNode& left, const CELFNode& right) const {
        if (left.marginal_gain != right.marginal_gain) {
            return left.marginal_gain < right.marginal_gain;
        }
        if (left.iteration != right.iteration) {
            return left.iteration > right.iteration;
        }
        return left.node > right.node;
    }
};

double elapsed_seconds(const std::chrono::steady_clock::time_point& start_time) {
    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    return duration.count();
}

}  // namespace

CelfResult celf(
    const Graph& graph,
    const CelfParams& params,
    const std::vector<ReachabilityGraph>* pre_sampled_subgraphs) {
    if (pre_sampled_subgraphs == nullptr && params.num_subgraphs <= 0) {
        throw std::invalid_argument("num_subgraphs must be a positive integer");
    }
    if (pre_sampled_subgraphs != nullptr && pre_sampled_subgraphs->empty()) {
        throw std::invalid_argument("pre_sampled_subgraphs must be non-empty when provided");
    }

    CelfResult result;
    const int total_nodes = node_count(graph);
    if (total_nodes == 0 || params.k <= 0) {
        return result;
    }

    const int effective_k = std::min(params.k, total_nodes);
    std::vector<ReachabilityGraph> local_subgraphs;
    const std::vector<ReachabilityGraph>* sampled_subgraphs = pre_sampled_subgraphs;

    const std::optional<int> resolved_subgraph_seed =
        params.subgraph_seed.has_value() ? params.subgraph_seed : params.seed;

    if (sampled_subgraphs == nullptr) {
        const auto t_sample_start = std::chrono::steady_clock::now();
        local_subgraphs = generate_live_edge_subgraphs(
            graph,
            params.p,
            params.num_subgraphs,
            resolved_subgraph_seed);
        result.timing.phase_sample_subgraphs = elapsed_seconds(t_sample_start);
        result.timing.subgraph_source = "generated";
        sampled_subgraphs = &local_subgraphs;

        if (params.verbose) {
            std::cout << "[CELF] Phase 0 done: sample "
                      << params.num_subgraphs
                      << " subgraphs in "
                      << result.timing.phase_sample_subgraphs
                      << "s"
                      << '\n';
        }
    } else {
        result.timing.phase_sample_subgraphs = 0.0;
        result.timing.subgraph_source = "external";
        if (params.verbose) {
            std::cout << "[CELF] Phase 0 skipped: reuse "
                      << sampled_subgraphs->size()
                      << " pre-sampled subgraphs"
                      << '\n';
        }
    }

    result.timing.subgraph_count = static_cast<int>(sampled_subgraphs->size());
    if (sampled_subgraphs->empty()) {
        return result;
    }

    const auto t_init_start = std::chrono::steady_clock::now();
    std::priority_queue<CELFNode, std::vector<CELFNode>, CELFNodeCompare> queue;

    for (int u = 0; u < total_nodes; ++u) {
        std::set<int> singleton_seed_set;
        singleton_seed_set.insert(u);
        const double marginal_gain = average_spread_on_live_edge_subgraphs(*sampled_subgraphs, singleton_seed_set);
        queue.push(CELFNode{u, marginal_gain, 0});
    }

    result.timing.phase_init_queue = elapsed_seconds(t_init_start);

    if (params.verbose) {
        std::cout << "[CELF] Phase 1 done: build queue in "
                  << result.timing.phase_init_queue
                  << "s"
                  << '\n';
    }

    double current_spread = 0.0;
    const auto t_loop_start = std::chrono::steady_clock::now();

    while (static_cast<int>(result.selected.size()) < effective_k && !queue.empty()) {
        const int current_iteration = static_cast<int>(result.selected.size());

        CELFNode best_node = queue.top();
        queue.pop();
        const int u = best_node.node;

        if (best_node.iteration == current_iteration) {
            result.selected.insert(u);
            current_spread += best_node.marginal_gain;

            if (params.verbose) {
                std::cout << "[CELF] k="
                          << result.selected.size()
                          << "/"
                          << effective_k
                          << " selected node="
                          << u
                          << ", marginal_gain="
                          << best_node.marginal_gain
                          << '\n';
            }
        } else {
            const auto t_recompute_start = std::chrono::steady_clock::now();
            std::set<int> candidate_seed_set = result.selected;
            candidate_seed_set.insert(u);
            const double new_spread = average_spread_on_live_edge_subgraphs(*sampled_subgraphs, candidate_seed_set);
            result.timing.phase_recompute_only += elapsed_seconds(t_recompute_start);

            const double new_marginal_gain = std::max(0.0, new_spread - current_spread);
            queue.push(CELFNode{u, new_marginal_gain, current_iteration});
            result.timing.stale_recompute_count += 1;
        }
    }

    result.timing.phase_k_loop = elapsed_seconds(t_loop_start);

    if (params.verbose) {
        std::cout << "[CELF] Phase 2 done: k-loop in "
                  << result.timing.phase_k_loop
                  << "s"
                  << '\n';
        std::cout << "[CELF] Recompute detail: count="
                  << result.timing.stale_recompute_count
                  << ", time="
                  << result.timing.phase_recompute_only
                  << "s"
                  << '\n';
    }

    return result;
}

}  // namespace im
