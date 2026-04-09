#pragma once

#include <random>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "im/shared.hpp"

namespace im {

struct SolutionUnionStats {
    std::vector<BitMask> union_masks;
    std::vector<int> union_counts;
    double avg_spread = 0.0;
};

std::vector<int> degree_heuristic(const Graph& graph, int k);

std::pair<std::vector<int>, std::unordered_map<int, std::string>> build_candidate_pool(
    const std::set<int>& current_solution,
    const std::vector<int>& ordered_vertices,
    const std::vector<int>& out_degree,
    const std::vector<double>& influence_score,
    int c,
    std::mt19937& rng,
    const std::vector<double>* unreached_gain,
    double unreached_ratio,
    double influence_ratio,
    double random_ratio);

std::vector<double> precompute_influence_scores(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    int total_nodes);

std::vector<double> estimate_unreached_gains(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::vector<BitMask>& union_masks,
    const std::vector<int>& candidate_vertices,
    int total_nodes);

SolutionUnionStats compute_solution_union_masks(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& current_solution);

std::unordered_map<int, std::vector<BitMask>> build_leave_one_out_unions(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& current_solution,
    const std::vector<int>& tabu_time,
    int iter_id);

std::pair<std::unordered_map<int, double>, std::unordered_map<int, std::vector<BitMask>>> estimate_removal_profiles(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& current_solution,
    double current_avg_spread,
    const std::vector<int>& tabu_time,
    int iter_id);

std::pair<std::unordered_map<int, double>, std::unordered_map<int, std::vector<BitMask>>> estimate_addition_profiles(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::vector<BitMask>& union_masks,
    const std::vector<int>& union_counts,
    const std::vector<int>& candidate_pool);

std::vector<int> select_removable_vertices(
    const std::unordered_map<int, double>& removal_loss,
    int limit);

std::vector<std::tuple<double, int, int>> rank_swap_moves(
    const std::vector<int>& removable_vertices,
    const std::vector<int>& candidate_vertices,
    const std::unordered_map<int, double>& removal_loss,
    const std::unordered_map<int, double>& addition_gain);

}  // namespace im
