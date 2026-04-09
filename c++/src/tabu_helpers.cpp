#include "im/tabu_helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace im {
namespace {

template <typename T>
std::vector<T> sample_without_replacement(const std::vector<T>& pool, int sample_size, std::mt19937& rng) {
    if (sample_size <= 0) {
        return {};
    }
    if (static_cast<int>(pool.size()) <= sample_size) {
        return pool;
    }

    std::vector<T> shuffled = pool;
    for (int i = 0; i < sample_size; ++i) {
        std::uniform_int_distribution<int> pick(i, static_cast<int>(shuffled.size()) - 1);
        std::swap(shuffled[static_cast<std::size_t>(i)], shuffled[static_cast<std::size_t>(pick(rng))]);
    }
    shuffled.resize(static_cast<std::size_t>(sample_size));
    return shuffled;
}

bool ratio_equal(double left, double right) {
    return std::abs(left - right) <= 1e-9;
}

}  // namespace

std::vector<int> degree_heuristic(const Graph& graph, int k) {
    std::vector<int> nodes(node_count(graph));
    for (int i = 0; i < node_count(graph); ++i) {
        nodes[static_cast<std::size_t>(i)] = i;
    }

    std::stable_sort(nodes.begin(), nodes.end(), [&graph](int lhs, int rhs) {
        const int lhs_degree = static_cast<int>(graph.adj[static_cast<std::size_t>(lhs)].size());
        const int rhs_degree = static_cast<int>(graph.adj[static_cast<std::size_t>(rhs)].size());
        return lhs_degree > rhs_degree;
    });

    if (k < static_cast<int>(nodes.size())) {
        nodes.resize(static_cast<std::size_t>(std::max(0, k)));
    }
    return nodes;
}

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
    double random_ratio) {
    if (c <= 0) {
        return {{}, {}};
    }

    const bool use_unreached_tier = (unreached_gain != nullptr) && (unreached_ratio > 0.0);

    int unreached_quota = 0;
    int influence_quota = 0;
    int random_quota = 0;

    if (use_unreached_tier) {
        unreached_quota = static_cast<int>(c * unreached_ratio);
        influence_quota = static_cast<int>(c * influence_ratio);
        random_quota = c - unreached_quota - influence_quota;
    } else {
        influence_quota = static_cast<int>(c * 0.80);
        random_quota = c - influence_quota;
    }

    if (random_quota < 0) {
        throw std::invalid_argument("candidate quotas exceed pool size c");
    }

    std::vector<int> outside_solution;
    outside_solution.reserve(ordered_vertices.size());
    for (int node : ordered_vertices) {
        if (current_solution.find(node) == current_solution.end()) {
            outside_solution.push_back(node);
        }
    }

    if (outside_solution.empty()) {
        return {{}, {}};
    }

    std::vector<int> unreached_candidates;
    if (use_unreached_tier && unreached_quota > 0) {
        unreached_candidates = outside_solution;
        std::sort(unreached_candidates.begin(), unreached_candidates.end(),
            [&out_degree, &influence_score, unreached_gain](int lhs, int rhs) {
                const double lhs_unreached = (*unreached_gain)[static_cast<std::size_t>(lhs)];
                const double rhs_unreached = (*unreached_gain)[static_cast<std::size_t>(rhs)];
                if (!ratio_equal(lhs_unreached, rhs_unreached)) {
                    return lhs_unreached > rhs_unreached;
                }

                const double lhs_influence = influence_score[static_cast<std::size_t>(lhs)];
                const double rhs_influence = influence_score[static_cast<std::size_t>(rhs)];
                if (!ratio_equal(lhs_influence, rhs_influence)) {
                    return lhs_influence > rhs_influence;
                }

                const int lhs_degree = out_degree[static_cast<std::size_t>(lhs)];
                const int rhs_degree = out_degree[static_cast<std::size_t>(rhs)];
                if (lhs_degree != rhs_degree) {
                    return lhs_degree > rhs_degree;
                }
                return lhs < rhs;
            });

        if (static_cast<int>(unreached_candidates.size()) > unreached_quota) {
            unreached_candidates.resize(static_cast<std::size_t>(unreached_quota));
        }
    }

    std::unordered_set<int> excluded(unreached_candidates.begin(), unreached_candidates.end());

    std::vector<int> influence_candidates;
    for (int node : outside_solution) {
        if (excluded.find(node) == excluded.end()) {
            influence_candidates.push_back(node);
        }
    }

    std::sort(influence_candidates.begin(), influence_candidates.end(),
        [&out_degree, &influence_score](int lhs, int rhs) {
            const double lhs_influence = influence_score[static_cast<std::size_t>(lhs)];
            const double rhs_influence = influence_score[static_cast<std::size_t>(rhs)];
            if (!ratio_equal(lhs_influence, rhs_influence)) {
                return lhs_influence > rhs_influence;
            }
            const int lhs_degree = out_degree[static_cast<std::size_t>(lhs)];
            const int rhs_degree = out_degree[static_cast<std::size_t>(rhs)];
            if (lhs_degree != rhs_degree) {
                return lhs_degree > rhs_degree;
            }
            return lhs < rhs;
        });

    if (static_cast<int>(influence_candidates.size()) > influence_quota) {
        influence_candidates.resize(static_cast<std::size_t>(influence_quota));
    }

    excluded.insert(influence_candidates.begin(), influence_candidates.end());

    std::vector<int> random_pool;
    for (int node : outside_solution) {
        if (excluded.find(node) == excluded.end()) {
            random_pool.push_back(node);
        }
    }

    std::vector<int> random_candidates;
    if (random_quota > 0) {
        random_candidates = sample_without_replacement(random_pool, random_quota, rng);
    }

    std::vector<int> candidates;
    candidates.reserve(static_cast<std::size_t>(c));
    candidates.insert(candidates.end(), unreached_candidates.begin(), unreached_candidates.end());
    candidates.insert(candidates.end(), influence_candidates.begin(), influence_candidates.end());
    candidates.insert(candidates.end(), random_candidates.begin(), random_candidates.end());

    std::unordered_set<int> candidate_set(candidates.begin(), candidates.end());

    if (static_cast<int>(candidates.size()) < c) {
        std::vector<int> fallback_pool;
        for (int node : outside_solution) {
            if (candidate_set.find(node) == candidate_set.end()) {
                fallback_pool.push_back(node);
            }
        }

        if (use_unreached_tier) {
            std::sort(fallback_pool.begin(), fallback_pool.end(),
                [&out_degree, &influence_score, unreached_gain](int lhs, int rhs) {
                    const double lhs_unreached = (*unreached_gain)[static_cast<std::size_t>(lhs)];
                    const double rhs_unreached = (*unreached_gain)[static_cast<std::size_t>(rhs)];
                    if (!ratio_equal(lhs_unreached, rhs_unreached)) {
                        return lhs_unreached > rhs_unreached;
                    }

                    const double lhs_influence = influence_score[static_cast<std::size_t>(lhs)];
                    const double rhs_influence = influence_score[static_cast<std::size_t>(rhs)];
                    if (!ratio_equal(lhs_influence, rhs_influence)) {
                        return lhs_influence > rhs_influence;
                    }

                    const int lhs_degree = out_degree[static_cast<std::size_t>(lhs)];
                    const int rhs_degree = out_degree[static_cast<std::size_t>(rhs)];
                    if (lhs_degree != rhs_degree) {
                        return lhs_degree > rhs_degree;
                    }
                    return lhs < rhs;
                });
        } else {
            std::sort(fallback_pool.begin(), fallback_pool.end(),
                [&out_degree, &influence_score](int lhs, int rhs) {
                    const double lhs_influence = influence_score[static_cast<std::size_t>(lhs)];
                    const double rhs_influence = influence_score[static_cast<std::size_t>(rhs)];
                    if (!ratio_equal(lhs_influence, rhs_influence)) {
                        return lhs_influence > rhs_influence;
                    }

                    const int lhs_degree = out_degree[static_cast<std::size_t>(lhs)];
                    const int rhs_degree = out_degree[static_cast<std::size_t>(rhs)];
                    if (lhs_degree != rhs_degree) {
                        return lhs_degree > rhs_degree;
                    }
                    return lhs < rhs;
                });
        }

        const int need = c - static_cast<int>(candidates.size());
        for (int i = 0; i < need && i < static_cast<int>(fallback_pool.size()); ++i) {
            candidates.push_back(fallback_pool[static_cast<std::size_t>(i)]);
        }
    }

    std::unordered_map<int, std::string> source_by_node;
    for (int node : unreached_candidates) {
        source_by_node[node] = "unreached";
    }
    for (int node : influence_candidates) {
        source_by_node[node] = "influence";
    }
    for (int node : random_candidates) {
        source_by_node[node] = "random";
    }
    for (int node : candidates) {
        if (source_by_node.find(node) == source_by_node.end()) {
            source_by_node[node] = "fallback";
        }
    }

    return {candidates, source_by_node};
}

std::vector<double> precompute_influence_scores(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    int total_nodes) {
    std::vector<double> score_by_node(static_cast<std::size_t>(total_nodes), 0.0);
    if (sampled_subgraphs.empty()) {
        return score_by_node;
    }

    const double sample_count = static_cast<double>(sampled_subgraphs.size());
    for (int node = 0; node < total_nodes; ++node) {
        std::int64_t total = 0;
        for (const auto& live_graph : sampled_subgraphs) {
            total += static_cast<std::int64_t>(live_graph.masks_by_node[static_cast<std::size_t>(node)].count_bits());
        }
        score_by_node[static_cast<std::size_t>(node)] = static_cast<double>(total) / sample_count;
    }

    return score_by_node;
}

std::vector<double> estimate_unreached_gains(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::vector<BitMask>& union_masks,
    const std::vector<int>& candidate_vertices,
    int total_nodes) {
    std::vector<double> unreached_gain(static_cast<std::size_t>(total_nodes), 0.0);
    if (sampled_subgraphs.empty() || union_masks.empty() || candidate_vertices.empty()) {
        return unreached_gain;
    }

    const double sample_count = static_cast<double>(sampled_subgraphs.size());
    for (std::size_t idx = 0; idx < sampled_subgraphs.size(); ++idx) {
        const auto& live_graph = sampled_subgraphs[idx];
        const BitMask& union_mask = union_masks[idx];

        for (int node : candidate_vertices) {
            const BitMask& node_mask = live_graph.masks_by_node[static_cast<std::size_t>(node)];
            unreached_gain[static_cast<std::size_t>(node)] += static_cast<double>(node_mask.and_not_count(union_mask));
        }
    }

    for (int node : candidate_vertices) {
        unreached_gain[static_cast<std::size_t>(node)] /= sample_count;
    }

    return unreached_gain;
}

SolutionUnionStats compute_solution_union_masks(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& current_solution) {
    SolutionUnionStats stats;
    if (sampled_subgraphs.empty()) {
        return stats;
    }

    const std::size_t sample_count = sampled_subgraphs.size();
    const std::size_t word_count = sampled_subgraphs.front().masks_by_node.empty()
        ? 0
        : sampled_subgraphs.front().masks_by_node.front().word_count();

    stats.union_masks.reserve(sample_count);
    stats.union_counts.reserve(sample_count);

    std::int64_t total_count = 0;

    for (const auto& live_graph : sampled_subgraphs) {
        BitMask union_mask(word_count);
        for (int seed : current_solution) {
            union_mask.or_with(live_graph.masks_by_node[static_cast<std::size_t>(seed)]);
        }

        const int count = static_cast<int>(union_mask.count_bits());
        stats.union_masks.push_back(std::move(union_mask));
        stats.union_counts.push_back(count);
        total_count += count;
    }

    stats.avg_spread = static_cast<double>(total_count) / static_cast<double>(sample_count);
    return stats;
}

std::unordered_map<int, std::vector<BitMask>> build_leave_one_out_unions(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& current_solution,
    const std::vector<int>& tabu_time,
    int iter_id) {
    std::unordered_map<int, std::vector<BitMask>> leave_one_out_unions;
    if (sampled_subgraphs.empty() || current_solution.empty()) {
        return leave_one_out_unions;
    }

    const std::vector<int> seeds(current_solution.begin(), current_solution.end());
    const int seed_count = static_cast<int>(seeds.size());
    std::vector<int> free_by_index(static_cast<std::size_t>(seed_count), -1);

    int free_count = 0;
    for (int index = 0; index < seed_count; ++index) {
        const int seed = seeds[static_cast<std::size_t>(index)];
        if (iter_id > tabu_time[static_cast<std::size_t>(seed)]) {
            free_by_index[static_cast<std::size_t>(index)] = seed;
            ++free_count;
        }
    }

    if (free_count == 0) {
        return leave_one_out_unions;
    }

    const std::size_t sample_count = sampled_subgraphs.size();
    const std::size_t word_count = sampled_subgraphs.front().masks_by_node.empty()
        ? 0
        : sampled_subgraphs.front().masks_by_node.front().word_count();

    for (int seed : seeds) {
        if (iter_id > tabu_time[static_cast<std::size_t>(seed)]) {
            leave_one_out_unions[seed] = std::vector<BitMask>(sample_count, BitMask(word_count));
        }
    }

    for (std::size_t sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        const auto& live_graph = sampled_subgraphs[sample_idx];

        std::vector<const BitMask*> seed_masks;
        seed_masks.reserve(static_cast<std::size_t>(seed_count));
        for (int seed : seeds) {
            seed_masks.push_back(&live_graph.masks_by_node[static_cast<std::size_t>(seed)]);
        }

        std::vector<BitMask> prefix_or(static_cast<std::size_t>(seed_count + 1), BitMask(word_count));
        for (int idx = 0; idx < seed_count; ++idx) {
            prefix_or[static_cast<std::size_t>(idx + 1)] = prefix_or[static_cast<std::size_t>(idx)];
            prefix_or[static_cast<std::size_t>(idx + 1)].or_with(*seed_masks[static_cast<std::size_t>(idx)]);
        }

        BitMask suffix_or(word_count);
        for (int idx = seed_count - 1; idx >= 0; --idx) {
            const int removed = free_by_index[static_cast<std::size_t>(idx)];
            if (removed >= 0) {
                BitMask union_mask = prefix_or[static_cast<std::size_t>(idx)];
                union_mask.or_with(suffix_or);
                leave_one_out_unions[removed][sample_idx] = std::move(union_mask);
            }
            suffix_or.or_with(*seed_masks[static_cast<std::size_t>(idx)]);
        }
    }

    return leave_one_out_unions;
}

std::pair<std::unordered_map<int, double>, std::unordered_map<int, std::vector<BitMask>>> estimate_removal_profiles(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::set<int>& current_solution,
    double current_avg_spread,
    const std::vector<int>& tabu_time,
    int iter_id) {
    std::unordered_map<int, double> loss_by_vertex;
    auto leave_one_out_unions = build_leave_one_out_unions(
        sampled_subgraphs,
        current_solution,
        tabu_time,
        iter_id);

    if (leave_one_out_unions.empty()) {
        return {loss_by_vertex, {}};
    }

    const double sample_count = static_cast<double>(sampled_subgraphs.size());
    for (const auto& item : leave_one_out_unions) {
        const int removed = item.first;
        const auto& union_masks = item.second;

        double spread_without_removed = 0.0;
        for (const auto& mask : union_masks) {
            spread_without_removed += static_cast<double>(mask.count_bits());
        }
        spread_without_removed /= sample_count;

        loss_by_vertex[removed] = current_avg_spread - spread_without_removed;
    }

    return {loss_by_vertex, leave_one_out_unions};
}

std::pair<std::unordered_map<int, double>, std::unordered_map<int, std::vector<BitMask>>> estimate_addition_profiles(
    const std::vector<ReachabilityGraph>& sampled_subgraphs,
    const std::vector<BitMask>& union_masks,
    const std::vector<int>& union_counts,
    const std::vector<int>& candidate_pool) {
    std::unordered_map<int, double> gain_by_vertex;
    std::unordered_map<int, std::vector<BitMask>> candidate_masks;

    if (sampled_subgraphs.empty() || candidate_pool.empty()) {
        return {gain_by_vertex, candidate_masks};
    }

    const std::size_t sample_count = sampled_subgraphs.size();
    const std::size_t word_count = sampled_subgraphs.front().masks_by_node.empty()
        ? 0
        : sampled_subgraphs.front().masks_by_node.front().word_count();

    std::unordered_map<int, double> gain_sums;
    for (int node : candidate_pool) {
        gain_sums[node] = 0.0;
        candidate_masks[node] = std::vector<BitMask>(sample_count, BitMask(word_count));
    }

    for (std::size_t idx = 0; idx < sample_count; ++idx) {
        const auto& live_graph = sampled_subgraphs[idx];
        const BitMask& union_mask = union_masks[idx];
        const int union_count = union_counts[idx];

        for (int node : candidate_pool) {
            const BitMask& node_mask = live_graph.masks_by_node[static_cast<std::size_t>(node)];
            candidate_masks[node][idx] = node_mask;
            const int gain = static_cast<int>(union_mask.or_count(node_mask)) - union_count;
            gain_sums[node] += static_cast<double>(gain);
        }
    }

    for (int node : candidate_pool) {
        gain_by_vertex[node] = gain_sums[node] / static_cast<double>(sample_count);
    }

    return {gain_by_vertex, candidate_masks};
}

std::vector<int> select_removable_vertices(
    const std::unordered_map<int, double>& removal_loss,
    int limit) {
    std::vector<int> ranked;
    ranked.reserve(removal_loss.size());

    for (const auto& item : removal_loss) {
        ranked.push_back(item.first);
    }

    std::sort(ranked.begin(), ranked.end(), [&removal_loss](int lhs, int rhs) {
        const double lhs_loss = removal_loss.at(lhs);
        const double rhs_loss = removal_loss.at(rhs);
        if (!ratio_equal(lhs_loss, rhs_loss)) {
            return lhs_loss < rhs_loss;
        }
        return lhs < rhs;
    });

    if (limit < static_cast<int>(ranked.size())) {
        ranked.resize(static_cast<std::size_t>(std::max(0, limit)));
    }
    return ranked;
}

std::vector<std::tuple<double, int, int>> rank_swap_moves(
    const std::vector<int>& removable_vertices,
    const std::vector<int>& candidate_vertices,
    const std::unordered_map<int, double>& removal_loss,
    const std::unordered_map<int, double>& addition_gain) {
    std::vector<std::tuple<double, int, int>> ranked_moves;
    ranked_moves.reserve(removable_vertices.size() * candidate_vertices.size());

    for (int u : removable_vertices) {
        const double loss_u = removal_loss.count(u) ? removal_loss.at(u) : 0.0;
        for (int v : candidate_vertices) {
            const double gain_v = addition_gain.count(v) ? addition_gain.at(v) : 0.0;
            const double score = gain_v - loss_u;
            ranked_moves.emplace_back(score, u, v);
        }
    }

    std::sort(ranked_moves.begin(), ranked_moves.end(),
        [](const std::tuple<double, int, int>& lhs, const std::tuple<double, int, int>& rhs) {
            const double lhs_score = std::get<0>(lhs);
            const double rhs_score = std::get<0>(rhs);
            if (!ratio_equal(lhs_score, rhs_score)) {
                return lhs_score > rhs_score;
            }
            if (std::get<1>(lhs) != std::get<1>(rhs)) {
                return std::get<1>(lhs) < std::get<1>(rhs);
            }
            return std::get<2>(lhs) < std::get<2>(rhs);
        });

    return ranked_moves;
}

}  // namespace im
