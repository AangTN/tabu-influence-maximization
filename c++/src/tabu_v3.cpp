#include "im/tabu_v3.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "im/tabu_helpers.hpp"

namespace im {
namespace {

std::mt19937 make_rng(std::optional<int> seed) {
    if (seed.has_value()) {
        return std::mt19937(static_cast<std::uint32_t>(*seed));
    }
    std::random_device rd;
    return std::mt19937(rd());
}

int py_round_to_int(double value) {
    const double floor_value = std::floor(value);
    const double diff = value - floor_value;

    if (diff > 0.5) {
        return static_cast<int>(floor_value) + 1;
    }
    if (diff < 0.5) {
        return static_cast<int>(floor_value);
    }

    const int floor_int = static_cast<int>(floor_value);
    if ((floor_int % 2) == 0) {
        return floor_int;
    }
    return floor_int + 1;
}

bool ratio_equal(double left, double right) {
    return std::abs(left - right) <= 1e-9;
}

double elapsed_seconds(const std::chrono::steady_clock::time_point& start_time) {
    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    return duration.count();
}

std::vector<int> solution_signature(const std::set<int>& solution) {
    return std::vector<int>(solution.begin(), solution.end());
}

int solution_distance(const std::set<int>& left, const std::set<int>& right) {
    int distance = 0;
    auto lit = left.begin();
    auto rit = right.begin();

    while (lit != left.end() && rit != right.end()) {
        if (*lit == *rit) {
            ++lit;
            ++rit;
        } else if (*lit < *rit) {
            ++distance;
            ++lit;
        } else {
            ++distance;
            ++rit;
        }
    }

    distance += static_cast<int>(std::distance(lit, left.end()));
    distance += static_cast<int>(std::distance(rit, right.end()));
    return distance;
}

void append_history(
    std::deque<HistoryEntry>& history,
    std::size_t max_history_size,
    const std::set<int>& solution,
    double estimate,
    int iter_id) {
    history.push_back(HistoryEntry{solution_signature(solution), estimate, iter_id});
    while (history.size() > max_history_size) {
        history.pop_front();
    }
}

void update_elite_archive(
    std::vector<EliteEntry>& elite_archive,
    const std::set<int>& solution,
    double estimate,
    double mc_spread,
    int iter_id,
    int elite_pool_size) {
    if (elite_pool_size <= 0) {
        return;
    }

    const std::vector<int> signature = solution_signature(solution);

    bool replaced = false;
    for (std::size_t idx = 0; idx < elite_archive.size(); ++idx) {
        if (elite_archive[idx].solution == signature) {
            if (mc_spread > elite_archive[idx].mc_spread ||
                (ratio_equal(mc_spread, elite_archive[idx].mc_spread) && estimate > elite_archive[idx].estimate)) {
                elite_archive[idx] = EliteEntry{signature, estimate, mc_spread, iter_id};
            }
            replaced = true;
            break;
        }
    }

    if (!replaced) {
        elite_archive.push_back(EliteEntry{signature, estimate, mc_spread, iter_id});
    }

    std::sort(elite_archive.begin(), elite_archive.end(),
        [](const EliteEntry& lhs, const EliteEntry& rhs) {
            if (!ratio_equal(lhs.mc_spread, rhs.mc_spread)) {
                return lhs.mc_spread > rhs.mc_spread;
            }
            if (!ratio_equal(lhs.estimate, rhs.estimate)) {
                return lhs.estimate > rhs.estimate;
            }
            return lhs.iter_id < rhs.iter_id;
        });

    if (static_cast<int>(elite_archive.size()) > elite_pool_size) {
        elite_archive.resize(static_cast<std::size_t>(elite_pool_size));
    }
}

struct AnchorCandidate {
    int distance = 0;
    double score = 0.0;
    int iter_id = 0;
    std::vector<int> solution;
    std::string source;
};

std::pair<std::optional<std::set<int>>, std::string> choose_anchor_solution(
    const std::set<int>& current_solution,
    const std::deque<HistoryEntry>& history,
    const std::vector<EliteEntry>& elite_archive,
    std::mt19937& rng) {
    std::vector<AnchorCandidate> candidates;

    for (const auto& item : elite_archive) {
        const std::set<int> candidate(item.solution.begin(), item.solution.end());
        const int distance = solution_distance(current_solution, candidate);
        if (distance <= 0) {
            continue;
        }
        candidates.push_back(AnchorCandidate{distance, item.mc_spread, item.iter_id, item.solution, "elite"});
    }

    for (const auto& item : history) {
        const std::set<int> candidate(item.solution.begin(), item.solution.end());
        const int distance = solution_distance(current_solution, candidate);
        if (distance <= 0) {
            continue;
        }
        candidates.push_back(AnchorCandidate{distance, item.estimate, item.iter_id, item.solution, "history"});
    }

    if (candidates.empty()) {
        return {std::nullopt, "none"};
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const AnchorCandidate& lhs, const AnchorCandidate& rhs) {
            if (lhs.distance != rhs.distance) {
                return lhs.distance > rhs.distance;
            }
            if (!ratio_equal(lhs.score, rhs.score)) {
                return lhs.score > rhs.score;
            }
            return lhs.iter_id < rhs.iter_id;
        });

    const int top_n = std::min(3, static_cast<int>(candidates.size()));
    std::uniform_int_distribution<int> pick(0, top_n - 1);
    const AnchorCandidate& chosen = candidates[static_cast<std::size_t>(pick(rng))];

    std::set<int> anchor_solution(chosen.solution.begin(), chosen.solution.end());
    return {anchor_solution, chosen.source};
}

std::tuple<double, double, double> scale_candidate_mix(
    double unreached_ratio,
    double influence_ratio,
    double random_ratio,
    double diversify_random_ratio,
    bool diversify_mode) {
    if (!diversify_mode) {
        return {unreached_ratio, influence_ratio, random_ratio};
    }

    random_ratio = std::min(1.0, std::max(0.0, diversify_random_ratio));
    const double remainder = std::max(0.0, 1.0 - random_ratio);
    const double base_sum = unreached_ratio + influence_ratio;

    if (base_sum > 0.0) {
        const double scale = remainder / base_sum;
        return {unreached_ratio * scale, influence_ratio * scale, random_ratio};
    }

    return {remainder, 0.0, random_ratio};
}

std::set<int> apply_long_jump(
    const std::set<int>& anchor_solution,
    int jump_count,
    const std::vector<int>& ordered_vertices,
    const std::vector<double>& influence_score,
    const std::vector<int>& out_degree,
    const std::vector<int>& node_frequency,
    const std::vector<double>& unreached_gain,
    std::mt19937& rng,
    double diversify_random_ratio) {
    if (jump_count <= 0 || anchor_solution.empty()) {
        return anchor_solution;
    }

    std::set<int> solution = anchor_solution;

    std::vector<int> removable_ranked(solution.begin(), solution.end());
    std::sort(removable_ranked.begin(), removable_ranked.end(),
        [&node_frequency, &influence_score, &out_degree](int lhs, int rhs) {
            const int lhs_freq = node_frequency[static_cast<std::size_t>(lhs)];
            const int rhs_freq = node_frequency[static_cast<std::size_t>(rhs)];
            if (lhs_freq != rhs_freq) {
                return lhs_freq < rhs_freq;
            }

            const double lhs_influence = influence_score[static_cast<std::size_t>(lhs)];
            const double rhs_influence = influence_score[static_cast<std::size_t>(rhs)];
            if (!ratio_equal(lhs_influence, rhs_influence)) {
                return lhs_influence < rhs_influence;
            }

            const int lhs_degree = out_degree[static_cast<std::size_t>(lhs)];
            const int rhs_degree = out_degree[static_cast<std::size_t>(rhs)];
            if (lhs_degree != rhs_degree) {
                return lhs_degree < rhs_degree;
            }
            return lhs < rhs;
        });

    std::vector<int> outside_solution;
    for (int node : ordered_vertices) {
        if (solution.find(node) == solution.end()) {
            outside_solution.push_back(node);
        }
    }

    if (outside_solution.empty()) {
        return solution;
    }

    std::vector<int> additions_ranked = outside_solution;
    std::sort(additions_ranked.begin(), additions_ranked.end(),
        [&unreached_gain, &influence_score, &out_degree, &node_frequency](int lhs, int rhs) {
            const double lhs_unreached = unreached_gain[static_cast<std::size_t>(lhs)];
            const double rhs_unreached = unreached_gain[static_cast<std::size_t>(rhs)];
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

            const int lhs_freq = node_frequency[static_cast<std::size_t>(lhs)];
            const int rhs_freq = node_frequency[static_cast<std::size_t>(rhs)];
            if (lhs_freq != rhs_freq) {
                return lhs_freq < rhs_freq;
            }
            return lhs < rhs;
        });

    const int actual_swaps = std::min(
        jump_count,
        std::min(static_cast<int>(removable_ranked.size()), static_cast<int>(additions_ranked.size())));
    if (actual_swaps <= 0) {
        return solution;
    }

    for (int i = 0; i < actual_swaps; ++i) {
        solution.erase(removable_ranked[static_cast<std::size_t>(i)]);
    }

    int random_add_count = py_round_to_int(static_cast<double>(actual_swaps) * diversify_random_ratio);
    random_add_count = std::max(0, std::min(actual_swaps, random_add_count));
    const int greedy_add_count = actual_swaps - random_add_count;

    std::vector<int> add_nodes;
    add_nodes.reserve(static_cast<std::size_t>(actual_swaps));
    for (int i = 0; i < greedy_add_count; ++i) {
        add_nodes.push_back(additions_ranked[static_cast<std::size_t>(i)]);
    }

    std::unordered_set<int> chosen_add(add_nodes.begin(), add_nodes.end());
    std::vector<int> remaining_pool;
    for (int node : additions_ranked) {
        if (chosen_add.find(node) == chosen_add.end()) {
            remaining_pool.push_back(node);
        }
    }

    if (random_add_count > 0 && !remaining_pool.empty()) {
        if (static_cast<int>(remaining_pool.size()) <= random_add_count) {
            add_nodes.insert(add_nodes.end(), remaining_pool.begin(), remaining_pool.end());
        } else {
            for (int i = 0; i < random_add_count; ++i) {
                std::uniform_int_distribution<int> pick(i, static_cast<int>(remaining_pool.size()) - 1);
                const int chosen_idx = pick(rng);
                std::swap(remaining_pool[static_cast<std::size_t>(i)], remaining_pool[static_cast<std::size_t>(chosen_idx)]);
            }
            add_nodes.insert(
                add_nodes.end(),
                remaining_pool.begin(),
                remaining_pool.begin() + random_add_count);
        }
    }

    for (int node : add_nodes) {
        solution.insert(node);
    }

    while (solution.size() < anchor_solution.size()) {
        std::vector<int> fallback_pool;
        fallback_pool.reserve(ordered_vertices.size());
        for (int node : ordered_vertices) {
            if (solution.find(node) == solution.end()) {
                fallback_pool.push_back(node);
            }
        }

        if (fallback_pool.empty()) {
            break;
        }

        std::uniform_int_distribution<int> pick(0, static_cast<int>(fallback_pool.size()) - 1);
        solution.insert(fallback_pool[static_cast<std::size_t>(pick(rng))]);
    }

    while (solution.size() > anchor_solution.size()) {
        std::vector<int> current_nodes(solution.begin(), solution.end());
        std::uniform_int_distribution<int> pick(0, static_cast<int>(current_nodes.size()) - 1);
        solution.erase(current_nodes[static_cast<std::size_t>(pick(rng))]);
    }

    return solution;
}

void decay_tabu_times(std::vector<int>& tabu_time, int iter_id, double decay) {
    for (std::size_t node = 0; node < tabu_time.size(); ++node) {
        const int expiry = tabu_time[node];
        if (expiry <= iter_id) {
            continue;
        }

        const int remaining = expiry - iter_id;
        const int new_remaining = py_round_to_int(static_cast<double>(remaining) * decay);
        if (new_remaining <= 0) {
            tabu_time[node] = iter_id - 1;
        } else {
            tabu_time[node] = iter_id + new_remaining;
        }
    }
}

std::pair<bool, std::string> should_restart(
    const std::string& restart_policy,
    int iter_id,
    int no_improve_count,
    int stagnation_trigger,
    int restart_interval,
    bool cycle_hit) {
    if (restart_policy == "periodic") {
        if (restart_interval > 0 && iter_id > 0 && (iter_id % restart_interval) == 0) {
            return {true, "periodic"};
        }
        return {false, "none"};
    }

    if (restart_policy == "adaptive") {
        if (cycle_hit) {
            return {true, "cycle"};
        }
        if (no_improve_count >= stagnation_trigger) {
            return {true, "stagnation"};
        }
        if (restart_interval > 0 && iter_id > 0 && (iter_id % restart_interval) == 0) {
            return {true, "periodic"};
        }
        return {false, "none"};
    }

    if (no_improve_count >= stagnation_trigger) {
        return {true, "stagnation"};
    }
    return {false, "none"};
}

}  // namespace

TabuV3Result tabu_search_v3(const Graph& graph, const TabuV3Params& params) {
    if (params.k < 0) {
        throw std::invalid_argument("k must be non-negative");
    }
    if (params.c <= 0) {
        throw std::invalid_argument("c must be a positive integer");
    }
    if (params.r <= 0) {
        throw std::invalid_argument("r must be a positive integer");
    }
    if (params.t_min < 0 || params.t_max < 0 || params.t_min > params.t_max) {
        throw std::invalid_argument("invalid t_min/t_max");
    }
    if (params.max_iter <= 0) {
        throw std::invalid_argument("max_iter must be a positive integer");
    }
    if (params.max_no_improve <= 0) {
        throw std::invalid_argument("max_no_improve must be a positive integer");
    }
    if (params.backtrack_depth <= 0) {
        throw std::invalid_argument("backtrack_depth must be positive");
    }
    if (params.restart_interval < 0) {
        throw std::invalid_argument("restart_interval must be non-negative");
    }
    if (params.diversify_steps < 0) {
        throw std::invalid_argument("diversify_steps must be non-negative");
    }

    const std::vector<std::pair<std::string, double>> ratio_checks = {
        {"unreached_ratio", params.unreached_ratio},
        {"influence_ratio", params.influence_ratio},
        {"random_ratio", params.random_ratio},
        {"backtrack_trigger_ratio", params.backtrack_trigger_ratio},
        {"backtrack_tenure_decay", params.backtrack_tenure_decay},
        {"jump_ratio", params.jump_ratio},
        {"diversify_random_ratio", params.diversify_random_ratio},
    };

    for (const auto& entry : ratio_checks) {
        if (entry.second < 0.0 || entry.second > 1.0) {
            throw std::invalid_argument(entry.first + " must be in [0, 1]");
        }
    }

    const double candidate_ratio_sum = params.unreached_ratio + params.influence_ratio + params.random_ratio;
    if (std::abs(candidate_ratio_sum - 1.0) > 1e-9) {
        throw std::invalid_argument("candidate ratios must sum to 1.0");
    }

    std::string policy = params.restart_policy;
    std::transform(policy.begin(), policy.end(), policy.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (!(policy == "stagnation" || policy == "periodic" || policy == "adaptive")) {
        throw std::invalid_argument("restart_policy must be one of: stagnation, periodic, adaptive");
    }

    TabuV3Result result;
    result.timing.restart_policy = policy;

    const int total_nodes = node_count(graph);
    if (total_nodes == 0 || params.k == 0) {
        return result;
    }

    const int effective_k = std::min(params.k, total_nodes);
    std::mt19937 rng = make_rng(params.seed);
    const std::optional<int> resolved_subgraph_seed = params.subgraph_seed.has_value() ? params.subgraph_seed : params.seed;

    const auto t_sample_start = std::chrono::steady_clock::now();
    const std::vector<ReachabilityGraph> sampled_subgraphs = generate_live_edge_subgraphs(
        graph,
        params.p,
        params.r,
        resolved_subgraph_seed);
    result.timing.phase_sample_subgraphs = elapsed_seconds(t_sample_start);

    if (params.verbose) {
        std::cout << "[TABU_V3] Phase 0 done: sample "
                  << params.r
                  << " subgraphs in "
                  << result.timing.phase_sample_subgraphs
                  << "s"
                  << '\n';
    }

    const auto t_init_start = std::chrono::steady_clock::now();

    std::set<int> current_solution;
    for (int node : degree_heuristic(graph, effective_k)) {
        current_solution.insert(node);
    }
    std::set<int> best_global_solution = current_solution;

    std::vector<int> ordered_vertices(static_cast<std::size_t>(total_nodes));
    for (int i = 0; i < total_nodes; ++i) {
        ordered_vertices[static_cast<std::size_t>(i)] = i;
    }

    std::vector<int> out_degree(static_cast<std::size_t>(total_nodes), 0);
    for (int node = 0; node < total_nodes; ++node) {
        out_degree[static_cast<std::size_t>(node)] = static_cast<int>(graph.adj[static_cast<std::size_t>(node)].size());
    }

    std::vector<int> tabu_time(static_cast<std::size_t>(total_nodes), 0);
    std::vector<int> node_frequency(static_cast<std::size_t>(total_nodes), 0);
    std::vector<double> influence_score = precompute_influence_scores(sampled_subgraphs, total_nodes);

    double global_estimate_threshold = average_spread_on_live_edge_subgraphs(sampled_subgraphs, best_global_solution);
    const int mc_validation_runs = params.global_eval_runs;
    const std::optional<int> mc_validation_seed = params.global_eval_seed;
    double global_best_spread = monte_carlo_ic(
        graph,
        best_global_solution,
        params.p,
        mc_validation_runs,
        mc_validation_seed);

    const std::size_t max_history_size = static_cast<std::size_t>(std::max(2, params.backtrack_depth + 1));
    std::deque<HistoryEntry> history;
    append_history(history, max_history_size, current_solution, global_estimate_threshold, 0);

    std::vector<EliteEntry> elite_archive;
    update_elite_archive(
        elite_archive,
        best_global_solution,
        global_estimate_threshold,
        global_best_spread,
        0,
        params.elite_pool_size);
    result.timing.elite_archive_size_max = static_cast<int>(elite_archive.size());

    std::map<std::vector<int>, int> seen_signatures;
    seen_signatures[solution_signature(current_solution)] = 0;

    result.timing.phase_init = elapsed_seconds(t_init_start);

    if (params.verbose) {
        std::cout << "[TABU_V3] Phase 1 done: initialize in "
                  << result.timing.phase_init
                  << "s"
                  << '\n';
        std::cout << "[TABU_V3] Start seed_count="
                  << best_global_solution.size()
                  << '\n';
        std::cout << "[TABU_V3] Start global_best (R="
                  << mc_validation_runs
                  << ")="
                  << global_best_spread
                  << '\n';
    }

    const auto t_search_start = std::chrono::steady_clock::now();
    int iter_id = 1;
    int no_improve_count = 0;
    double sigma_current_solution = global_estimate_threshold;
    int diversify_iters_left = 0;
    const int stagnation_trigger = std::max(1, static_cast<int>(std::ceil(params.backtrack_trigger_ratio * params.max_no_improve)));

    while (iter_id <= params.max_iter && no_improve_count < params.max_no_improve) {
        const std::vector<int> signature = solution_signature(current_solution);
        const auto found_signature = seen_signatures.find(signature);
        const bool cycle_hit =
            found_signature != seen_signatures.end() &&
            found_signature->second > 0 &&
            (iter_id - found_signature->second) <= std::max(3, params.backtrack_depth);

        if (cycle_hit) {
            result.timing.cycle_hits += 1;
        }
        seen_signatures[signature] = iter_id;

        const SolutionUnionStats union_stats = compute_solution_union_masks(sampled_subgraphs, current_solution);
        sigma_current_solution = union_stats.avg_spread;

        std::vector<int> outside_solution;
        outside_solution.reserve(ordered_vertices.size());
        for (int node : ordered_vertices) {
            if (current_solution.find(node) == current_solution.end()) {
                outside_solution.push_back(node);
            }
        }

        const std::vector<double> unreached_gain = estimate_unreached_gains(
            sampled_subgraphs,
            union_stats.union_masks,
            outside_solution,
            total_nodes);

        const auto restart_decision = should_restart(
            policy,
            iter_id,
            no_improve_count,
            stagnation_trigger,
            params.restart_interval,
            cycle_hit);

        if (params.use_deep_backtracking && restart_decision.first) {
            const auto anchor_pick = choose_anchor_solution(current_solution, history, elite_archive, rng);
            if (anchor_pick.first.has_value()) {
                const int jump_count = std::max(1, py_round_to_int(params.jump_ratio * static_cast<double>(effective_k)));
                current_solution = apply_long_jump(
                    *anchor_pick.first,
                    jump_count,
                    ordered_vertices,
                    influence_score,
                    out_degree,
                    node_frequency,
                    unreached_gain,
                    rng,
                    params.diversify_random_ratio);

                decay_tabu_times(tabu_time, iter_id, params.backtrack_tenure_decay);
                sigma_current_solution = average_spread_on_live_edge_subgraphs(sampled_subgraphs, current_solution);
                append_history(history, max_history_size, current_solution, sigma_current_solution, iter_id);

                no_improve_count = std::max(0, no_improve_count / 2);
                diversify_iters_left = std::max(diversify_iters_left, params.diversify_steps);
                result.timing.deep_backtracks += 1;
                if (restart_decision.second == "periodic") {
                    result.timing.periodic_restarts += 1;
                }

                if (params.verbose) {
                    std::cout << "[TABU_V3] Iter="
                              << iter_id
                              << ": deep-backtrack ("
                              << restart_decision.second
                              << "), anchor="
                              << anchor_pick.second
                              << ", jump="
                              << jump_count
                              << ", estimate="
                              << sigma_current_solution
                              << '\n';
                }
            }
        }

        const bool in_diversify_mode = diversify_iters_left > 0;
        const auto ratios = scale_candidate_mix(
            params.unreached_ratio,
            params.influence_ratio,
            params.random_ratio,
            params.diversify_random_ratio,
            in_diversify_mode);

        if (in_diversify_mode) {
            diversify_iters_left -= 1;
            result.timing.diversify_iters += 1;
        }

        const auto candidate_pack = build_candidate_pool(
            current_solution,
            ordered_vertices,
            out_degree,
            influence_score,
            params.c,
            rng,
            &unreached_gain,
            std::get<0>(ratios),
            std::get<1>(ratios),
            std::get<2>(ratios));

        const std::vector<int>& candidate_pool = candidate_pack.first;
        const auto& candidate_source = candidate_pack.second;

        if (candidate_pool.empty()) {
            result.timing.stopped_no_valid_move = 1;
            if (params.verbose) {
                std::cout << "[TABU_V3] Iter=" << iter_id << ": empty candidate pool, stop early" << '\n';
            }
            break;
        }

        const auto removal_pack = estimate_removal_profiles(
            sampled_subgraphs,
            current_solution,
            sigma_current_solution,
            tabu_time,
            iter_id);
        const auto& removal_loss = removal_pack.first;
        const auto& leave_one_out_unions = removal_pack.second;

        const std::vector<int> removable_vertices = select_removable_vertices(
            removal_loss,
            std::min(std::max(2, effective_k / 3), static_cast<int>(removal_loss.size())));

        if (removable_vertices.empty()) {
            result.timing.stopped_no_valid_move = 1;
            if (params.verbose) {
                std::cout << "[TABU_V3] Iter=" << iter_id
                          << ": no non-tabu removable vertex in S, stop early"
                          << '\n';
            }
            break;
        }

        const auto addition_pack = estimate_addition_profiles(
            sampled_subgraphs,
            union_stats.union_masks,
            union_stats.union_counts,
            candidate_pool);

        const auto& addition_gain = addition_pack.first;
        const auto& candidate_masks = addition_pack.second;

        if (addition_gain.empty()) {
            result.timing.stopped_no_valid_move = 1;
            break;
        }

        const std::vector<std::tuple<double, int, int>> ranked_moves = rank_swap_moves(
            removable_vertices,
            candidate_pool,
            removal_loss,
            addition_gain);

        if (ranked_moves.empty()) {
            result.timing.stopped_no_valid_move = 1;
            if (params.verbose) {
                std::cout << "[TABU_V3] Iter=" << iter_id << ": no ranked swap move, stop early" << '\n';
            }
            break;
        }

        const int limit_eval = std::min(1200, static_cast<int>(ranked_moves.size()));

        double sigma_best_local = -1.0;
        int best_u = -1;
        int best_v = -1;
        const int sample_count = static_cast<int>(sampled_subgraphs.size());
        std::vector<std::tuple<double, int, int>> valid_moves;

        for (int move_idx = 0; move_idx < limit_eval; ++move_idx) {
            const int u = std::get<1>(ranked_moves[static_cast<std::size_t>(move_idx)]);
            const int v = std::get<2>(ranked_moves[static_cast<std::size_t>(move_idx)]);

            const auto removed_it = leave_one_out_unions.find(u);
            const auto added_it = candidate_masks.find(v);
            if (removed_it == leave_one_out_unions.end() || added_it == candidate_masks.end()) {
                continue;
            }

            const auto& removed_unions = removed_it->second;
            const auto& added_masks = added_it->second;

            double spread_sum = 0.0;
            for (int idx = 0; idx < sample_count; ++idx) {
                BitMask combined_mask = removed_unions[static_cast<std::size_t>(idx)];
                combined_mask.or_with(added_masks[static_cast<std::size_t>(idx)]);
                spread_sum += static_cast<double>(combined_mask.count_bits());
            }

            const double sigma_current = spread_sum / static_cast<double>(sample_count);
            result.timing.neighbor_evaluations += 1;

            const bool is_tabu = iter_id <= tabu_time[static_cast<std::size_t>(v)];
            if (!is_tabu || sigma_current > global_estimate_threshold) {
                valid_moves.emplace_back(sigma_current, u, v);
            }
        }

        if (!valid_moves.empty()) {
            std::sort(valid_moves.begin(), valid_moves.end(),
                [](const std::tuple<double, int, int>& lhs, const std::tuple<double, int, int>& rhs) {
                    if (!ratio_equal(std::get<0>(lhs), std::get<0>(rhs))) {
                        return std::get<0>(lhs) > std::get<0>(rhs);
                    }
                    if (std::get<1>(lhs) != std::get<1>(rhs)) {
                        return std::get<1>(lhs) < std::get<1>(rhs);
                    }
                    return std::get<2>(lhs) < std::get<2>(rhs);
                });

            const int top_k = std::min(12, static_cast<int>(valid_moves.size()));
            std::uniform_int_distribution<int> pick(0, top_k - 1);
            const auto& chosen = valid_moves[static_cast<std::size_t>(pick(rng))];
            sigma_best_local = std::get<0>(chosen);
            best_u = std::get<1>(chosen);
            best_v = std::get<2>(chosen);
        }

        if (best_u < 0 || best_v < 0) {
            if (params.use_deep_backtracking && history.size() > 1) {
                no_improve_count += 1;
                if (params.verbose) {
                    std::cout << "[TABU_V3] Iter="
                              << iter_id
                              << ": no valid move, keep searching with backtracking"
                              << '\n';
                }
                iter_id += 1;
                continue;
            }

            result.timing.stopped_no_valid_move = 1;
            if (params.verbose) {
                std::cout << "[TABU_V3] Iter=" << iter_id << ": no valid move after tabu filter, stop early" << '\n';
            }
            break;
        }

        current_solution.erase(best_u);
        current_solution.insert(best_v);
        sigma_current_solution = sigma_best_local;

        const std::uniform_real_distribution<double> tenure_dist(0.4, 0.8);
        const int tenure = std::max(1, py_round_to_int(tenure_dist(rng) * static_cast<double>(effective_k)));
        tabu_time[static_cast<std::size_t>(best_u)] = iter_id + tenure;
        tabu_time[static_cast<std::size_t>(best_v)] = iter_id + tenure;

        node_frequency[static_cast<std::size_t>(best_u)] += 1;
        node_frequency[static_cast<std::size_t>(best_v)] += 1;

        append_history(history, max_history_size, current_solution, sigma_current_solution, iter_id);

        std::string improved = "no";
        std::string mc_verified = "n/a";

        if (sigma_best_local > global_estimate_threshold) {
            result.timing.global_mc_validations += 1;
            const double candidate_mc_spread = monte_carlo_ic(
                graph,
                current_solution,
                params.p,
                mc_validation_runs,
                mc_validation_seed);

            if (candidate_mc_spread > global_best_spread) {
                best_global_solution = current_solution;
                global_best_spread = candidate_mc_spread;
                global_estimate_threshold = average_spread_on_live_edge_subgraphs(
                    sampled_subgraphs,
                    best_global_solution);

                update_elite_archive(
                    elite_archive,
                    best_global_solution,
                    global_estimate_threshold,
                    global_best_spread,
                    iter_id,
                    params.elite_pool_size);

                result.timing.elite_archive_size_max = std::max(
                    result.timing.elite_archive_size_max,
                    static_cast<int>(elite_archive.size()));

                no_improve_count = 0;
                improved = "yes";
                mc_verified = "yes";
            } else {
                no_improve_count += 1;
                improved = "no";
                mc_verified = "no";
            }
        } else {
            no_improve_count += 1;
        }

        if (params.verbose) {
            std::string selected_source = "unknown";
            const auto source_it = candidate_source.find(best_v);
            if (source_it != candidate_source.end()) {
                selected_source = source_it->second;
            }

            std::cout << "[TABU_V3] Iter="
                      << iter_id
                      << "/"
                      << params.max_iter
                      << ", local_best="
                      << sigma_best_local
                      << ", global_best="
                      << global_best_spread
                      << ", swap="
                      << best_u
                      << "->"
                      << best_v
                      << ", v_source="
                      << selected_source
                      << ", improved="
                      << improved
                      << ", mc_verified="
                      << mc_verified
                      << ", no_improve="
                      << no_improve_count
                      << "/"
                      << params.max_no_improve
                      << ", diversify="
                      << (in_diversify_mode ? 1 : 0)
                      << '\n';
        }

        result.timing.iterations_completed += 1;
        iter_id += 1;
    }

    result.timing.phase_search = elapsed_seconds(t_search_start);
    result.timing.global_best_spread = global_best_spread;
    result.timing.final_no_improve_count = no_improve_count;

    if (params.verbose) {
        std::cout << "[TABU_V3] Phase 2 done: search in "
                  << result.timing.phase_search
                  << "s"
                  << '\n';
        std::cout << "[TABU_V3] Detail: iterations="
                  << result.timing.iterations_completed
                  << ", neighbor_evaluations="
                  << result.timing.neighbor_evaluations
                  << ", global_best="
                  << result.timing.global_best_spread
                  << ", deep_backtracks="
                  << result.timing.deep_backtracks
                  << ", cycle_hits="
                  << result.timing.cycle_hits
                  << '\n';
    }

    result.best_solution = std::move(best_global_solution);
    return result;
}

}  // namespace im
