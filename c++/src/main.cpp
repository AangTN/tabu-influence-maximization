#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "im/celf.hpp"
#include "im/shared.hpp"
#include "im/tabu_helpers.hpp"
#include "im/tabu_v3.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string algo = "celf";
    std::string dataset = "facebook_combined.txt";
    int k = 50;
    double p = 0.05;

    int num_subgraphs = 100;
    std::optional<int> seed = 10;

    int c = 100;
    int r = 100;
    int t_min = 5;
    int t_max = 9;
    int max_iter = 3000;
    int max_no_improve = 1000;

    double tabu_unreached_ratio = 0.80;
    double tabu_influence_ratio = 0.0;
    double tabu_random_ratio = 0.20;

    int tabu_v3_backtrack_depth = 5;
    double tabu_v3_backtrack_trigger_ratio = 0.60;
    double tabu_v3_backtrack_tenure_decay = 0.70;
    int tabu_v3_elite_pool_size = 4;
    double tabu_v3_jump_ratio = 0.10;
    std::string tabu_v3_restart_policy = "adaptive";
    int tabu_v3_restart_interval = 50;
    double tabu_v3_diversify_random_ratio = 0.30;
    int tabu_v3_diversify_steps = 3;
    bool tabu_v3_disable_deep_backtracking = false;

    int r_eval = 1000;
    bool skip_final_eval = false;
};

void print_usage() {
    std::cout
        << "Usage: tabu_im --algo <degree_heuristic|celf|tabu_v3> [options]\n"
        << "\nCore options:\n"
        << "  --algo <degree_heuristic|celf|tabu_v3>\n"
        << "  --dataset <file or name in input_txt>\n"
        << "  --k <int>\n"
        << "  --p <float>\n"
        << "  --seed <int>\n"
        << "  --r-eval <int>\n"
        << "  --skip-final-eval\n"
        << "\nDegree heuristic options:\n"
        << "  (no extra options)\n"
        << "\nCELF options:\n"
        << "  --num-subgraphs <int>\n"
        << "\nTabu_v3 options:\n"
        << "  --c <int> --r <int> --t-min <int> --t-max <int>\n"
        << "  --max-iter <int> --max-no-improve <int>\n"
        << "  --tabu-unreached-ratio <float>\n"
        << "  --tabu-influence-ratio <float>\n"
        << "  --tabu-random-ratio <float>\n"
        << "  --tabu-v3-backtrack-depth <int>\n"
        << "  --tabu-v3-backtrack-trigger-ratio <float>\n"
        << "  --tabu-v3-backtrack-tenure-decay <float>\n"
        << "  --tabu-v3-elite-pool-size <int>\n"
        << "  --tabu-v3-jump-ratio <float>\n"
        << "  --tabu-v3-restart-policy <stagnation|periodic|adaptive>\n"
        << "  --tabu-v3-restart-interval <int>\n"
        << "  --tabu-v3-diversify-random-ratio <float>\n"
        << "  --tabu-v3-diversify-steps <int>\n"
        << "  --tabu-v3-disable-deep-backtracking\n";
}

std::string require_next(int& index, int argc, char** argv, const std::string& option_name) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for option " + option_name);
    }
    ++index;
    return argv[index];
}

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];

        if (option == "--help" || option == "-h") {
            print_usage();
            std::exit(0);
        } else if (option == "--skip-final-eval") {
            args.skip_final_eval = true;
        } else if (option == "--tabu-v3-disable-deep-backtracking") {
            args.tabu_v3_disable_deep_backtracking = true;
        } else if (option == "--algo") {
            args.algo = require_next(i, argc, argv, option);
        } else if (option == "--dataset") {
            args.dataset = require_next(i, argc, argv, option);
        } else if (option == "--k") {
            args.k = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--p") {
            args.p = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--num-subgraphs") {
            args.num_subgraphs = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--seed") {
            args.seed = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--r-eval") {
            args.r_eval = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--c") {
            args.c = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--r") {
            args.r = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--t-min") {
            args.t_min = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--t-max") {
            args.t_max = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--max-iter") {
            args.max_iter = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--max-no-improve") {
            args.max_no_improve = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--tabu-unreached-ratio") {
            args.tabu_unreached_ratio = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--tabu-influence-ratio") {
            args.tabu_influence_ratio = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--tabu-random-ratio") {
            args.tabu_random_ratio = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-backtrack-depth") {
            args.tabu_v3_backtrack_depth = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-backtrack-trigger-ratio") {
            args.tabu_v3_backtrack_trigger_ratio = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-backtrack-tenure-decay") {
            args.tabu_v3_backtrack_tenure_decay = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-elite-pool-size") {
            args.tabu_v3_elite_pool_size = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-jump-ratio") {
            args.tabu_v3_jump_ratio = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-restart-policy") {
            args.tabu_v3_restart_policy = require_next(i, argc, argv, option);
        } else if (option == "--tabu-v3-restart-interval") {
            args.tabu_v3_restart_interval = std::stoi(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-diversify-random-ratio") {
            args.tabu_v3_diversify_random_ratio = std::stod(require_next(i, argc, argv, option));
        } else if (option == "--tabu-v3-diversify-steps") {
            args.tabu_v3_diversify_steps = std::stoi(require_next(i, argc, argv, option));
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }

    if (!(args.algo == "degree_heuristic" || args.algo == "celf" || args.algo == "tabu_v3")) {
        throw std::invalid_argument("--algo must be one of: degree_heuristic, celf, tabu_v3");
    }

    return args;
}

std::string resolve_dataset_path(const std::string& dataset_arg) {
    const fs::path direct(dataset_arg);
    if (fs::exists(direct)) {
        return direct.string();
    }

    const std::vector<fs::path> candidates = {
        fs::path("input_txt") / dataset_arg,
        fs::path("..") / "input_txt" / dataset_arg,
        fs::path("..") / ".." / "input_txt" / dataset_arg,
    };

    for (const auto& candidate : candidates) {
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }

    throw std::runtime_error("Dataset not found: " + dataset_arg);
}

void print_seed_set(const im::Graph& graph, const std::set<int>& seed_set) {
    std::vector<int> original_ids;
    original_ids.reserve(seed_set.size());

    for (int index : seed_set) {
        if (index >= 0 && index < static_cast<int>(graph.index_to_node.size())) {
            original_ids.push_back(graph.index_to_node[static_cast<std::size_t>(index)]);
        }
    }

    std::sort(original_ids.begin(), original_ids.end());

    std::cout << "[";
    for (std::size_t i = 0; i < original_ids.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << original_ids[i];
    }
    std::cout << "]";
}

double elapsed_seconds(const std::chrono::steady_clock::time_point& start_time) {
    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    return duration.count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::string dataset_path = resolve_dataset_path(args.dataset);

        std::cout << "========================================================\n";
        std::cout << "Influence Maximization - C++ Runner\n";
        std::cout << "========================================================\n";
        std::cout << "Algorithm: " << args.algo << '\n';
        std::cout << "Dataset:   " << fs::path(dataset_path).filename().string() << '\n';

        im::Graph graph = im::load_graph(dataset_path);

        const std::optional<int> subgraph_seed = args.seed;
        const std::optional<int> eval_seed = args.seed;

        std::set<int> seed_set;

        im::CelfTiming celf_timing;
        bool has_celf_timing = false;

        im::TabuV3Timing tabu_timing;
        bool has_tabu_timing = false;

        double degree_select_time = 0.0;
        bool has_degree_timing = false;

        if (args.algo == "degree_heuristic") {
            std::cout << "\nRunning Degree Heuristic...\n";

            const auto t_degree_start = std::chrono::steady_clock::now();
            for (int node : im::degree_heuristic(graph, args.k)) {
                seed_set.insert(node);
            }
            degree_select_time = elapsed_seconds(t_degree_start);
            has_degree_timing = true;
        } else if (args.algo == "celf") {
            std::cout << "\nRunning CELF...\n";
            std::cout << "Live-edge subgraphs: " << args.num_subgraphs << '\n';

            im::CelfParams params;
            params.k = args.k;
            params.p = args.p;
            params.num_subgraphs = args.num_subgraphs;
            params.seed = args.seed;
            params.subgraph_seed = subgraph_seed;
            params.verbose = true;

            const im::CelfResult result = im::celf(graph, params, nullptr);
            seed_set = result.selected;
            celf_timing = result.timing;
            has_celf_timing = true;
        } else {
            std::cout << "\nRunning Tabu Search v3...\n";
            std::cout << "Candidate pool size C: " << args.c << '\n';
            std::cout << "Live-edge subgraphs R: " << args.r << '\n';
            std::cout << "Candidate mix (unreached/influence/random): "
                      << args.tabu_unreached_ratio << "/"
                      << args.tabu_influence_ratio << "/"
                      << args.tabu_random_ratio << '\n';

            im::TabuV3Params params;
            params.k = args.k;
            params.p = args.p;
            params.c = args.c;
            params.r = args.r;
            params.t_min = args.t_min;
            params.t_max = args.t_max;
            params.max_iter = args.max_iter;
            params.max_no_improve = args.max_no_improve;
            params.seed = args.seed;
            params.subgraph_seed = subgraph_seed;
            params.global_eval_runs = args.r_eval;
            params.global_eval_seed = eval_seed;
            params.unreached_ratio = args.tabu_unreached_ratio;
            params.influence_ratio = args.tabu_influence_ratio;
            params.random_ratio = args.tabu_random_ratio;
            params.use_deep_backtracking = !args.tabu_v3_disable_deep_backtracking;
            params.backtrack_depth = args.tabu_v3_backtrack_depth;
            params.backtrack_trigger_ratio = args.tabu_v3_backtrack_trigger_ratio;
            params.backtrack_tenure_decay = args.tabu_v3_backtrack_tenure_decay;
            params.elite_pool_size = args.tabu_v3_elite_pool_size;
            params.jump_ratio = args.tabu_v3_jump_ratio;
            params.restart_policy = args.tabu_v3_restart_policy;
            params.restart_interval = args.tabu_v3_restart_interval;
            params.diversify_random_ratio = args.tabu_v3_diversify_random_ratio;
            params.diversify_steps = args.tabu_v3_diversify_steps;
            params.verbose = true;

            const im::TabuV3Result result = im::tabu_search_v3(graph, params);
            seed_set = result.best_solution;
            tabu_timing = result.timing;
            has_tabu_timing = true;
        }

        double spread = 0.0;
        double eval_time = 0.0;
        const bool has_final_eval = !args.skip_final_eval;

        if (has_final_eval) {
            std::cout << "\nEvaluating spread with Monte Carlo (R=" << args.r_eval << ")...\n";
            const auto t_eval_start = std::chrono::steady_clock::now();
            spread = im::monte_carlo_ic(graph, seed_set, args.p, args.r_eval, eval_seed);
            eval_time = elapsed_seconds(t_eval_start);
        } else {
            std::cout << "\nSkipping final Monte Carlo evaluation (--skip-final-eval).\n";
        }

        std::cout << "\n========================================================\n";
        std::cout << "Result\n";
        std::cout << "Seed set S: ";
        print_seed_set(graph, seed_set);
        std::cout << '\n';
        std::cout << "k:          " << args.k << '\n';
        std::cout << "p:          " << args.p << '\n';
        std::cout << std::fixed << std::setprecision(4);
        if (has_final_eval) {
            std::cout << "Spread:     " << spread << '\n';
        } else {
            std::cout << "Spread:     N/A (skipped)\n";
        }

        std::cout << "\nTiming\n";
        if (has_celf_timing) {
            std::cout << "P0 sample subgraphs:" << std::setw(10) << celf_timing.phase_sample_subgraphs << "s\n";
            std::cout << "P1 queue init:      " << celf_timing.phase_init_queue << "s\n";
            std::cout << "P2 k-loop total:    " << celf_timing.phase_k_loop << "s\n";
            std::cout << "Algorithm core time (exclude P0,P3): "
                      << (celf_timing.phase_init_queue + celf_timing.phase_k_loop)
                      << "s\n";
            std::cout << "   recompute time:  " << celf_timing.phase_recompute_only << "s\n";
            std::cout << "   recompute count: " << celf_timing.stale_recompute_count << '\n';
        }

        if (has_tabu_timing) {
            std::cout << "P0 sample subgraphs:" << std::setw(10) << tabu_timing.phase_sample_subgraphs << "s\n";
            std::cout << "P1 initialize:      " << tabu_timing.phase_init << "s\n";
            std::cout << "P2 tabu search:     " << tabu_timing.phase_search << "s\n";
            std::cout << "Algorithm core time (exclude P0,P3): "
                      << (tabu_timing.phase_init + tabu_timing.phase_search)
                      << "s\n";
            std::cout << "   iterations:      " << tabu_timing.iterations_completed << '\n';
            std::cout << "   neighbor evals:  " << tabu_timing.neighbor_evaluations << '\n';
            std::cout << "   global best:     " << tabu_timing.global_best_spread << '\n';
            std::cout << "   no-improve cnt:  " << tabu_timing.final_no_improve_count << '\n';
            std::cout << "   stop no move:    " << tabu_timing.stopped_no_valid_move << '\n';
            std::cout << "   deep backtracks: " << tabu_timing.deep_backtracks << '\n';
            std::cout << "   cycle hits:      " << tabu_timing.cycle_hits << '\n';
            std::cout << "   periodic restart:" << tabu_timing.periodic_restarts << '\n';
            std::cout << "   diversify iters: " << tabu_timing.diversify_iters << '\n';
            std::cout << "   elite size max:  " << tabu_timing.elite_archive_size_max << '\n';
        }

        if (has_degree_timing) {
            std::cout << "P1 degree select:   " << degree_select_time << "s\n";
            std::cout << "Algorithm core time (exclude P0,P3): " << degree_select_time << "s\n";
        }

        if (has_final_eval) {
            std::cout << "P3 final eval:      " << eval_time << "s\n";
        } else {
            std::cout << "P3 final eval:      skipped\n";
        }
        std::cout << "========================================================\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        print_usage();
        return 1;
    }

    return 0;
}
