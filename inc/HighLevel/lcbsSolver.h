#pragma once

#include "Definitions.h"
#include "HighLevel/HighLevelSolver.h"
#include "LowLevel/Utils/MapQueue.h"
#include "HighLevel/ConflictResolver.h"
#include "LowLevel/RunApex.h"



class lcbsSolver: public HighLevelSolver
{
protected:
    int SOLUTION_NUM;
    MergingStrategy DEFAULT_MS = MergingStrategy::MORE_SLACK;
    MergingStrategy MS;
    double EPS = 0;


    void MergeJointPaths(HighLevelNodePtr node, int solution_num, double max_eps=INT_MAX);
    void LexFilter(std::list<std::pair<CostVector,int>>& apex_idx_combos, double eps1, double eps2, double eps3, double eps4, double eps5, 
                   double eps6, double eps7, double eps8, double eps9, double eps10);
    void NonDomVec(std::list<std::pair<CostVector, int>>& apex_idx_combos, std::vector<CostVector>& real_costs_vector, std::vector<std::vector<size_t>>& ids_vector, 
        std::vector<int>& conflict_nums_vector, MergingStrategy ms, double eps);
    void MergeUntil(std::list<std::pair<CostVector, int>>& apex_idx_combos, std::vector<CostVector>& real_costs_vector, int solution_num, double max_eps=INT_MAX);
    void MergeUntil(std::vector<CostVector>& apex_vectors, std::vector<CostVector>& real_costs_vector, int solution_num, double max_eps=INT_MAX);
    void calculate_CAT(HighLevelNodePtr node, VertexCAT& vertex_cat, EdgeCAT& edge_cat, int agent_id);

    
public:
    virtual ~lcbsSolver() = default;
    lcbsSolver(size_t graph_size, int agent_num, Algorithm algorithm, bool if_eager, int dim, int turn_dim, int turn_cost, int time_limit)
    : HighLevelSolver(graph_size, agent_num, algorithm, if_eager, dim, turn_dim, turn_cost, time_limit){}
    void set_merging_strategy(MergingStrategy ms){MS = ms;}
    void set_solution_num(int solution_num){SOLUTION_NUM = solution_num;}
    void set_eps(double eps1_ = 0.0, double eps2_ = 0.0, double eps3_ = 0.0, double eps4_ = 0.0, double eps5_ = 0.0,
                   double eps6_ = 0.0, double eps7_ = 0.0, double eps8_ = 0.0, double eps9_ = 0.0, double eps10_ = 0.0)
    {
        eps1 = eps1_;
        eps2 = eps2_;
        eps3 = eps3_;
        eps4 = eps4_;
        eps5 = eps5_;
        eps6 = eps6_;
        eps7 = eps7_;
        eps8 = eps8_;
        eps9 = eps9_;
        eps10 = eps10_;
    }
    void printLCBSResult(bool is_success) {
        const int box_width = 50;
        std::string box_color = is_success ? "\033[1;32m" : "\033[1;31m"; // Green for success, Red for failure
        std::string message = is_success
            ? "\033[1;32mLCBS Solver was successful 😄\033[0m"
            : "\033[1;31mLCBS Solver was unsuccessful 🙁\033[0m";

        int text_length = message.length() - 13;
        int padding = (box_width - 2 - text_length) / 2; // 2 for side borders

        // Top border
        std::cout << box_color << "┌";
        for (int i = 0; i < box_width - 2; ++i) std::cout << "─";
        std::cout << "┐\033[0m\n";

        // Empty line
        std::cout << box_color << "│\033[0m";
        std::cout << std::string(box_width - 2, ' ');
        std::cout << box_color << "│\033[0m\n";

        // Centered message line
        std::cout << box_color << "│\033[0m";
        std::cout << std::string(padding, ' ') << message
                << std::string(box_width - 2 - padding - text_length, ' ');
        std::cout << box_color << "│\033[0m\n";

        // Empty line
        std::cout << box_color << "│\033[0m";
        std::cout << std::string(box_width - 2, ' ');
        std::cout << box_color << "│\033[0m\n";

        // Bottom border
        std::cout << box_color << "└";
        for (int i = 0; i < box_width - 2; ++i) std::cout << "─";
        std::cout << "┘\033[0m\n";
    }

    OutputTuple run(std::vector<Edge>& edges, std::vector<std::pair<size_t, size_t>>& start_end, HSolutionID& hsolution_ids, std::vector<CostVector>& hsolution_costs, LoggerPtr& logger) override;
};