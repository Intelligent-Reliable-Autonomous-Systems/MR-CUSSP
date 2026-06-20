#pragma once


#include "Definitions.h"
#include "LexCompare.h"
#include "LowLevel/Utils/MapQueue.h"
#include "HighLevel/ConflictResolver.h"
#include "LowLevel/RunApex.h"


class HighLevelSolver 
{
protected:
    size_t GRAPH_SIZE;
    int AGENT_NUM;
    int DIM;
    Algorithm ALGORITHM;
    LSolver LSOLVER;
    bool EAGER;
    int TIME_LIMIT;

    int TURN_DIM;
    int TURN_COST;

    time_t start_time;

    int solution_size; // number of solutions
    macussp::ObjectiveOrdering lex_ordering_;

public:
    double eps1, eps2, eps3, eps4, eps5, eps6, eps7, eps8, eps9, eps10;
    void set_objective_ordering(const macussp::ObjectiveOrdering& ordering) { lex_ordering_ = ordering; }
    const macussp::ObjectiveOrdering& objective_ordering() const { return lex_ordering_; }

    virtual ~HighLevelSolver() = default;
    HighLevelSolver(size_t graph_size, int agent_num, Algorithm algorithm, bool if_eager, int dim, int turn_dim, int turn_cost, int time_limit, 
                    double eps1 = 0.0, double eps2 = 0.0, double eps3 = 0.0, double eps4 = 0.0, double eps5 = 0.0, double eps6 = 0.0, 
                    double eps7 = 0.0, double eps8 = 0.0, double eps9 = 0.0, double eps10 = 0.0);

    virtual OutputTuple run(std::vector<Edge>& edges, std::vector<std::pair<size_t, size_t>>& start_end, HSolutionID& hsolution_ids, std::vector<CostVector>& hsolution_costs, LoggerPtr& logger) = 0;
};