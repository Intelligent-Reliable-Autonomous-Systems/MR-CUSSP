#include "HighLevel/scalarizationSolver.h"
#include "LexCompare.h"
#include "Utils.h"
#include "HighLevel/ConflictResolver.h"
#include <cstdlib>
#include <list>
#include <algorithm>
#include <set>
#include <random>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

extern std::unordered_map<size_t, std::vector<int>> id2coord;
extern std::string map_name;
extern std::ofstream output;

// Verify vertex/edge conflicts on solver IDs with "wait-at-goal" padding.
static bool verify_joint_paths_conflict_free(const std::vector<std::vector<size_t>>& A)
{
    const int n = static_cast<int>(A.size());
    size_t makespan = 0;
    for (const auto& p : A) makespan = std::max(makespan, p.size());
    if (makespan == 0) return false;

    auto at = [&](const std::vector<size_t>& p, size_t t){
        return p[p.empty() ? 0 : std::min(t, p.size()-1)];
    };

    // Vertex conflicts
    for (size_t t = 0; t < makespan; ++t) {
        for (int i = 0; i < n; ++i) {
            size_t vi = at(A[i], t);
            for (int j = i+1; j < n; ++j) {
                size_t vj = at(A[j], t);
                if (vi == vj) return false;
            }
        }
    }
    // Edge conflicts (swap)
    if (makespan >= 2) {
        for (size_t t = 0; t+1 < makespan; ++t) {
            for (int i = 0; i < n; ++i) {
                size_t ui = at(A[i], t), vi = at(A[i], t+1);
                for (int j = i+1; j < n; ++j) {
                    size_t uj = at(A[j], t), vj = at(A[j], t+1);
                    if (ui == vj && vi == uj) return false;
                }
            }
        }
    }
    return true;
}

// Draw a single-line progress bar: elapsed / total (seconds)
inline void render_progress_bar(double elapsed_sec, double total_sec, int width = 40, bool end_flag = false){
    if (total_sec <= 0.0) return;
    if (elapsed_sec < 0.0) elapsed_sec = 0.0;
    if (elapsed_sec > total_sec) elapsed_sec = total_sec;

    double ratio = elapsed_sec / total_sec;
    if (end_flag) {
        ratio = 1.0; // force full bar on completion
    }
    int filled = static_cast<int>(ratio * width);

    // Colors: bar (green), border (blue), reset
    const char* BLUE  = "\033[1;34m";
    const char* GREEN = "\033[1;32m";
    const char* RESET = "\033[0m";
    
    // Return to line start, clear it, then redraw in place
    std::cout << std::flush;
    
    // ETA
    double remaining = total_sec - elapsed_sec;
    int eta_int = static_cast<int>(remaining);

    std::cout << "\r" << BLUE << "[" << RESET;

    std::cout << GREEN << std::string(filled, '=') << RESET
              << std::string(width - filled, ' ');

    std::cout << BLUE << "]" << RESET
              << " " << std::fixed << std::setprecision(0)
              << (ratio * 100.0) << "%  "
              << "(" << static_cast<int>(total_sec-eta_int) << "s/"
              << static_cast<int>(total_sec) << "s"
              << ", ETA " << ((end_flag)?0:eta_int) << "s)   "   // trailing spaces keep line clean
              << std::flush;
}

double scalarizationSolver::weighted_scalar_cost(const CostVector& c) const {
    double sum = 0.0;
    for (size_t i = 0; i < c.size(); ++i) {
        const double w = (i < scalar_weights_.size()) ? scalar_weights_[i] : 1.0;
        sum += w * static_cast<double>(c[i]);
    }
    return sum;
}

// Number of Solution
/******************  REPLACEMENT –– MergeJointPaths  ********************/
void scalarizationSolver::MergeJointPaths(HighLevelNodePtr node,
                                 int  solution_num,
                                 double /*max_eps not used any more*/)
{
    std::chrono::high_resolution_clock::time_point _t1 = std::chrono::high_resolution_clock::now();
    (void)_t1; // avoid unused variable warning

    /* ---------- data structures that build joint paths ---------- */
    std::list<std::pair<CostVector,int>> apex_idx_combos;   // lex-order list
    std::vector<CostVector> real_costs_vector;              // summed real costs
    std::vector<std::vector<size_t>> ids_vector;            // path indices per agent

    node->all_jps.clear();
    /* seed with the zero vector */
    apex_idx_combos.push_back({CostVector(DIM,0),0});
    real_costs_vector.push_back(CostVector(DIM,0));
    ids_vector.push_back({});

    /* ------------- expand agent by agent ------------------------ */
    int new_id = 0;
    for (int ag = 0; ag < AGENT_NUM; ++ag)
    {
        std::list<std::pair<CostVector,int>>  old_apex = apex_idx_combos;
        std::vector<CostVector>               old_real = real_costs_vector;
        std::vector<std::vector<size_t>>      old_ids  = ids_vector;

        apex_idx_combos.clear();
        real_costs_vector.clear();
        ids_vector.clear();
        new_id = 0;

        /* combine with every single-agent path of agent ag */
        for (const auto& combo : old_apex)
        {
            for (const auto& pathPair : node->sop_waypoints[ag])   // (id , pathVec)
            {
                CostVector new_apex = combo.first;
                CostVector new_real = old_real[combo.second];
                auto       new_idx  = old_ids [combo.second];

                add_cost(new_apex , node->sop_apex [ag][pathPair.first]);
                add_cost(new_real , node->sop_cost [ag][pathPair.first]);
                new_idx.push_back(pathPair.first);

                apex_idx_combos.push_back({new_apex , new_id});
                real_costs_vector.push_back(new_real);
                ids_vector .push_back(new_idx);
                ++new_id;
            }
        }

        /* keep lowest weighted-scalar joint costs */
        const auto scalar_less = [&](const std::pair<CostVector, int>& a,
                                     const std::pair<CostVector, int>& b) {
            return weighted_scalar_cost(real_costs_vector[static_cast<size_t>(a.second)])
                < weighted_scalar_cost(real_costs_vector[static_cast<size_t>(b.second)]);
        };
        apex_idx_combos.sort(scalar_less);
        scalarizationFilter(apex_idx_combos, eps1, eps2, eps3, eps4, eps5,
                  eps6, eps7, eps8, eps9, eps10);

        /* OPTIONAL –– cap list length (maintains earliest elements) */
        while ((int)apex_idx_combos.size() > solution_num)
            apex_idx_combos.pop_back();
    }

    /* ---------- move into node->all_jps in weighted-scalar order --------- */
    const auto scalar_less = [&](const std::pair<CostVector, int>& a,
                                 const std::pair<CostVector, int>& b) {
        return weighted_scalar_cost(real_costs_vector[static_cast<size_t>(a.second)])
            < weighted_scalar_cost(real_costs_vector[static_cast<size_t>(b.second)]);
    };
    apex_idx_combos.sort(scalar_less);
    for (const auto& el : apex_idx_combos)
        node->all_jps.emplace_back(el.first, ids_vector[el.second]);

    // auto _t2 = std::chrono::high_resolution_clock::now();
    // std::cout << "MergeJointPaths (Lex) took "
            //   << std::chrono::duration_cast<std::chrono::microseconds>(_t2-_t1).count()/1e6
            //   << " sec - kept " << node->all_jps.size() << " joint paths\n";
}
/*************************************************************************/

/******************** Lexicographic filter  ********************/
void scalarizationSolver::scalarizationFilter(std::list<std::pair<CostVector,int>>& L,
                           double eps1, double eps2, double eps3, double eps4, double eps5,
                           double eps6, double eps7, double eps8, double eps9, double eps10)
{
    if (L.empty()) return;

    const double eps[10] = {eps1,eps2,eps3,eps4,eps5,eps6,eps7,eps8,eps9,eps10};
    const int d = static_cast<int>(L.front().first.size());
    if (d == 0) return;

    // For j = 0..d-1, keep items within (1+eps_j) of the min on component j
    for (int j = 0; j < d; ++j) {
        const double ej = (j < 10) ? eps[j] : 0.0;

        double m = std::numeric_limits<double>::infinity();
        for (const auto& p : L) m = std::min(m, static_cast<double>(p.first[j]));

        L.remove_if([&](const auto& p){
            return static_cast<double>(p.first[j]) > m * (1.0 + ej);
        });

        if (L.empty()) break;
    }
}



void scalarizationSolver::MergeUntil(std::list<std::pair<CostVector, int>>& apex_idx_combos, std::vector<CostVector>& real_costs_vector, int solution_num, double max_eps)
{
    int total_num = apex_idx_combos.size();

    using mutual_eps = std::tuple<double, size_t, size_t>; // eps,apex_idx,real_idx,if_valid
    // // print the real_cost_vector for debugging purposes in green clor
    // std::cout << "\033[1;32m";
    // for (const auto& cost : real_costs_vector) {
    //     std::cout << "[scalarizationSolver.cpp - l185]Real Cost: ";
    //     for (const auto& c : cost) {
    //         std::cout << c << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\033[0m";

    std::list<mutual_eps>               eps_list;
    std::vector<CostVector>             apex_vector;
    std::vector<bool>                   valid_vector(total_num, true);
    std::vector<int>                    id_vector;

    for(auto ele: apex_idx_combos){
        apex_vector.push_back(ele.first);
        id_vector.push_back(ele.second);
    }

    for(int i = 0; i < total_num; i++){
        for(int j = 0; j < total_num; j++){
            if(j == i){
                continue;
            }
            double eps = calculate_BF(apex_vector.at(i), real_costs_vector.at(id_vector.at(j)));
            eps_list.push_back(std::make_tuple(eps, i, j));
        }
    }

    int  erase_num = 0;
    while(total_num - erase_num > solution_num){
        if(difftime(time(NULL), start_time) > TIME_LIMIT){
            return;
        }
        mutual_eps element;
        double min_eps = INT_MAX;
        for(auto iter = eps_list.begin(); iter != eps_list.end(); iter++){
            if(std::get<0>(*iter) < min_eps){
                element = *iter;
                min_eps = std::get<0>(*iter);
            }
        }
        if(min_eps > max_eps){
            break;
        }
        erase_num ++;
        size_t invalid_id = std::get<1>(element);
        size_t valid_id = std::get<2>(element);
        apex_vector.at(valid_id) = vector_min(apex_vector.at(valid_id), apex_vector.at(invalid_id));
        valid_vector.at(invalid_id) = false;
        for(auto iter = eps_list.begin(); iter != eps_list.end(); ){
            if(std::get<1>(*iter) == invalid_id || std::get<2>(*iter) == invalid_id){
                iter = eps_list.erase(iter);
                continue;
            }
            if(std::get<1>(*iter) == valid_id){
                std::get<0>(*iter) = calculate_BF(apex_vector.at(valid_id), real_costs_vector.at(id_vector.at(std::get<2>(*iter))));
            }
            iter ++;
        }
    }

    apex_idx_combos.clear();
    int j = 0;
    (void)j;
    for(int i = 0; i < total_num; i++){
        if(!valid_vector.at(i)){
            continue;
        }
        apex_idx_combos.push_back(std::make_pair(apex_vector.at(i), id_vector.at(i)));
    }
}


void scalarizationSolver::MergeUntil(std::vector<CostVector>& apex_vectors, std::vector<CostVector>& real_costs_vector, int solution_num, double max_eps)
{
    int total_num = apex_vectors.size();

    using mutual_eps = std::tuple<double, size_t, size_t>; // eps,apex_idx,real_idx,if_valid
    std::list<mutual_eps>               eps_list;
    std::vector<CostVector>             apex_vector = apex_vectors;
    std::vector<CostVector>             cost_vector = real_costs_vector;
    std::vector<bool>                   valid_vector(total_num, true);

    for(int i = 0; i < total_num; i++){
        for(int j = 0; j < total_num; j++){
            if(j == i){
                continue;
            }
            double eps = calculate_BF(apex_vector.at(i), cost_vector.at(j));
            eps_list.push_back(std::make_tuple(eps, i, j));
        }
    }

    int  erase_num = 0;
    while(total_num - erase_num > solution_num){
        mutual_eps element;
        double min_eps = INT_MAX;
        for(auto iter = eps_list.begin(); iter != eps_list.end(); iter++){
            if(std::get<0>(*iter) < min_eps){
                element = *iter;
                min_eps = std::get<0>(*iter);
            }
        }
        if(min_eps > max_eps){
            break;
        }
        erase_num ++;
        size_t invalid_id = std::get<1>(element);
        size_t valid_id = std::get<2>(element);
        apex_vector.at(valid_id) = vector_min(apex_vector.at(valid_id), apex_vector.at(invalid_id));
        valid_vector.at(invalid_id) = false;
        for(auto iter = eps_list.begin(); iter != eps_list.end(); ){
            if(std::get<1>(*iter) == invalid_id || std::get<2>(*iter) == invalid_id){
                iter = eps_list.erase(iter);
                continue;
            }
            if(std::get<1>(*iter) == valid_id){
                std::get<0>(*iter) = calculate_BF(apex_vector.at(valid_id), cost_vector.at(std::get<2>(*iter)));
            }
            iter ++;
        }
    }

    apex_vectors.clear();
    real_costs_vector.clear();
    
    for(int i = 0; i < total_num; i++){
        if(!valid_vector.at(i)){
            continue;
        }
        apex_vectors.push_back(apex_vector.at(i));
        real_costs_vector.push_back(cost_vector.at(i));
    }
}



void scalarizationSolver::calculate_CAT(HighLevelNodePtr node, VertexCAT& vertex_cat, EdgeCAT& edge_cat, int agent_id) // CAT - detaist about the constraints- Conflict Agent TIme
{
    vertex_cat.clear(); edge_cat.clear();
    for(int i = 0; i < node->cur_ids.size(); i++){
        if(i == agent_id){
            continue;
        }
        auto path_nodes = node->sop_waypoints.at(i)[node->cur_ids.at(i)];
        for(int t = 0; t < path_nodes.size(); t++){
            size_t node_id = path_nodes.at(t);
            if(!vertex_cat.count(t)){
                vertex_cat.insert(std::make_pair(t, std::vector<int>(GRAPH_SIZE, 0)));
            }
            vertex_cat.at(t).at(node_id) ++;
        }
        for(int t = 0; t < path_nodes.size() - 1; t++){
            size_t source_id = path_nodes.at(t);
            size_t target_id = path_nodes.at(t+1);
            if(!edge_cat.count(t)){
                edge_cat.insert(std::make_pair(t, std::vector<std::unordered_map<size_t, int>>(GRAPH_SIZE)));
            }
            if(!edge_cat.at(t).at(source_id).count(target_id)){
                edge_cat.at(t).at(source_id).insert(std::make_pair(target_id, 0));
            }
            edge_cat.at(t).at(source_id).at(target_id) ++;
        }
    }
}


OutputTuple scalarizationSolver::run(std::vector<Edge>& edges, std::vector<std::pair<size_t, size_t>>& start_goal, HSolutionID& hsolution_ids, std::vector<CostVector>& hsolution_costs, LoggerPtr& logger)
{
    const bool STOP_ON_FIRST = true;  
    std::chrono::high_resolution_clock::time_point   precise_start_time, precise_end_time;
    precise_start_time = std::chrono::high_resolution_clock::now();
    using clock_t = std::chrono::steady_clock;
    const auto wall_start = clock_t::now();
    auto last_progress_update = wall_start;
    const std::chrono::milliseconds progress_every(100); // throttle to ~10 Hz

    std::chrono::high_resolution_clock::time_point t1, t2;
    double duration;

    std::vector<CostVector> hsolution_apex_costs;

    bool is_success = true;
    
    double HLMergingTime = 0, LowLevelTime = 0, TotalTime;
    int ConflictSolvingNum = 0, SolutionNum = 0;
    (void)SolutionNum;

    ConflictResolver conflict_resolver;

    start_time = time(NULL);
    HLQueue open_list;

    VertexCAT  vertex_cat; // <time, <vertex, num>>
    EdgeCAT    edge_cat; // <time, <source, <target, num>>>

    // calculate heuristic
    std::vector<Heuristic> heuristics(AGENT_NUM);
    AdjacencyMatrix graph(GRAPH_SIZE, edges);   // can run outside and only once
    AdjacencyMatrix inv_graph(GRAPH_SIZE, edges, true);
    for(int i = 0; i < AGENT_NUM; i++){
        ShortestPathHeuristic sp_heuristic(start_goal.at(i).second, GRAPH_SIZE, inv_graph, TURN_DIM, TURN_COST);
        heuristics.at(i) = std::bind( &ShortestPathHeuristic::operator(), sp_heuristic, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    }

    //  initialize open_list
    HighLevelNodePtr root_node = std::make_shared<HighLevelNode>(AGENT_NUM);

    // Keep the best verified solution seen so far
    bool have_best = false;
    CostVector best_cost;                   // real cost vector of the best solution
    std::vector<std::vector<size_t>> best_paths_ids; // per-agent solver ID paths

    auto lex_less = [this](const CostVector& a, const CostVector& b){
        return macussp::lex_less(a, b, objective_ordering());
    };

    for(size_t i = 0; i < AGENT_NUM; i ++){
        t1 = std::chrono::high_resolution_clock::now();

        if(MS == MergingStrategy::CONFLICT_BASED){
            single_run_map(GRAPH_SIZE, graph, heuristics.at(i), start_goal.at(i).first, start_goal.at(i).second, 
                LSOLVER, EPS, DEFAULT_MS, logger, TIME_LIMIT, root_node->sop_waypoints.at(i), root_node->sop_apex.at(i), root_node->sop_cost.at(i), 
                root_node->vertex_constraints.at(i), root_node->edge_constraints.at(i), vertex_cat, edge_cat, root_node->conflict_num, TURN_DIM, TURN_COST);
        }else{
            single_run_map(GRAPH_SIZE, graph, heuristics.at(i), start_goal.at(i).first, start_goal.at(i).second, 
                LSOLVER, EPS, MS, logger, TIME_LIMIT, root_node->sop_waypoints.at(i), root_node->sop_apex.at(i), root_node->sop_cost.at(i), 
                root_node->vertex_constraints.at(i), root_node->edge_constraints.at(i), vertex_cat, edge_cat, root_node->conflict_num, TURN_DIM, TURN_COST);
        }

        std::list<std::pair<CostVector, int>>   apex_idx_combos;
        std::vector<CostVector>     real_costs_vector;

        for(auto ele: root_node->sop_waypoints.at(i)){
            apex_idx_combos.push_back(std::make_pair(root_node->sop_apex.at(i)[ele.first], ele.first));
            real_costs_vector.push_back(root_node->sop_cost.at(i)[ele.first]);
        }

        MergeUntil(apex_idx_combos, real_costs_vector, SOLUTION_NUM);

        auto apex_map = root_node->sop_apex.at(i);
        auto cost_map = root_node->sop_cost.at(i);
        auto path_map = root_node->sop_waypoints.at(i);
        root_node->sop_apex.at(i).clear(); 
        root_node->sop_cost.at(i).clear();
        root_node->sop_waypoints.at(i).clear();
        
        int idx = 0;
        for (auto ele: apex_idx_combos){
            root_node->sop_apex.at(i).insert(std::make_pair(idx, apex_map[ele.second]));
            root_node->sop_cost.at(i).insert(std::make_pair(idx, cost_map[ele.second]));
            root_node->sop_waypoints.at(i).insert(std::make_pair(idx, path_map[ele.second]));
            idx ++;
        }
        
        t2 = std::chrono::high_resolution_clock::now();
        duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
        LowLevelTime += duration;
        // std::cout << "Agent ID: " << i << " Low Level Time = " << duration << "  size = " << root_node->sop_waypoints.at(i).size() << std::endl;
    }

    t1 = std::chrono::high_resolution_clock::now();

    MergeJointPaths(root_node, SOLUTION_NUM);

    if (root_node->all_jps.empty()) {
        is_success = false;
        // Nothing to explore.
        AgentPaths empty_paths;
        return std::make_tuple(0.0, 0.0, 0.0, 0, 0, is_success, empty_paths, CostVector{});
    }

    t2 = std::chrono::high_resolution_clock::now();
    duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
    HLMergingTime += duration;

    if(difftime(time(NULL), start_time) > TIME_LIMIT){
        output << "FAIL" << std::endl 
            << "apex cost:" << std::endl 
            << std::endl 
            << "real cost:" << std::endl 
            << std::endl << std::endl << std::endl;
        AgentPaths empty_paths;
        return std::make_tuple(HLMergingTime, LowLevelTime, difftime(time(NULL), start_time), 0, 0, is_success, empty_paths, CostVector{});
    }

    root_node->cur_ids = root_node->all_jps.front().second;
    root_node->cur_apex = root_node->all_jps.front().first;
    open_list.insert(root_node);

    std::tuple<int, int, CostVector, size_t> cft;
    // Main loop doing the high-level search
    while(!open_list.empty())
    {
        // progress update (before heavy work / after a pop)
        const auto now = clock_t::now();
        if (now - last_progress_update >= progress_every) {
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            render_progress_bar(elapsed, static_cast<double>(TIME_LIMIT));
            // std::cout << std::endl;
            last_progress_update = now;
        }
        if(difftime(time(NULL), start_time) > TIME_LIMIT){
            is_success = false;
            break;
        }
        auto node = open_list.pop();

        auto current_path = node->all_jps.front();

        
        if(node->all_jps.empty()){
            continue;
        }

        node->cur_ids = node->all_jps.front().second;
        node->cur_apex = node->all_jps.front().first;

        if(node->all_jps.front() != current_path){
            open_list.insert(node);
            continue;
        }
        
        // Detect first conflict
        cft = conflict_resolver.DetectConflict(node->all_jps.front(), node->sop_waypoints);

        if(std::get<2>(cft).empty()){
            if(EAGER){
                output << "error : cft not empty with eager solution update";
                exit(1);
            }else{
                // Build actual per-agent ID paths for verification
                std::vector<std::vector<size_t>> cand_paths(AGENT_NUM);
                for (int i = 0; i < AGENT_NUM; ++i) {
                    const auto pid = node->cur_ids.at(i);
                    cand_paths[i]  = node->sop_waypoints.at(i).at(pid); // sequence of nodeIDs
                }
                // Verify joint plan truly conflict-free
                if (!verify_joint_paths_conflict_free(cand_paths)) {
                    // Do NOT accept; move to next joint candidate
                    node->all_jps.pop_front();
                    if(node->all_jps.empty()) { continue; }
                    node->cur_ids  = node->all_jps.front().second;
                    node->cur_apex = node->all_jps.front().first;
                    open_list.insert(node);
                    continue;
                }

                // Compute real cost of this verified solution
                CostVector  real_cost(node->cur_apex.size(), 0);
                for(int i = 0; i < AGENT_NUM; i++){
                    add_cost(real_cost, node->sop_cost.at(i)[node->cur_ids.at(i)]);
                }

                // Maintain non-dominated pool as before
                int flag = 0;
                for(auto iter1 = hsolution_apex_costs.begin(); iter1 != hsolution_apex_costs.end(); ){
                    if(is_dominated(*iter1, real_cost, EPS)){
                        node->cur_apex = vector_min(node->cur_apex, *iter1);
                        iter1 = hsolution_apex_costs.erase(iter1);
                        hsolution_costs.erase(hsolution_costs.begin()+flag);
                        hsolution_ids.erase(hsolution_ids.begin()+flag);
                    }else{
                        flag ++;
                        iter1 ++;
                    }
                }

                // Record into hsolution_* (ids here are the indices; we also keep the actual ID sequences in best_paths_ids)
                std::vector<std::vector<size_t>> new_hsolution;
                new_hsolution.push_back(node->cur_ids);
                hsolution_ids.push_back(new_hsolution);
                hsolution_costs.push_back(real_cost);
                hsolution_apex_costs.push_back(node->cur_apex);

                // Update the best verified solution (minimize weighted scalar cost)
                if (!have_best || weighted_scalar_cost(real_cost) < weighted_scalar_cost(best_cost)) {
                    have_best = true;
                    best_cost = real_cost;
                    best_paths_ids = cand_paths; // store exact plan we verified
                }
                
                // EARLY EXIT: stop searching after the first verified solution
                if (STOP_ON_FIRST) {
                    break;  // breaks out of while (!open_list.empty())
                }
                // Move to next joint candidate
                node->all_jps.pop_front();
                if(node->all_jps.empty()){
                    continue;
                }
                node->cur_ids = node->all_jps.front().second;
                node->cur_apex = node->all_jps.front().first;

                open_list.insert(node);
                continue;
            }
        }
        
        // print constraint info
        ConflictSolvingNum ++;
        // if (ConflictSolvingNum % (500/AGENT_NUM) == 0) {
        //     std::cout << "\n[INFO] * Solver::Search, after " << ConflictSolvingNum << " conflict splits " 
        //     << "       time = " << difftime(time(NULL), start_time) << std::endl
        //     << map_name << std::endl;
        // }

        // Now setting constraint - Binary Branching CT
        if(std::get<2>(cft).size() == 1){   // vertex confliction
            for(int i = 0; i < 2; i ++){
                int agent_id = i == 0 ? std::get<0>(cft) : std::get<1>(cft);
                if(agent_id < 0){
                    continue;
                }
                auto new_node = std::make_shared<HighLevelNode>(*node);
                new_node->sop_waypoints.at(agent_id).clear();
                new_node->sop_apex.at(agent_id).clear();
                new_node->sop_cost.at(agent_id).clear();
                new_node->conflict_num.clear();
                new_node->all_jps.clear();


                conflict_resolver.AddConstraint(new_node->vertex_constraints, agent_id, std::get<2>(cft).front(), std::get<3>(cft));

                //  Low Level Search
                if(MS == MergingStrategy::CONFLICT_BASED){
                    calculate_CAT(node, vertex_cat, edge_cat, agent_id);
                }

                t1 = std::chrono::high_resolution_clock::now();

                // now running the mapf solver again with constraints set to avaid the conflict
                single_run_map(GRAPH_SIZE, graph, heuristics.at(agent_id), start_goal.at(agent_id).first, start_goal.at(agent_id).second, 
                    LSOLVER, EPS, MS, logger, TIME_LIMIT, new_node->sop_waypoints.at(agent_id),  
                    new_node->sop_apex.at(agent_id), new_node->sop_cost.at(agent_id), 
                    new_node->vertex_constraints[agent_id], new_node->edge_constraints[agent_id], vertex_cat, edge_cat, new_node->conflict_num, TURN_DIM, TURN_COST); 

                // DEBUG1: Enforce the vertex forbid (v at time t): drop any single-agent path that violates it.
                {
                    const int forbid_t    = static_cast<int>(std::get<3>(cft));
                    const size_t forbid_v = std::get<2>(cft).front(); // vertex id

                    auto& P   = new_node->sop_waypoints.at(agent_id);
                    auto& Cst = new_node->sop_cost.at(agent_id);
                    auto& Ax  = new_node->sop_apex.at(agent_id);

                    for (auto it = P.begin(); it != P.end(); ) {
                        const auto& path = it->second;
                        bool violates = (forbid_t >= 0 && forbid_t < static_cast<int>(path.size()) && path[forbid_t] == forbid_v);
                        if (violates) {
                            int pid = it->first;
                            it = P.erase(it);
                            Cst.erase(pid);
                            Ax.erase(pid);
                        } else {
                            ++it;
                        }
                    }
                    if (P.empty()) {
                        // No feasible path for this agent under the new constraint, skip inserting this branch.
                        continue;
                    }
                }
                // DEBUG2 (Patch4)
                {
                    const int forbid_t    = static_cast<int>(std::get<3>(cft));
                    const size_t forbid_v = std::get<2>(cft).front();
                    const auto& P = new_node->sop_waypoints.at(agent_id);
                    for (const auto& kv : P) {
                        const auto& path = kv.second;
                        if (forbid_t >= 0 && forbid_t < static_cast<int>(path.size())) {
                            if (path[forbid_t] == forbid_v) {
                                std::cerr << "[ASSERT] vertex constraint not enforced for agent "
                                        << agent_id << " at t=" << forbid_t << " v=" << forbid_v << "\n";
                            }
                        }
                    }
                }


                std::list<std::pair<CostVector, int>>   apex_idx_combos;
                std::vector<CostVector>     real_costs_vector;

                for(auto ele: new_node->sop_waypoints.at(agent_id)){
                    apex_idx_combos.push_back(std::make_pair(new_node->sop_apex.at(agent_id)[ele.first], ele.first));
                    real_costs_vector.push_back(new_node->sop_cost.at(agent_id)[ele.first]);
                }

                MergeUntil(apex_idx_combos, real_costs_vector, SOLUTION_NUM);

                auto apex_map = new_node->sop_apex.at(agent_id);
                auto cost_map = new_node->sop_cost.at(agent_id);
                auto path_map = new_node->sop_waypoints.at(agent_id);
                new_node->sop_apex.at(agent_id).clear(); 
                new_node->sop_cost.at(agent_id).clear();
                new_node->sop_waypoints.at(agent_id).clear();
                
                int idx = 0;
                for (auto ele: apex_idx_combos){
                    new_node->sop_apex.at(agent_id).insert(std::make_pair(idx, apex_map[ele.second]));
                    new_node->sop_cost.at(agent_id).insert(std::make_pair(idx, cost_map[ele.second]));
                    new_node->sop_waypoints.at(agent_id).insert(std::make_pair(idx, path_map[ele.second]));
                    idx ++;
                }


                t2 = std::chrono::high_resolution_clock::now();
                duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
                LowLevelTime += duration;
                // std::cout << "LowLevelTime = " << duration << std::endl;

                if(new_node->sop_waypoints.at(agent_id).empty()){
                    continue;
                }

                //  Merging joint paths - highlevel solver stuff
                t1 = std::chrono::high_resolution_clock::now();
                
                MergeJointPaths(new_node, SOLUTION_NUM);

                if (new_node->all_jps.empty()) {
                    // No feasible joint plan from this branch; discard.
                    continue;
                }

                if(difftime(time(NULL), start_time) > TIME_LIMIT){
                    is_success = false;
                    continue;
                }
                t2 = std::chrono::high_resolution_clock::now();
                duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
                HLMergingTime += duration;
                
                new_node->cur_ids = new_node->all_jps.front().second;
                new_node->cur_apex = new_node->all_jps.front().first;
                open_list.insert(new_node);
            }

        }
        else{  //  edge confliction
            for(int i = 0; i < 2; i++){
                int agent_id = i == 0 ? std::get<0>(cft) : std::get<1>(cft);
                auto new_node = std::make_shared<HighLevelNode>(*node);
                new_node->sop_waypoints.at(agent_id).clear();
                new_node->sop_apex.at(agent_id).clear();
                new_node->sop_cost.at(agent_id).clear();
                new_node->conflict_num.clear();
                new_node->all_jps.clear();


                conflict_resolver.AddConstraint(new_node->edge_constraints, agent_id, std::get<2>(cft).at(i), std::get<2>(cft).at(1-i), std::get<3>(cft));

                if(MS == MergingStrategy::CONFLICT_BASED){
                    calculate_CAT(node, vertex_cat, edge_cat, agent_id);
                }

                t1 = std::chrono::high_resolution_clock::now();
                single_run_map(GRAPH_SIZE, graph, heuristics.at(agent_id), start_goal.at(agent_id).first, start_goal.at(agent_id).second, 
                    LSOLVER, EPS, MS, logger, TIME_LIMIT, new_node->sop_waypoints.at(agent_id), 
                    new_node->sop_apex.at(agent_id), new_node->sop_cost.at(agent_id), 
                    new_node->vertex_constraints[agent_id], new_node->edge_constraints[agent_id], vertex_cat, edge_cat, new_node->conflict_num, TURN_DIM, TURN_COST);

                // DEBUG1: Enforce the edge forbid (u->v at time t): drop any single-agent path that violates it.
                {
                    const int    forbid_t = static_cast<int>(std::get<3>(cft));
                    const size_t u = std::get<2>(cft).at(i);
                    const size_t v = std::get<2>(cft).at(1 - i);

                    auto& P   = new_node->sop_waypoints.at(agent_id);
                    auto& Cst = new_node->sop_cost.at(agent_id);
                    auto& Ax  = new_node->sop_apex.at(agent_id);

                    for (auto it = P.begin(); it != P.end(); ) {
                        const auto& path = it->second;
                        bool violates = (forbid_t >= 0
                                        && forbid_t + 1 < static_cast<int>(path.size())
                                        && path[forbid_t] == u && path[forbid_t + 1] == v);
                        if (violates) {
                            int pid = it->first;
                            it = P.erase(it);
                            Cst.erase(pid);
                            Ax.erase(pid);
                        } else {
                            ++it;
                        }
                    }
                    if (P.empty()) {
                        // No feasible path for this agent under the new constraint, skip inserting this branch.
                        continue;
                    }
                }

                // DEBUG2 : Patch 4
                {
                    const int forbid_t = static_cast<int>(std::get<3>(cft));
                    const size_t u = std::get<2>(cft).at(i);
                    const size_t v = std::get<2>(cft).at(1 - i);
                    const auto& P = new_node->sop_waypoints.at(agent_id);
                    for (const auto& kv : P) {
                        const auto& path = kv.second;
                        if (forbid_t >= 0 && forbid_t + 1 < static_cast<int>(path.size())) {
                            if (path[forbid_t] == u && path[forbid_t + 1] == v) {
                                std::cerr << "[ASSERT] edge constraint not enforced for agent "
                                        << agent_id << " at t=" << forbid_t
                                        << " (" << u << "->" << v << ")\n";
                            }
                        }
                    }
                }



                std::list<std::pair<CostVector, int>>   apex_idx_combos;
                std::vector<CostVector>     real_costs_vector;

                for(auto ele: new_node->sop_waypoints.at(agent_id)){
                    apex_idx_combos.push_back(std::make_pair(new_node->sop_apex.at(agent_id)[ele.first], ele.first));
                    real_costs_vector.push_back(new_node->sop_cost.at(agent_id)[ele.first]);
                }

                MergeUntil(apex_idx_combos, real_costs_vector, SOLUTION_NUM);

                auto apex_map = new_node->sop_apex.at(agent_id);
                auto cost_map = new_node->sop_cost.at(agent_id);
                auto path_map = new_node->sop_waypoints.at(agent_id);
                new_node->sop_apex.at(agent_id).clear(); 
                new_node->sop_cost.at(agent_id).clear();
                new_node->sop_waypoints.at(agent_id).clear();
                
                int idx = 0;
                for (auto ele: apex_idx_combos){
                    new_node->sop_apex.at(agent_id).insert(std::make_pair(idx, apex_map[ele.second]));
                    new_node->sop_cost.at(agent_id).insert(std::make_pair(idx, cost_map[ele.second]));
                    new_node->sop_waypoints.at(agent_id).insert(std::make_pair(idx, path_map[ele.second]));
                    idx ++;
                }

                t2 = std::chrono::high_resolution_clock::now();
                duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
                // std::cout << "LowLevelTime = " << duration << std::endl;
                LowLevelTime += duration;

                if(new_node->sop_waypoints.at(agent_id).empty()){
                    continue;
                }

                //  Merging joint paths - highlevel solver stuff
                t1 = std::chrono::high_resolution_clock::now();
                
                MergeJointPaths(new_node, SOLUTION_NUM);

                if (new_node->all_jps.empty()) {
                    // No feasible joint plan from this branch; discard.
                    continue;
                }

                if(difftime(time(NULL), start_time) > TIME_LIMIT){
                    is_success = false;
                    continue;
                }
                t2 = std::chrono::high_resolution_clock::now();
                duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0;
                HLMergingTime += duration;
                
                new_node->cur_ids = new_node->all_jps.front().second;
                new_node->cur_apex = new_node->all_jps.front().first;
                open_list.insert(new_node);
            }
        } // done setting relevant constraints for egde or vertex conflict here
    }
    // std::cout << "LCBS Solver was" << (is_success ? "\033[1;32m successful 😄 \033[0m" : "\033[1;31m unsuccessful 🙁\033[0m") << std::endl;
    precise_end_time = std::chrono::high_resolution_clock::now();
    TotalTime = (double)(std::chrono::duration_cast<std::chrono::microseconds>(precise_end_time - precise_start_time).count())/1000000.0;


    
    std::vector<std::pair<CostVector, int>> apex_id_list(hsolution_costs.size());
    auto _hsolution_costs = hsolution_costs;
    for(int i = 0; i < hsolution_costs.size(); i++){
        apex_id_list.at(i) = std::make_pair(hsolution_apex_costs.at(i), i);
    }
    hsolution_costs.clear(); hsolution_costs.shrink_to_fit();
    hsolution_apex_costs.clear(); hsolution_apex_costs.shrink_to_fit();    
    std::sort(apex_id_list.begin(), apex_id_list.end(), [](const std::pair<CostVector, int>& a, const std::pair<CostVector, int>& b){
        for(int i = 0; i < a.first.size(); i++){
            if(a.first.at(i) != b.first.at(i)){
                return a.first.at(i) < b.first.at(i);
            }
        }
        return true;
    });
    for(auto ele: apex_id_list){
        hsolution_apex_costs.push_back(ele.first);
        hsolution_costs.push_back(_hsolution_costs.at(ele.second));
    }
    

    // Decide success strictly by existence of a verified solution
    is_success = have_best;

    output << (is_success ? "SUCCESS" : "FAIL") << std::endl;

    // (Optional) re-verify before writing/returning
    if (is_success && !verify_joint_paths_conflict_free(best_paths_ids)) {
        is_success = false;
        output << "FAIL" << std::endl;
    }

    // Write the verified plan (if any)
    std::ofstream plan_file("../results/MAPF_OUTPUT/final_paths.txt");
    plan_file.clear();

    AgentPaths agent_paths; // to return

    if (is_success) {
        for (int ag = 0; ag < AGENT_NUM; ++ag) {
            plan_file << "Agent " << ag + 1 << ": ";
            const auto& path_ids = best_paths_ids[ag];
            std::vector<std::tuple<int,int>> agent_path_xy;
            for (size_t i = 0; i < path_ids.size(); ++i) {
                const auto& coord = id2coord[path_ids[i]]; // {row, col}
                int x = coord[1], y = coord[0];
                plan_file << "(" << x << "," << y << ")";
                if (i + 1 < path_ids.size()) plan_file << " -> ";
                agent_path_xy.emplace_back(x, y);
            }
            plan_file << std::endl;
            agent_paths[ag] = std::move(agent_path_xy);
        }
    } else {
        // leave file empty or write a note
        plan_file << "No conflict-free solution found within limits.\n";
    }
    plan_file.close();

    const auto now2 = clock_t::now();
    double elapsed2 = std::chrono::duration<double>(now2 - wall_start).count();
    render_progress_bar(elapsed2, static_cast<double>(TIME_LIMIT), 40, true);
    std::cout << std::endl;

    printScalarizationResult(is_success);
    if (is_success) {
        // print best cost
        for (int i = 0; i < best_cost.size(); ++i) {
            output << "Best cost for agent " << (i + 1) << ": " << best_cost[i] << std::endl;
        }
    }

    return std::make_tuple(HLMergingTime, LowLevelTime, TotalTime, ConflictSolvingNum,
                        static_cast<int>(hsolution_costs.size()), is_success, agent_paths, best_cost);
}