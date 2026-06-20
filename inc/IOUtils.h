#ifndef UTILS_IO_UTILS_H
#define UTILS_IO_UTILS_H

#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include "Definitions.h"
#include "MAP.h"

bool load_gr_files(std::string gr_file1, std::string gr_file2, std::vector<Edge> &edges, size_t &graph_size);
bool load_gr_files(std::vector<std::string> gr_files, std::vector<Edge> &edges, size_t &graph_size);
bool load_queries(std::string query_file, std::vector<std::pair<size_t, size_t>> &queries_out);

class PreProcessor
{
public:
    void read_map(std::string map_file_name, Map& map, std::unordered_map<size_t, std::vector<int>>& id2coord);
    void read_scenario(std::string config_file_name, Map& map, int agent_num, std::vector<std::pair<size_t, size_t>>&  start_end, int refine_R = 1);
    void read_cost(std::string cost_file_name, Map& map, std::vector<Edge>& edges, int dim);
    void cost_init(Map& map, std::vector<Edge>& edges, int dim);
    
    // NEW: raw ASCII rows after the "map" header (height/width lines are parsed here)
    void read_map_rows(const std::string& map_file, std::vector<std::string>& rows);

    // NEW: load D cost grids (dim × H×W), using your existing cost-*.cost files
    std::vector<std::vector<double>>
    load_cost_grids(const std::vector<std::string>& cost_files, int H, int W);

    // In-memory start/goal pairs (same coordinate rules as read_scenario MovingAI format).
    void build_start_goal_from_coords(Map& map,
                                      const std::vector<std::pair<int,int>>& start_xy,
                                      const std::vector<std::pair<int,int>>& goal_xy,
                                      std::vector<std::pair<size_t, size_t>>& start_end,
                                      int refine_R = 1);

};

#endif //UTILS_IO_UTILS_H
