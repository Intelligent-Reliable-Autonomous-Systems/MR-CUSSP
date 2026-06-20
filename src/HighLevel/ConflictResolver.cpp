#include "HighLevel/ConflictResolver.h"

std::tuple<int, int, std::vector<size_t>, size_t> ConflictResolver::DetectConflict(JointPathPair &joint_path, std::vector<PathSet>& indiv_paths_list)
{
    // std::cout << "\n\n\n\n------------------------------" << std::endl;
    size_t agent_num = joint_path.second.size();  
    // // print joint path
    // for(size_t agent_id = 0; agent_id < agent_num; agent_id ++){
    //     size_t path_id = joint_path.second.at(agent_id);
    //     std::cout << "Agent " << agent_id << " coordinates: " << std::endl;
    //     // in terms of coordinates sequence... (x_1, y_1) -> (x_2, y_2) -> ...
    //     for(size_t node_id : indiv_paths_list.at(agent_id).at(path_id)){
    //         std::cout << "(" << node_id % 32 << "," << node_id / 32 << ") ";
    //     }
    // std::cout << "------------------------------\n\n\n\n" << std::endl;
    // }


    //  vertex collision check
    for(size_t i = 0; i < agent_num; i ++){
        for(size_t j = i+1; j < agent_num; j ++){
            size_t id_i = joint_path.second.at(i), id_j = joint_path.second.at(j);
            std::vector<size_t> node_ids_i = indiv_paths_list.at(i)[id_i];
            std::vector<size_t> node_ids_j = indiv_paths_list.at(j)[id_j];
            for(size_t k = 0; k < node_ids_i.size() && k < node_ids_j.size(); k ++){
                if(node_ids_i.at(k) == node_ids_j.at(k)){

                    // // PRINT before returning
                    // std::cout << "[CONFLICT1] vertex  t=" << k
                    //           << "  agents=(" << i << "," << j << ")"
                    //           << "  node=" << node_ids_i.at(k) << std::endl;
                    return std::make_tuple(i, j, std::vector<size_t>({node_ids_i.at(k)}), k);
                }
            }

        //  if one agent has reached, another conflict with its target
            if (node_ids_i.size() < node_ids_j.size()) {
                for(size_t k = node_ids_i.size(); k < node_ids_j.size(); k++){
                    if(node_ids_i.back() == node_ids_j.at(k)){

                    // // PRINT before returning
                    // std::cout << "[CONFLICT2] vertex  t=" << k
                    //           << "  agents=(" << i << "," << j << ")"
                    //           << "  node=" << node_ids_i.at(k) << std::endl;
                        return std::make_tuple(-i-1, j, std::vector<size_t>({node_ids_i.back()}), k);
                    }
                }
            }else{
                for(size_t k = node_ids_j.size(); k < node_ids_i.size(); k++){
                    if(node_ids_i.at(k) == node_ids_j.back()){

                    // // PRINT before returning
                    // std::cout << "[CONFLICT3] vertex  t=" << k
                    //           << "  agents=(" << i << "," << j << ")"
                    //           << "  node=" << node_ids_i.at(k) << std::endl;
                        return std::make_tuple(i, -j-1, std::vector<size_t>({node_ids_j.back()}), k);
                    }
                }
            }
        }
    }
    
    //  edge check
    for(int i = 0; i < agent_num; i ++){
        for(int j = i+1; j < agent_num; j++){
            size_t id_i = joint_path.second.at(i), id_j = joint_path.second.at(j);
            std::vector<size_t> node_ids_i = indiv_paths_list.at(i)[id_i];
            std::vector<size_t> node_ids_j = indiv_paths_list.at(j)[id_j];

            for(size_t k = 0; k < node_ids_i.size()-1 && k < node_ids_j.size()-1; k ++){
                if(node_ids_i.at(k) == node_ids_j.at(k+1) && node_ids_i.at(k+1) == node_ids_j.at(k)){
                    return std::make_tuple(i, j, std::vector<size_t>({node_ids_i.at(k), node_ids_j.at(k)}), k);
                }
            }
        }
    }
                
    return std::make_tuple(-1, -1, std::vector<size_t>(), -1);
}


void ConflictResolver::AddConstraint(std::vector<VertexConstraint>& vertex_constraints, size_t agent_id, size_t node_id, size_t time)
{
    if(!vertex_constraints.at(agent_id).count(time)){
        vertex_constraints.at(agent_id).insert(std::make_pair(time, std::vector<size_t>()));
    }
    vertex_constraints.at(agent_id)[time].push_back(node_id);
    // // print the constraint
    // std::cout << "[VERTEX CONSTRAINT] agent_id=" << agent_id 
    //           << "  node_id=" << node_id 
    //           << "  time=" << time << std::endl;
    // std::cout << "Current vertex constraints: " << std::endl;
    // for (const auto& [time, nodes] : vertex_constraints.at(agent_id)) {
    //     std::cout << "  Time " << time << ": ";
    //     for (const auto& node : nodes) {
    //         std::cout << node << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

}


void ConflictResolver::AddConstraint(std::vector<EdgeConstraint>& edge_constraints, size_t agent_id, size_t source, size_t target, size_t time)
{
    if(!edge_constraints.at(agent_id).count(time)){
        edge_constraints.at(agent_id).insert(std::make_pair(time, std::unordered_map<size_t, std::vector<size_t>>()));
    }
    if(!edge_constraints.at(agent_id)[time].count(source)){
        edge_constraints.at(agent_id)[time].insert(std::make_pair(source, std::vector<size_t>()));
    }

    edge_constraints.at(agent_id)[time][source].push_back(target);
    // // print the constraint
    // std::cout << "[EDGE CONSTRAINT] agent_id=" << agent_id 
    //           << "  source=" << source 
    //           << "  target=" << target 
    //           << "  time=" << time << std::endl;
    // std::cout << "Current edge constraints: " << std::endl;
    // for (const auto& [time, edges] : edge_constraints.at(agent_id)) {
    //     std::cout << "  Time " << time << ": ";
    //     for (const auto& [source, targets] : edges) {
    //         std::cout << "Source " << source << ": ";
    //         for (const auto& target : targets) {
    //             std::cout << target << " ";
    //         }
    //         std::cout << "; ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

}