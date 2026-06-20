#pragma once

#include <array>
#include <string>
#include <vector>

class Environment;

namespace macussp {

// Three domain objective cost grids (row-major), loaded from maps/<domain>/cost-1..3.cost.
struct DomainObjectiveGrids {
    int width{0};
    int height{0};
    std::array<std::vector<std::vector<double>>, 3> grids{};
};

// Load the three objective grids for a domain map (authoritative Stage 2 costs).
DomainObjectiveGrids load_domain_objective_grids(const std::string& map_name);

// Ensure Environment caches domain grids; returns cached copy.
const DomainObjectiveGrids& ensure_domain_costs(Environment& env);

std::vector<std::string> domain_objective_cost_paths(const std::string& map_name);

}  // namespace macussp
