#include "DomainCosts.h"

#include "environment.h"
#include "IOUtils.h"

#include <stdexcept>

namespace macussp {

std::vector<std::string> domain_objective_cost_paths(const std::string& map_name) {
    return {
        "../maps/" + map_name + "/cost-1.cost",
        "../maps/" + map_name + "/cost-2.cost",
        "../maps/" + map_name + "/cost-3.cost",
    };
}

DomainObjectiveGrids load_domain_objective_grids(const std::string& map_name) {
    const std::string map_path = "../maps/" + map_name + "/" + map_name + ".map";
    PreProcessor p;
    std::vector<std::string> rows;
    p.read_map_rows(map_path, rows);
    const int H = static_cast<int>(rows.size());
    const int W = static_cast<int>(rows.front().size());

    DomainObjectiveGrids out;
    out.width = W;
    out.height = H;
    auto paths = domain_objective_cost_paths(map_name);
    for (int i = 0; i < 3; ++i) {
        auto flat = p.load_cost_grids({paths[static_cast<size_t>(i)]}, H, W)[0];
        out.grids[static_cast<size_t>(i)].assign(static_cast<size_t>(H), std::vector<double>(static_cast<size_t>(W)));
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                out.grids[static_cast<size_t>(i)][static_cast<size_t>(r)][static_cast<size_t>(c)] =
                    flat[static_cast<size_t>(r * W + c)];
            }
        }
    }
    return out;
}

const DomainObjectiveGrids& ensure_domain_costs(Environment& env) {
    return env.ensureDomainObjectiveGrids();
}

}  // namespace macussp
