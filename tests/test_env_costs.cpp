#include "DomainCosts.h"
#include "environment.h"
#include "IOUtils.h"

#include <iostream>

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FAIL: " #cond "\n"; return 1; } } while (0)

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    Environment env("salp", 1);
    env.getLandmarkStates2ContextSubsetMap_manually();
    const auto& grids = env.ensureDomainObjectiveGrids();

    PreProcessor p;
    Map map;
    std::vector<Edge> legacy6;
    std::unordered_map<size_t, std::vector<int>> id2coord;
    p.read_map("../maps/salp/salp.map", map, id2coord);
    p.cost_init(map, legacy6, 6);
    for (int i = 0; i < 6; ++i) {
        p.read_cost("../maps/salp/cost-" + std::to_string(i + 1) + ".cost", map, legacy6, i);
    }

    std::vector<Edge> env3;
    p.cost_init(map, env3, 3);
    for (int y = 0; y < map.height; ++y) {
        for (int x = 0; x < map.width; ++x) {
            if (map.getVal(y, x) == -1) continue;
            for (auto& e : env3) {
                if (e.target == map.getID(y, x)) {
                    for (int d = 0; d < 3; ++d) {
                        e.cost[static_cast<size_t>(d)] = static_cast<size_t>(
                            grids.grids[static_cast<size_t>(d)][static_cast<size_t>(y)][static_cast<size_t>(x)]);
                    }
                }
            }
        }
    }

    CHECK(legacy6.size() == env3.size());

    size_t mismatch01 = 0;
    size_t nonzero_extra_dims = 0;
    for (size_t i = 0; i < legacy6.size(); ++i) {
        for (int d = 0; d < 3; ++d) {
            if (legacy6[i].cost[static_cast<size_t>(d)] != env3[i].cost[static_cast<size_t>(d)]) {
                ++mismatch01;
            }
        }
        for (int d = 3; d < 6; ++d) {
            if (legacy6[i].cost[static_cast<size_t>(d)] != 0) ++nonzero_extra_dims;
        }
    }

    std::cout << "test_env_costs: edges=" << legacy6.size()
              << " first3_mismatches=" << mismatch01
              << " legacy_dim456_nonzero=" << nonzero_extra_dims << "\n";
    CHECK(mismatch01 == 0);
    std::cout << "test_env_costs: OK\n";
    return 0;
}
