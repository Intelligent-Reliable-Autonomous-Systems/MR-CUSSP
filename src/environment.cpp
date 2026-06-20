#include "environment.h"

#include "DomainCosts.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>


// ----- Internal helpers -----
namespace {
inline long long keyXY(int x, int y) {
    return (static_cast<long long>(y) << 32) | static_cast<unsigned>(x);
}

inline bool isObstacleChar(char c) { return c == '@' || c == '#' || c == '$'; }
inline bool isLandmarkChar(char c) { return c == 'L'; }
inline bool isFreeChar(char c)     { return c == '.' || c == 'S' || c == 'G' || c == 'L' || c == 'F' || c == 'C'; } // Note: Landmark is also free

// push v into adj[u] if not already present
inline void push_unique(std::vector<int>& nbrs, int v) {
    if (std::find(nbrs.begin(), nbrs.end(), v) == nbrs.end()) nbrs.push_back(v);
}

inline long long packEdge(int u, int v) {
    return ( (static_cast<long long>(u) << 32) ^ static_cast<unsigned>(v) );
}

} // namespace

bool Environment::blocksAgentMoveAt(int cur_vid, int next_vid, int t_abs) const {
    if (domain_ != DomainType::ForestFire) return false;
    if (cur_vid == next_vid) {
        // waiting in place: cell must not be burning at arrival time
        return isOnFireVertex(t_abs, cur_vid);
    }
    // stepping: next cell cannot be burning, and the edge cannot be traversed by fire at this tick
    return isOnFireVertex(t_abs, next_vid) || isOnFireEdge(t_abs, cur_vid, next_vid);
}

// ----- Mutator retained for compatibility -----
void Environment::setCell(int x, int y, char type) {
    if (!isWithinBounds(x, y)) return;
    int id = nodeIdAt(x,y);
    if (id < 0) return;

    // Update node type
    char old = V_[id].type;
    V_[id].type = type;

    // Maintain obstacle list and adjacency accordingly
    if (isObstacleChar(type)) {
        // Remove all edges incident to id
        for (int nb : adj_[id]) {
            auto& lst = adj_[nb];
            lst.erase(std::remove(lst.begin(), lst.end(), id), lst.end());
        }
        adj_[id].clear();

        // ensure in obstacles_
        if (std::find(obstacles_.begin(), obstacles_.end(), id) == obstacles_.end())
            obstacles_.push_back(id);
    } else {
        // If previously obstacle, reinsert edges to free neighbors
        if (isObstacleChar(old)) {
            // neighbors in 4-neighborhood
            const int dx[4] = {1,-1,0,0};
            const int dy[4] = {0,0,1,-1};
            for (int k = 0; k < 4; ++k) {
                int nx = x + dx[k], ny = y + dy[k];
                if (!isWithinBounds(nx, ny)) continue;
                int nid = nodeIdAt(nx, ny);
                if (nid >= 0 && !isObstacleChar(V_[nid].type)) {
                    push_unique(adj_[id], nid);
                    push_unique(adj_[nid], id);
                }
            }
            // remove from obstacles_
            obstacles_.erase(std::remove(obstacles_.begin(), obstacles_.end(), id), obstacles_.end());
        }
    }
}

// ----- Build G = (V, E) from .map -----
void Environment::initializeGraphFromMap() {
    // Parse map file
    std::string mapPath = "../maps/" + mapName + "/" + mapName + ".map";
    std::ifstream in(mapPath);
    if (!in) {
        throw std::runtime_error("Could not open map file: " + mapPath);
    }

    // MovingAI-style .map: skip header until "map"
    std::string line;
    bool inGrid = false;
    std::vector<std::string> rows;
    while (std::getline(in, line)) {
        if (!inGrid) {
            // normalize to lowercase for check
            std::string lcl = line;
            std::transform(lcl.begin(), lcl.end(), lcl.begin(), ::tolower);
            if (lcl.find("map") != std::string::npos) {
                inGrid = true;
            }
            continue;
        } else {
            if (!line.empty()) rows.push_back(line);
        }
    }
    in.close();

    if (rows.empty()) {
        throw std::runtime_error("Map contains no grid rows: " + mapPath);
    }

    height = static_cast<int>(rows.size());
    width  = static_cast<int>(rows.front().size());
    for (const auto& r : rows) {
        if ((int)r.size() != width) {
            throw std::runtime_error("Non-rectangular map row detected.");
        }
    }

    // Build vertices
    V_.clear(); adj_.clear(); xy2id_.clear(); obstacles_.clear();
    V_.reserve(width * height);
    adj_.resize(width * height); // provisional; some may remain isolated if obstacle

    auto add_node = [&](int x, int y, char c, int id) {
        char type = c;
        V_.push_back(Node{id, x, y, type});
        xy2id_[keyXY(x,y)] = id;
        if (type == '@') obstacles_.push_back(id);
    };

    int id = 0;
    for (int y = 0; y < height; ++y) {
        const std::string& r = rows[y];
        for (int x = 0; x < width; ++x, ++id) {
            add_node(x, y, r[x], id);
        }
    }

    // Build undirected edges among non-obstacle nodes (4-connected by default)
    const int dx[4] = {1,-1,0,0};
    const int dy[4] = {0,0,1,-1};

    auto add_undirected = [&](int u, int v) {
        if (u == v) return;
        push_unique(adj_[u], v);
        push_unique(adj_[v], u);
    };

    for (const auto& n : V_) {
        if (n.type == '@') continue; // no edges from obstacles
        for (int k = 0; k < 4; ++k) {
            int nx = n.x + dx[k], ny = n.y + dy[k];
            if (!isWithinBounds(nx, ny)) continue;
            int vid = nodeIdAt(nx, ny);
            if (vid < 0) continue;
            if (V_[vid].type == '@') continue; // do not connect to obstacles
            add_undirected(n.id, vid);
        }
    }
}


std::pair<int,int> Environment::firePosAt(int t) const {
    if (!fire_.has_t(t)) return {-1, -1};
    return { fire_.xs[t], fire_.ys[t] };
}

int Environment::fireVidAt(int t) const {
    if (!fire_.has_t(t)) return -1;
    int x = fire_.xs[t], y = fire_.ys[t];
    if (!isWithinBounds(x,y)) return -1;
    return nodeIdAt(x,y);
}

// void Environment::reserveFireCAT(std::vector<std::unordered_set<int>>& vertex_cat,
//                                  int t0, int t1) const {
//     if (!(domain_ == DomainType::ForestFire && fire_.active)) return; // NO-OP
//     for (int t = std::max(0, t0); t <= t1; ++t) {
//         int vid = fireVidAt(t);
//         if (vid >= 0) {
//             if ((int)vertex_cat.size() <= t) vertex_cat.resize(t+1);
//             vertex_cat[t].insert(vid);
//         }
//     }
// }

void Environment::loadFuelCostIfAvailable() {
    fuelGrid_.clear();
    std::string costPath = "../maps/" + mapName + "/firefuel.cost";
    std::ifstream in(costPath);
    if (!in) return; // optional

    fuelGrid_.resize(width * height, 1.0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double v;
            if (!(in >> v)) { fuelGrid_.clear(); return; } // fallback to uniform
            fuelGrid_[y * width + x] = v;
        }
    }
}
double Environment::fuelAt(int x, int y) const {
    if (fuelGrid_.empty()) return 1.0;
    if (!isWithinBounds(x,y)) return 1.0;
    return fuelGrid_[y * width + x];
}

std::vector<std::pair<int,int>> Environment::findFireSeeds(int k) const {
    std::vector<std::pair<int,int>> fire_seeds{};
    // find all "F" cells in the map and return their coordinates
    for (auto v: V_){
        if (v.type == 'F') {
            fire_seeds.emplace_back(v.x, v.y);
        }
    }
    // if (fire_seeds.size() != k) {
    //     throw std::runtime_error("[e.cpp l205]Number of fire seeds found does not match expected count k. Put appropriate amount of 'F's on the map file!");
    // }
    return fire_seeds;
}   

void Environment::generateKSquareFires(int Hmax, int k, int L, float fire_speed_factor) {
    H_MAX = Hmax;
    fires_.clear();
    fires_.resize(k);

    // 1) Seed locations (existing helper)
    auto fires_starts = findFireSeeds(k);

    // 2) Speed control identical to your current logic
    const int move_period = std::max(1, (int)(1.0f / std::min(1.0f, fire_speed_factor)));

    // For reproducibility you may want to seed from params; random_device is fine if nondet OK.
    std::mt19937 rng(std::random_device{}());

    // Helper to pack (x,y) for visited set
    auto pack_xy = [](int x, int y) -> uint64_t {
        return (uint64_t(uint32_t(x)) << 32) | uint32_t(y);
    };

    // Number of distinct positions needed before repetition
    const int raw_limit = (Hmax + move_period - 1) / move_period;

    for (int f = 0; f < k; ++f) {
        const auto [sx, sy] = fires_starts[f];

        // 3) Self-avoiding random walk confined to [sx, sx+L-1] x [sy, sy+L-1]
        const int xmin = sx - L + 1, xmax = sx + L - 1;
        const int ymin = sy - L + 1, ymax = sy + L - 1;

        std::vector<std::pair<int,int>> raw;       // distinct steps (before speed repetition)
        raw.reserve(raw_limit);
        std::unordered_set<uint64_t> visited;      // track cells already used by this fire

        int x = sx, y = sy;
        raw.emplace_back(x, y);
        visited.insert(pack_xy(x, y));

        while ((int)raw.size() < raw_limit) {
            // Gather unvisited 4-neighbors within the LxL window
            int candx[4] = { x + 1, x - 1, x,     x     };
            int candy[4] = { y,     y,     y + 1, y - 1 };
            int valid_idx[4]; int m = 0;

            for (int d = 0; d < 4; ++d) {
                int nx = candx[d], ny = candy[d];
                if (nx < xmin || nx > xmax || ny < ymin || ny > ymax) continue;
                // If obstacles matter, add your check here (e.g., if(!isFreeCell(nx,ny)) continue;).
                if (visited.count(pack_xy(nx, ny)) == 0) valid_idx[m++] = d;
            }

            if (m == 0) {
                // Trapped: no unvisited neighbor in the window → stop walking
                break;
            }

            std::uniform_int_distribution<int> dist(0, m - 1);
            int d = valid_idx[dist(rng)];
            x = candx[d]; y = candy[d];

            raw.emplace_back(x, y);
            visited.insert(pack_xy(x, y));
        }

        // 4) Expand by move_period and pad to Hmax (hold last cell after burnout)
        FireTraj T; 
        T.id = f;
        T.xs.reserve(Hmax); T.ys.reserve(Hmax);

        if (!raw.empty()) {
            for (const auto& [rx, ry] : raw) {
                for (int r = 0; r < move_period && (int)T.xs.size() < Hmax; ++r) {
                    T.xs.push_back(rx);
                    T.ys.push_back(ry);
                }
            }
            while ((int)T.xs.size() < Hmax) { // hold last position if needed
                T.xs.push_back(T.xs.back());
                T.ys.push_back(T.ys.back());
            }
        }

        T.active = !T.xs.empty();
        fires_[f] = std::move(T);
    }

    // Preserve legacy single-fire mirror if needed
    if (k >= 1) {
        fire_.xs = fires_[0].xs;
        fire_.ys = fires_[0].ys;
        fire_.active = fires_[0].active;
    }

    makeMapGrid();
    clearFireCache();
}


// void Environment::generateFireTrajectory(int Hmax, float fire_speed_factor)
// { 
//     generateFireTrajectory_stencil_loop(Hmax, /*L=*/5, fire_speed_factor, /*wrap=*/false);
// }

std::pair<int,int> Environment::firePosAt(int f, int t) const {
    if (!fires_.empty()) {
        if (f < 0 || f >= (int)fires_.size()) return {0,0};
        const auto &F = fires_[f];
        if (!F.active || t < 0 || t >= (int)F.xs.size()) return {0,0};
        return {F.xs[t], F.ys[t]};
    } else {
        // legacy: single fire_
        return firePosAt(t); // existing method
    }
}

int Environment::fireVidAt(int f, int t) const {
    if (!fires_.empty()) {
        if (f < 0 || f >= (int)fires_.size()) return -1;
        const auto &F = fires_[f];
        if (!F.has_t(t)) return -1;
        int x = F.xs[t], y = F.ys[t];
        return isWithinBounds(x,y) ? nodeIdAt(x,y) : -1;
    } else {
        return fireVidAt(t); // existing method
    }
}

std::vector<int> Environment::fireVidsAt(int t) const {
    std::vector<int> vids;
    if (!fires_.empty()) {
        for (int f = 0; f < (int)fires_.size(); ++f) {
            int v = fireVidAt(f, t);
            if (v >= 0) vids.push_back(v);
        }
    } else {
        int v = fireVidAt(t); // legacy
        if (v >= 0) vids.push_back(v);
    }
    return vids;
}

Environment::FireState Environment::fireStateAt(int f, int t_abs) const {
    FireState s;
    if (f < 0 || f >= (int)fires_.size()) return s;

    const auto& F = fires_[f];
    if (!F.active) { s.extinguished = true; return s; }

    // Clamp index to last known sample (or change to return {-1,-1,true} if you prefer)
    int idx = std::min(t_abs, (int)F.xs.size() - 1);
    if (idx < 0) { s.extinguished = true; return s; }

    s.x = F.xs[idx];
    s.y = F.ys[idx];

    // extinction-aware
    if (F.time_of_extinguish >= 0 && t_abs >= F.time_of_extinguish)
        s.extinguished = true;
    else
        s.extinguished = false;

    return s;
}

bool Environment::fireActiveAt(int f, int t_abs) const {
    auto s = fireStateAt(f, t_abs);
    return (s.x >= 0 && s.y >= 0 && !s.extinguished);
}

// Generalize CAT reservation to all fires
void Environment::reserveFireCAT(std::vector<std::unordered_set<int>>& vertex_cat,
                                 int t0, int t1) const
{
    if (domain_ != DomainType::ForestFire) return;
    for (int t = std::max(0, t0); t <= t1; ++t) {
        if ((int)vertex_cat.size() <= t) vertex_cat.resize(t+1);
        for (int f = 0; f < (int)fires_.size(); ++f) {
            if (!fireActiveAt(f, t)) continue;
            int v = fireVidAt(f, t);
            if (v >= 0) vertex_cat[t].insert(v);
        }
    }
}


void Environment::clearFireCache() const {
    cache_fire_vids_.clear();
    cache_fire_edges_.clear();
}

bool Environment::isOnFireVertex(int t_abs, int vid) const {
    if (domain_ != DomainType::ForestFire) return false;

    auto it = cache_fire_vids_.find(t_abs);
    if (it == cache_fire_vids_.end()) {
        std::unordered_set<int> S;
        for (int f = 0; f < (int)fires_.size(); ++f) {
            if (!fireActiveAt(f, t_abs)) continue;
            auto s = fireStateAt(f, t_abs);
            int fv = nodeIdAt(s.x, s.y);
            if (fv >= 0) S.insert(fv);
        }
        it = cache_fire_vids_.emplace(t_abs, std::move(S)).first;
    }
    return it->second.count(vid) > 0;
}

bool Environment::isOnFireEdge(int t_abs, int u_vid, int v_vid) const {
    // fire traversing an edge between t_abs-1 -> t_abs blocks that edge (both directions)
    if (domain_ != DomainType::ForestFire || t_abs <= 0) return false;

    auto it = cache_fire_edges_.find(t_abs);
    if (it == cache_fire_edges_.end()) {
        std::unordered_set<long long> S;

        for (int f = 0; f < (int)fires_.size(); ++f) {
            if (!fireActiveAt(f, t_abs) || !fireActiveAt(f, t_abs - 1)) continue;

            auto s0 = fireStateAt(f, t_abs - 1);
            auto s1 = fireStateAt(f, t_abs);
            int a = nodeIdAt(s0.x, s0.y);
            int b = nodeIdAt(s1.x, s1.y);
            if (a >= 0 && b >= 0 && a != b) {
                S.insert(packEdge(a, b));
                S.insert(packEdge(b, a));
            }
        }
        it = cache_fire_edges_.emplace(t_abs, std::move(S)).first;
    }
    return it->second.count(packEdge(u_vid, v_vid)) > 0;
}


bool Environment::isStaticObstacle(int x, int y) const {
    if (!isWithinBounds(x,y)) return true;
    int vid = nodeIdAt(x,y);
    if (vid < 0) return true;
    const auto& n = node(vid);
    return (n.type == '@');
}

bool Environment::isCellFreeForPlanner(int x, int y, int t) const {
    if (!isWithinBounds(x,y)) return false;
    int vid = nodeIdAt(x,y);
    if (vid < 0) return false;

    // Static obstacle?
    if (node(vid).type == '@') return false;

    // Domain dynamic hazards (e.g., ForestFire)
    if (!isVertexFreeAtTime(vid, t)) return false;

    // Optional ad-hoc planner blocklist
    auto it = planner_block_vids_.find(t);
    if (it != planner_block_vids_.end() && it->second.count(vid)) return false;

    return true;
}

bool Environment::isEdgeFreeForPlanner(int x1, int y1, int x2, int y2, int t) const {
    int u = nodeIdAt(x1,y1);
    int v = nodeIdAt(x2,y2);
    if (u < 0 || v < 0) return false;

    // Domain dynamic hazards (e.g., ForestFire edge transitions)
    if (isOnFireEdge(t, u, v)) return false;

    // Generic domain rule, if implemented by you
    if (blocksAgentMoveAt(u, v, t)) return false;

    // Optional ad-hoc planner blocklist
    auto it = planner_block_edges_.find(t);
    if (it != planner_block_edges_.end() && it->second.count(packEdge(u,v))) return false;

    return true;
}

void Environment::plannerBlockCell(int x, int y, int t) const {
    int vid = nodeIdAt(x,y);
    if (vid < 0) return;
    planner_block_vids_[t].insert(vid);
}

void Environment::plannerUnblockCell(int x, int y, int t) const {
    int vid = nodeIdAt(x,y);
    if (vid < 0) return;
    auto it = planner_block_vids_.find(t);
    if (it != planner_block_vids_.end()) it->second.erase(vid);
}

void Environment::plannerBlockEdge(int x1, int y1, int x2, int y2, int t) const {
    int u = nodeIdAt(x1,y1), v = nodeIdAt(x2,y2);
    if (u < 0 || v < 0) return;
    planner_block_edges_[t].insert(packEdge(u,v));
}

void Environment::plannerUnblockEdge(int x1, int y1, int x2, int y2, int t) const {
    int u = nodeIdAt(x1,y1), v = nodeIdAt(x2,y2);
    if (u < 0 || v < 0) return;
    auto it = planner_block_edges_.find(t);
    if (it != planner_block_edges_.end()) it->second.erase(packEdge(u,v));
}

void Environment::plannerClearBlocks() const {
    planner_block_vids_.clear();
    planner_block_edges_.clear();
}

const macussp::DomainObjectiveGrids& Environment::ensureDomainObjectiveGrids() const {
    if (!domain_costs_ready_) {
        domain_objective_grids_ = macussp::load_domain_objective_grids(mapName);
        domain_costs_ready_ = true;
    }
    return domain_objective_grids_;
}
