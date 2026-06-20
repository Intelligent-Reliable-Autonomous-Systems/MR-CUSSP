#include "continuous.h"
#include "Definitions.h"  // your Edge
#include <algorithm>

static inline CPoint center_of_cell(int r, int c) { return {r+0.5, c+0.5}; }

// --- 1) Build continuous domain from your ASCII .map + cost rasters ---
ContinuousDomain build_continuous_from_raster(
    const std::vector<std::string>& rows,
    const std::vector<std::vector<double>>& cost_grids)
{
    ContinuousDomain D;
    D.H = (int)rows.size();
    D.W = (int)rows.front().size();
    // obstacles: rectangle per occupied cell
    for (int r=0;r<D.H;r++){
        for (int c=0;c<D.W;c++){
            char ch = rows[r][c];
            if (ch=='@' || ch=='T') {        // blocked
                Polygon rect = {{(double)r,(double)c},{(double)r,(double)c+1},
                                {(double)r+1,(double)c+1},{(double)r+1,(double)c}};
                D.obstacles.push_back(std::move(rect));
            } else if (ch=='L') {
                D.landmarks.push_back(center_of_cell(r,c));
            }
        }
    }
    // costs
    int dim = (int)cost_grids.size();
    D.costs.resize(dim);
    for (int d=0; d<dim; ++d){
        D.costs[d].H = D.H; D.costs[d].W = D.W;
        D.costs[d].data = cost_grids[d];     // row-major H*W
    }
    return D;
}

// --- helpers for lattice construction ---
static bool free_cell(const ContinuousDomain& D, int r, int c, const std::vector<std::string>& rows){
    char ch = rows[r][c];
    return !(ch=='@' || ch=='T');
}

static inline double seglen_grid_units(int dr, int dc, int R){
    // length in coarse-cell units
    double lr = (double)dr / R, lc = (double)dc / R;
    return std::sqrt(lr*lr + lc*lc);
}

// Assign line-integral cost to an edge by sampling along its segment
static void assign_edge_costs_line_integral(
    const ContinuousDomain& D, Map& map, std::vector<Edge>& edges,
    int R, int num_objectives)
{
    // build inverse id -> refined (r,c)
    std::vector<std::pair<int,int>> inv(map.graph_size);
    for (int r=0;r<map.height;r++){
        for (int c=0;c<map.width;c++){
            size_t id = map.getID(r,c);
            if (id!=(size_t)-1) inv[id] = {r,c};
        }
    }
    const int SAMPLES = 3;
    const double SCALE = 1000.0; // keep integer costs

    for (auto& e : edges){
        // self-loop stays zero
        if (e.source == e.target) { for(int d=0; d<num_objectives; ++d) e.cost[d]=0; continue; }

        auto [sr,sc] = inv[e.source];
        auto [tr,tc] = inv[e.target];

        // endpoints in coarse units:
        double sx = (sr + 0.5) / R, sy = (sc + 0.5) / R;
        double tx = (tr + 0.5) / R, ty = (tc + 0.5) / R;

        double L = std::hypot(tx - sx, ty - sy);  // in coarse-cell units

        for (int d=0; d<num_objectives; ++d){
            double acc = 0.0;
            for (int k=1; k<=SAMPLES; ++k){
                double t = (k)/(double)(SAMPLES+1);
                double x = sx + t*(tx - sx);
                double y = sy + t*(ty - sy);
                acc += D.costs[d].sample(x, y);
            }
            double avg = acc / SAMPLES;
            size_t integ = (size_t) std::llround(avg * L * SCALE);
            e.cost[d] = integ;
        }
    }
}

// --- 2) Discretize to refined octile lattice ---
void discretize_lattice_from_continuous(const ContinuousDomain& D, int R, bool allow_diag, Map& out_map, std::vector<Edge>& out_edges, int num_objectives)
{
    out_map.setSize(D.W*R, D.H*R);      // NOTE: Map uses (height,width) ordered as (rows,cols)
    size_t gid=0;
    // mark refined free/blocked by lifting coarse occupancy
    for (int r=0;r<D.H;r++){
        for (int c=0;c<D.W;c++){
            bool free = true; // coarse free if not obstacle rect
            // with raster origin, free if not '@'/'T'; assume obstacles already encoded in rows
            int val = free ? 0 : -1;
            for (int sr=0; sr<R; ++sr){
                for (int sc=0; sc<R; ++sc){
                    int rr = r*R + sr, cc = c*R + sc;
                    out_map.setVal(rr,cc,val);
                    if (val==0) { out_map.setID(rr,cc,gid++); } else { out_map.setID(rr,cc,-1); }
                }
            }
        }
    }
    out_map.graph_size = gid;

    auto push_edge=[&](int rr,int cc,int nr,int nc){
        if (nr<0||nr>=out_map.height||nc<0||nc>=out_map.width) return;
        if (out_map.getVal(nr,nc)==-1 || out_map.getVal(rr,cc)==-1) return;
        out_edges.emplace_back(out_map.getID(nr,nc), out_map.getID(rr,cc), std::vector<size_t>(num_objectives,0));
    };

    // 4-neighbors
    static const int dx4[4]={1,-1,0,0}, dy4[4]={0,0,1,-1};
    // diagonals (with corner-cut prevention)
    static const int dx8[4]={1,1,-1,-1}, dy8[4]={1,-1,1,-1};
    for (int rr=0; rr<out_map.height; ++rr){
        for (int cc=0; cc<out_map.width; ++cc){
            if (out_map.getVal(rr,cc)==-1) continue;
            out_edges.emplace_back(out_map.getID(rr,cc), out_map.getID(rr,cc), std::vector<size_t>(num_objectives,0));
            for (int k=0;k<4;k++) push_edge(rr,cc, rr+dx4[k], cc+dy4[k]);
            if (allow_diag){
                for (int k=0;k<4;k++){
                    int nr=rr+dx8[k], nc=cc+dy8[k];
                    // corner cutting check
                    int br=rr, bc=nc; int cr=nr, cc2=cc;
                    if (nr>=0&&nr<out_map.height&&nc>=0&&nc<out_map.width
                        && out_map.getVal(br,bc)==0 && out_map.getVal(cr,cc2)==0){
                        push_edge(rr,cc,nr,nc);
                    }
                }
            }
        }
    }

    // line-integral costs along edges using the continuous fields
    assign_edge_costs_line_integral(D, out_map, out_edges, R, num_objectives);
}
