#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <MAP.h>

struct CPoint { double x, y; };          // continuous (row=x, col=y) in grid units
using Polygon = std::vector<CPoint>;
template <typename T>
const T& clamp(const T& value, const T& low, const T& high) {
    return std::max(low, std::min(value, high));
}
struct ContinuousField {
    // coarse raster HxW; values interpreted as piecewise-constant per cell,
    // but sampled with bilinear to be continuous.
    int H=0, W=0;                         // coarse map size
    std::vector<double> data;             // row-major size H*W

    double sample(double x, double y) const { // x=row, y=col in [0,H)×[0,W)
        if (H==0 || W==0) return 0.0;
        x = clamp(x, 0.0, (double)H-1.000001);
        y = clamp(y, 0.0, (double)W-1.000001);
        int i = (int)std::floor(x), j = (int)std::floor(y);
        double fx = x - i, fy = y - j;
        auto at=[&](int r,int c){ return data[r*W + c]; };
        double v00=at(i,j), v01=at(i,j+1), v10=at(i+1,j), v11=at(i+1,j+1);
        return (1-fx)*((1-fy)*v00 + fy*v01) + fx*((1-fy)*v10 + fy*v11);
    }
};

struct ContinuousDomain {
    int H=0, W=0;                    // coarse raster size (from .map)
    double cell = 1.0;               // meters per coarse cell (optional, default 1)
    std::vector<Polygon> obstacles;  // axis-aligned rects from occupied cells
    std::vector<CPoint> landmarks;   // centers of 'L' cells
    std::vector<ContinuousField> costs; // one per objective
};

// Build ContinuousDomain from your existing rasters
ContinuousDomain build_continuous_from_raster(
    const std::vector<std::string>& map_rows,
    const std::vector<std::vector<double>>& cost_grids // [dim][H*W]
);

// Discretize to a refined 8-connected lattice and fill Map + Edges (your types)
struct Edge;
void discretize_lattice_from_continuous(
    const ContinuousDomain& dom, int R, bool allow_diag,
    Map& out_map, std::vector<Edge>& out_edges, int num_objectives
);
