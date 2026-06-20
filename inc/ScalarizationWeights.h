#pragma once

#include "DomainCosts.h"
#include "environment.h"

#include <array>
#include <string>
#include <vector>

namespace macussp {

constexpr int kScalarizationSamples = 50;

// Geometric scaling weights w_j = (prod M)^(1/d) / M_j from paper-style normalization.
std::vector<double> geometric_scaling_weights(const std::vector<double>& M);

// Estimate per-domain M once from random vertex-pair shortest-path cost samples.
std::vector<double> estimate_domain_M(const Environment& env,
                                      const DomainObjectiveGrids& grids,
                                      int samples = kScalarizationSamples);

const std::vector<double>& cached_domain_M(const std::string& domain,
                                           const Environment& env);

}  // namespace macussp
