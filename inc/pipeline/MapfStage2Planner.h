#pragma once

#include "pipeline/PipelineTypes.h"
#include <memory>
#include <string>

namespace macussp {

std::unique_ptr<Stage2Planner> make_mapf_stage2_planner(const std::string& algorithm);

}  // namespace macussp
