#pragma once

#include "pipeline/PipelineTypes.h"
#include <memory>

namespace macussp {
std::unique_ptr<Stage1Planner> make_arvi_stage1_planner();
}  // namespace macussp
