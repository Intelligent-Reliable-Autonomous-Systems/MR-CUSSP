#include "pipeline/ArviStage1Planner.h"
#include "pipeline/Stage1Loop.h"

namespace macussp {

class ArviStage1Planner : public Stage1Planner {
public:
    std::string name() const override { return "arvi"; }
    Stage1Result run(const Stage1Input& input) override {
        return run_belief_collapse_stage1(input.env, input.agents, "ARVI", input.deadline, input.seed);
    }
};

std::unique_ptr<Stage1Planner> make_arvi_stage1_planner() {
    return std::make_unique<ArviStage1Planner>();
}

}  // namespace macussp
