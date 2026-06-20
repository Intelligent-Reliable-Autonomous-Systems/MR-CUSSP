#include "pipeline/CimopStage1Planner.h"
#include "pipeline/Stage1Loop.h"

namespace macussp {

class CimopStage1Planner : public Stage1Planner {
public:
    std::string name() const override { return "cimop"; }
    Stage1Result run(const Stage1Input& input) override {
        return run_belief_collapse_stage1(input.env, input.agents, "OURS", input.deadline, input.seed);
    }
};

std::unique_ptr<Stage1Planner> make_cimop_stage1_planner() {
    return std::make_unique<CimopStage1Planner>();
}

}  // namespace macussp
