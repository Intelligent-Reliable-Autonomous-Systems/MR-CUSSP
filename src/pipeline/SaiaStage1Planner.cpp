#include "pipeline/SaiaStage1Planner.h"
#include "pipeline/Stage1Loop.h"

namespace macussp {

class SaiaStage1Planner : public Stage1Planner {
public:
    std::string name() const override { return "saia"; }
    Stage1Result run(const Stage1Input& input) override {
        return run_belief_collapse_stage1(input.env, input.agents, "SAIA", input.deadline, input.seed);
    }
};

std::unique_ptr<Stage1Planner> make_saia_stage1_planner() {
    return std::make_unique<SaiaStage1Planner>();
}

}  // namespace macussp
