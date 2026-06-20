#include "ThetaTable.h"

#include <unordered_map>

namespace macussp {

namespace {

using Table = std::unordered_map<int, ObjectiveOrdering>;

ObjectiveOrdering salp_theta(int id) {
    static const Table kTable{
        {1, {0, 1, 2}},  // c1: oe > oc > ot
        {2, {1, 0, 2}},  // c2: oc > oe > ot
        {3, {2, 0, 1}},  // c3: ot > oe > oc
    };
    auto it = kTable.find(id);
    if (it != kTable.end()) return it->second;
    return identity_ordering(3);
}

ObjectiveOrdering warehouse_theta(int id) {
    static const Table kTable{
        {1, {0, 1, 2}},  // c1: ot > oc > oh
        {2, {1, 0, 2}},  // c2: oc > ot > oh
        {3, {2, 1, 0}},  // c3: oh > oc > ot
    };
    auto it = kTable.find(id);
    if (it != kTable.end()) return it->second;
    return identity_ordering(3);
}

ObjectiveOrdering forest_theta(int id) {
    static const Table kTable{
        {1, {1, 2, 0}},  // c1: oe > ol > ot
        {2, {2, 1, 0}},  // c2: ol > oe > ot
        {3, {0, 1, 2}},  // c3: ot > oe > ol
    };
    auto it = kTable.find(id);
    if (it != kTable.end()) return it->second;
    return identity_ordering(3);
}

}  // namespace

ObjectiveOrdering theta_for_context(const std::string& domain, int context_id) {
    if (domain == "salp") return salp_theta(context_id);
    if (domain == "warehouse") return warehouse_theta(context_id);
    if (domain == "forestfire") return forest_theta(context_id);
    return identity_ordering(3);
}

std::set<int> viable_true_context_ids(const std::string& domain) {
    (void)domain;
    // Landmark subsets only eliminate to singletons over {0..4}; paper contexts are ids 1..3.
    return {1, 2, 3};
}

bool is_viable_true_context(const std::string& domain, int context_id) {
    return viable_true_context_ids(domain).count(context_id) > 0;
}

}  // namespace macussp
