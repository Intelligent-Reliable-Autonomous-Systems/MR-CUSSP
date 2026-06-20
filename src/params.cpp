#include "params.h"

// Default values (you may override them in main() by assignment)
std::set<int> PossibleContexts = {0,1,2,3,4};
int TrueContext = 2;
int TIME_LIMIT = 1000; // in seconds (legacy default; pipeline uses --time_budget instead)
int g_active_mapf_time_limit_sec = 0;

int FIRE_HMAX        = 300;  // horizon used when evaluating fire intercept
float FIRE_SPEED_FACTOR = 0.15 ;   // fire relative speed factor (1.0 = same speed as agents, 0.5 = half speed, 0.1 = one tenth speed, etc.)

int  SALP_SUBRES         = 1;
bool SALP_ALLOW_DIAGONAL = true;