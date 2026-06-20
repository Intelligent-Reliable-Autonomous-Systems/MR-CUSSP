#ifndef PARAMS_H
#define PARAMS_H

#include <set>

extern int TIME_LIMIT;
// When > 0, overrides TIME_LIMIT for formation Phase-1 MAPF (pipeline budget path).
extern int g_active_mapf_time_limit_sec;
extern std::set<int> PossibleContexts;
extern int TrueContext;

extern int FIRE_HMAX ;  // horizon used when evaluating fire intercept
extern float FIRE_SPEED_FACTOR;   // fire relative speed factor (1.0 = same speed as agents, 0.5 = half speed, 0.1 = one tenth speed, etc.)

extern int  SALP_SUBRES;         // e.g., 1 (off), 2, 3, 4
extern bool SALP_ALLOW_DIAGONAL; // default true

#endif // PARAMS_H