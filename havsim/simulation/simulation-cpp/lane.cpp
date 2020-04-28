
#include "lane.h"
#include "vehicle.h"

Lane::Lane(LaneMethod lane_method) : lane_method_(lane_method) {}


double Lane::call_downstream(Vehicle *vehicle, int timeind, double dt) {
    switch (lane_method_) {
        case SPEED:
            return (speed_fun(timeind) - vehicle->get_speed()) / dt;
        default:
            return -1;
    }
}

