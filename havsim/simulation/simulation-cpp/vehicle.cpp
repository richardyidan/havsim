#include "vehicle.h"
#include "lane.h"

#include <cmath>

Vehicle::Vehicle(int vehid,
                 Lane *lane,
                 std::vector<double> cf_parameters,
                 std::vector<double> lc_parameters,
                 double relaxp,
                 double length,
                 double check_lc) :
        vehid_(vehid),
        lane_(lane),
        cf_parameters_(std::move(cf_parameters)),
        lc_parameters_(std::move(lc_parameters)),
        relaxp_(relaxp),
        length_(length),
        lead_(nullptr),
        fol_(nullptr),
        lfol_(nullptr),
        rfol_(nullptr),
        llead_(nullptr),
        rlead_(nullptr),
        pos_(0),
        speed_(0),
        headway_(0),
        action_(0),
        inittime_(-1),
        endtime_(-1),
        in_relax_(false),
        relax_start_(0.0),
        check_lc_(check_lc) {}


double Vehicle::call_cf(int timeind, double dt) {
    double acc;
    if (!lead_) {
        acc = lane_->call_downstream(this, timeind, dt);
    } else {
        double headway = headway_;
        if (in_relax_) {
            headway_ += relax_[timeind - relax_start_];
        }
        std::vector<double> state = {headway_, speed_, lead_->speed_};
        acc = IDM(state);
    }
    // controls [lower, upper] bounds on acceleration
    double lower = -7, upper = 3;
    acc = std::min(std::max(acc, lower), upper);
    if (speed_ + dt * acc < 0) {
        acc = -speed_ / dt;
    }
    return acc;
}
double Vehicle::get_speed() {
    return speed_;
}
double Vehicle::IDM(const std::vector<double> &state) {
// state = headway, velocity, lead velocity
// p = parameters
// returns acceleration
    return cf_parameters_[3] * (1 - std::pow(state[1] / cf_parameters_[0], 4) - std::pow(
            ((cf_parameters_[2] + state[1] * cf_parameters_[1] +
              (state[1] * (state[1] - state[2])) / (2 * std::sqrt(cf_parameters_[3] * cf_parameters_[4]))) /
             (state[0])), 2));
}


