#include "vehicle.h"
#include "lane.h"
#include <cmath>

Vehicle::Vehicle(int vehid,
                 Lane *lane,
                 std::vector<double> cf_parameters,
                 std::vector<double> lc_parameters,
                 double length,
                 double relaxp) :
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
        pos_(-1),
        speed_(-1),
        headway_(-1),
        action_(-1),
        inittime_(-1),
        endtime_(-1),
        in_relax_(false),
        relax_start_(-1) {}
Vehicle* Vehicle::get_lead() const {
    return lead_;
}
void Vehicle::initialize(double pos,
                         double speed,
                         double headway,
                         int inittime,
                         Vehicle *lead,
                         Vehicle *fol,
                         Vehicle *lfol,
                         Vehicle *rfol,
                         Vehicle *llead,
                         Vehicle *rlead) {
    pos_ = pos;
    speed_ = speed;
    headway_ = headway;
    inittime_ = inittime;
    lead_ = lead;
    fol_ = fol;
    lfol_ = lfol;
    rfol_ = rfol;
    llead_ = llead;
    rlead_ = rlead;
    leadmem_.emplace_back(lead_,inittime_);
    lanemem_.emplace_back(lane_,inittime_);
    posmem_.push_back(pos_);
    speedmem_.push_back(speed_);
}


void Vehicle::call_cf(int timeind, double dt) {
    double acc;
    if (!lead_) {
        acc = lane_->call_downstream(this, timeind, dt);
    } else {

        double headway = headway_;
        if (in_relax_) {
            headway += relax_[timeind - relax_start_];
        }
        std::vector<double> state = {headway, speed_, lead_->speed_};
        acc = IDM(cf_parameters_,state);

    }
    // controls [lower, upper] bounds on acceleration
    const double lower = -7, upper = 3;
    acc = std::min(std::max(acc, lower), upper);
    if (speed_ + dt * acc < 0) {
        acc = -speed_ / dt;
    }

    action_ = acc;

}

double Vehicle::get_speed() const {
    return speed_;
}

double Vehicle::IDM(const std::vector<double>& p, const std::vector<double> &state) {
// state = headway, velocity, lead velocity
// p = parameters
// returns acceleration
    double t5=2*std::sqrt(p[3]*p[4]);
    double t4=state[1]*(state[1]-state[2]);
    double t3=p[2]+state[1]*p[1]+t4/t5;
    double t2=t3/state[0];
    double t1=1-std::pow(state[1]/p[0],4)-std::pow(t2,2);
    return p[3]*t1;
}



void Vehicle::update(int timeind, double dt){

    double temp=action_*dt;
    pos_+=speed_*dt+.5*temp*dt;
    speed_+=temp;

    posmem_.push_back(pos_);
    speedmem_.push_back(speed_);
    if(in_relax_ && timeind==relax_start_+relax_.size()-1){
        in_relax_= false;
        relaxmem_.push_back({{relax_start_,timeind},relax_});
    }
}

void Vehicle::update_headway(){
    if(lead_){
        headway_=Lane::get_headway(this, lead_);
    }
}

double Vehicle::get_pos() const{
    return pos_;
}

double Vehicle::get_len() const{
    return length_;
}

std::vector<double> Vehicle::get_posmem() const{
    return posmem_;
}

int Vehicle::get_vehid() const{
    return vehid_;
}

