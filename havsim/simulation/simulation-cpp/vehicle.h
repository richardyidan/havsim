#ifndef HAVSIM_CPP_VEHICLE_H
#define HAVSIM_CPP_VEHICLE_H

#include <vector>
class Lane;

class Vehicle{
public:
    Vehicle(int vehid,
            Lane* lane,
            std::vector<double>  cf_parameters,
            std::vector<double> lc_parameters,
            double relaxp,
            double length=2,
            double check_lc=0.25
            );

    // car following model: returns the action (acceleration by default)
    virtual double call_cf(int timeind, double dt);
    virtual double IDM(const std::vector<double>& state);
    double get_speed();

private:
    // id of the vehicle
    int vehid_;
    Lane* lane_;
    //TODO: road attribute


    // ====== Model parameters =====
    std::vector<double> cf_parameters_;
    std::vector<double> lc_parameters_;
    double relaxp_;
    double length_;

    // Leader/follower relationships
    Vehicle* lead_;
    Vehicle* fol_;
    Vehicle* lfol_;
    Vehicle* rfol_;
    Vehicle* llead_;
    Vehicle* rlead_;

    // State
    double pos_;
    double speed_;
    double headway_;
    double action_;

    // Memory
    int inittime_;
    int endtime_;
    std::vector<Vehicle*> leadmem_;
    std::vector<Lane*> lanemem_;
    std::vector<double> posmem_;
    std::vector<double> speedmem_;
    std::vector<double> relaxmem_;

    bool in_relax_;
    std::vector<double> relax_;
    double relax_start_;
    double check_lc_;

};

#endif //HAVSIM_CPP_VEHICLE_H
